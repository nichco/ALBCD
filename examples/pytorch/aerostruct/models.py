"""Wing setup and the aerodynamic (VLM) and structural (beam) models, shared by
aerostruct.py and monolithic.py so both solve exactly the same problem.
"""

import numpy as np
import torch
torch.set_default_dtype(torch.float64)  # modopt and the models work in float64

from beam_torch import Beam, CSTube
from vlm_torch import aerodynamics as vlm_aero
from vlm_torch import geometry as vlm_geom
from vlm_torch.analysis import solve_aero
from vlm_torch.bsplines import get_bspline_mtx
from vlm_torch.meshing import generate_mesh

# aero setup
b = 10.0                      # span (m)
c_root = 1.0                  # root chord (m)
c_tip = 0.65                  # tip chord (m)
v_inf = 60.0                  # freestream velocity (m/s)
rho_atm = 1.225               # density (kg/m^3)
q = 0.5 * rho_atm * v_inf**2  # dynamic pressure (Pa)

# Full-span VLM mesh whose 31 spanwise stations double as the beam nodes.
# Cosine spanwise spacing clusters the stations toward the tips.
mesh0 = generate_mesh({"num_x": 5, "num_y": 31, "wing_type": "rect", "span": b,
                       "root_chord": c_root, "taper_ratio": c_tip / c_root,
                       "span_cos_spacing": 1.0, "symmetry": False})
mesh0_torch = torch.as_tensor(mesh0)
y = mesh0[0, :, 1]   # spanwise station coordinates

# structures setup
num_nodes = len(y)
num_elems = num_nodes - 1
struct_mesh = np.zeros((num_nodes, 3))
struct_mesh[:, 1] = y
fixed_nodes = [num_nodes // 2] # the center node is fixed in all DOFs

_, _, chords0, _, _, _, _ = vlm_aero.vlm_geometry(mesh0_torch, symmetry=False)
r = 0.2 * (chords0[:-1] + chords0[1:]) / 4 # radius of the tube as a function of chord
E = 69e9
G = 26e9
rho_mat = 3000
m0 = 1e3
load_factor = 3
safety_factor = 1.5
tip_disp_target = 0.1
min_gauge = 0.001   # minimum wall thickness (m)

# design variables: B-spline control points for the twist (rad) and the wall thickness (m)
num_cp_twist = 7
num_cp_thickness = 10
bspline_twist = torch.as_tensor(get_bspline_mtx(num_cp_twist, np.linspace(0, 1, num_nodes)))
bspline_thickness = torch.as_tensor(get_bspline_mtx(num_cp_thickness, np.linspace(0, 1, num_elems)))

twist_cp0 = np.ones(num_cp_twist) * np.deg2rad(5)
thickness_cp0 = np.ones(num_cp_thickness) * 0.002


def aero_model(twist_cp):
    """Return the drag coefficient, the nodal aero loads [drag, lift] and the total lift."""
    twist = bspline_twist @ twist_cp
    def_mesh = vlm_geom.apply_twist_rotation(mesh0_torch, torch.rad2deg(twist),
                                             ref_axis_pos=0.25, symmetry=False, rotate_x=True)
    out = solve_aero(def_mesh, v_inf, rho_atm, symmetry=False, with_viscous=False)
    # (nx-1, ny-1, 3) panel forces -> one net force per spanwise strip -> nodal.
    # Keep drag (x) and lift (z) separately, stacked into one flat vector.
    strip_forces = vlm_aero.sectional_forces(out["sec_forces"])
    nodal_forces = vlm_aero.lump_to_nodes(strip_forces)   # (num_nodes, 3)
    aero_loads = torch.cat([nodal_forces[:, 0], nodal_forces[:, 2]])
    lift = out["CL"] * q * out["S_ref"]

    return out["CD"], aero_loads, lift


def structures_model(aero_loads, thickness_cp):
    """Return the right and left tip displacements and the total weight."""
    thickness = bspline_thickness @ thickness_cp

    F = torch.zeros((num_nodes, 6))
    F[:, 0] = aero_loads[:num_nodes] * load_factor * safety_factor  # drag (chordwise)
    F[:, 2] = aero_loads[num_nodes:] * load_factor * safety_factor  # lift (vertical)

    cs = CSTube(radius=r, thickness=thickness)
    beam = Beam(mesh=struct_mesh, E=E, G=G, rho=rho_mat, cs=cs, F=F, fixed_nodes=fixed_nodes)
    u = torch.linalg.norm(beam.solve()[:, :3], dim=1)
    weight = (beam.mass + m0) * 9.81

    return u[-1], u[0], weight
