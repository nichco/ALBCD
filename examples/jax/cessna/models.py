"""Cessna 182-like wing setup and the aerodynamic (VLM) and structural (beam)
models, shared by cessna.py and monolithic.py so both solve exactly the same problem.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # modopt and the models work in float64
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from beam_jax import Beam, CSTube
from vlm_jax import aerodynamics as vlm_aero
from vlm_jax import geometry as vlm_geom
from vlm_jax.analysis import solve_aero
from vlm_jax.bsplines import get_bspline_mtx
from vlm_jax.meshing import generate_mesh

# aero setup
b = 11.0                      # span (m)
c_root = 1.5                  # root chord (m)
c_tip = 1.0                   # tip chord (m)
v_inf = 62.0                  # freestream velocity (m/s)
rho_atm = 1.0                 # density (kg/m^3)
q = 0.5 * rho_atm * v_inf**2  # dynamic pressure (Pa)

# Full-span VLM mesh whose 41 spanwise stations double as the beam nodes. Like the
# Cessna 182 wing, the chord is constant over the inboard half of each semispan
# and tapers linearly from there to the tip.
mesh0 = generate_mesh({"num_x": 8, "num_y": 41, "wing_type": "rect", "span": b,
                       "root_chord": c_root, "taper_ratio": c_tip / c_root,
                       "symmetry": False, "taper_break_fraction": 0.5, "sweep": 0})
mesh0_jnp = jnp.asarray(mesh0)
y = mesh0[0, :, 1]   # spanwise station coordinates

# structures setup
num_nodes = len(y)
num_elems = num_nodes - 1
struct_mesh = np.zeros((num_nodes, 3))
struct_mesh[:, 1] = y
fixed_nodes = [num_nodes // 2] # the center node is fixed in all DOFs

_, _, chords0, _, _, _, _ = vlm_aero.vlm_geometry(mesh0_jnp, symmetry=False)
t_over_c = 0.1   # airfoil thickness-to-chord ratio, sets the spar tube radius
r = t_over_c * (chords0[:-1] + chords0[1:]) / 4
E = 69e9
G = 26e9
rho_mat = 3000
m0 = 1e3
load_factor = 5
safety_factor = 1.5
sigma_yield_mpa = 350.0
rho_ks = 2e-1         # KS aggregation parameter for the max stress (1/MPa)
min_gauge = 0.001     # minimum wall thickness (m)
rho_ks_thick = 100.0  # KS aggregation parameter for the min thickness (1/mm)

# design variables: B-spline control points for the twist (rad) and the wall thickness (m)
num_cp_twist = 7
num_cp_thickness = 10
bspline_twist = jnp.asarray(get_bspline_mtx(num_cp_twist, np.linspace(0, 1, num_nodes)))
bspline_thickness = jnp.asarray(get_bspline_mtx(num_cp_thickness, np.linspace(0, 1, num_elems)))

twist_cp0 = np.ones(num_cp_twist) * np.deg2rad(3)
thickness_cp0 = np.concatenate([np.linspace(min_gauge, 0.01, num_cp_thickness // 2),
                                np.linspace(0.01, min_gauge, num_cp_thickness // 2)])


def aero_model(twist_cp):
    """Return the drag coefficient, the nodal aero loads [drag, lift] and the total lift."""
    twist = bspline_twist @ twist_cp
    def_mesh = vlm_geom.apply_twist_rotation(mesh0_jnp, jnp.rad2deg(twist),
                                             ref_axis_pos=0.25, symmetry=False, rotate_x=True)
    out = solve_aero(def_mesh, v_inf, rho_atm, symmetry=False, with_viscous=False)
    # (nx-1, ny-1, 3) panel forces -> one net force per spanwise strip -> nodal.
    # Keep drag (x) and lift (z) separately, stacked into one flat vector.
    strip_forces = vlm_aero.sectional_forces(out["sec_forces"])
    nodal_forces = vlm_aero.lump_to_nodes(strip_forces)   # (num_nodes, 3)
    aero_loads = jnp.concatenate([nodal_forces[:, 0], nodal_forces[:, 2]])
    lift = out["CL"] * q * out["S_ref"]

    return out["CD"], aero_loads, lift


def structures_model(aero_loads, thickness_cp):
    """Return the KS max von Mises stress (MPa), the KS min wall thickness (mm) and the total weight."""
    thickness = bspline_thickness @ thickness_cp

    F = jnp.zeros((num_nodes, 6))
    F = F.at[:, 0].set(aero_loads[:num_nodes] * load_factor * safety_factor)  # drag (chordwise)
    F = F.at[:, 2].set(aero_loads[num_nodes:] * load_factor * safety_factor)  # lift (vertical)

    cs = CSTube(radius=r, thickness=thickness)
    beam = Beam(mesh=struct_mesh, E=E, G=G, rho=rho_mat, cs=cs, F=F, fixed_nodes=fixed_nodes)
    u = beam.solve()

    # logsumexp(rho*x)/rho over-estimates max(x), and -logsumexp(-rho*x)/rho
    # under-estimates min(x), so constraining either one is conservative
    sigma_mpa = beam.recover_stress(u) / 1e6
    max_sigma_mpa = logsumexp(rho_ks * sigma_mpa) / rho_ks
    min_thickness_mm = -logsumexp(-rho_ks_thick * thickness * 1e3) / rho_ks_thick

    weight = (beam.mass + m0) * 9.81

    return max_sigma_mpa, min_thickness_mm, weight
