"""Cessna 182-like strut-braced wing setup and the aerodynamic (VLM) and structural (jframe)
models, shared by cessna.py and monolithic.py so both solve exactly the same problem.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # modopt and the models work in float64
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from jframe import Beam, CSTube, Frame
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

# Full-span VLM mesh whose 41 spanwise stations double as the spar nodes. Like the
# Cessna 182 wing, the chord is constant over the inboard half of each semispan
# and tapers linearly from there to the tip.
mesh0 = generate_mesh({"num_x": 8, "num_y": 41, "wing_type": "rect", "span": b,
                       "root_chord": c_root, "taper_ratio": c_tip / c_root,
                       "symmetry": False, "taper_break_fraction": 0.5, "sweep": 0})
mesh0_jnp = jnp.asarray(mesh0)
y = mesh0[0, :, 1]   # spanwise station coordinates

# structures setup
# Like the Cessna 182, each wing is pinned to the top of the cabin at its root fittings
# and braced by a strut from the lower fuselage. Each spar is a tube along the root
# quarter chord from its root fitting to the tip; the root fitting holds the spar's
# translations, its twist and its chordwise bending, but lets it rotate in flapwise
# bending, so the strut carries the bending moment. The strut is a tube from the
# lower-fuselage fitting, where it is pinned, to the spar, where it is rigidly joined
# (jframe joints share all 6 dofs). The aero loads inboard of the root fittings go
# straight into the fuselage.
num_nodes = len(y)
num_elems = num_nodes - 1
x_spar = 0.25 * c_root   # chordwise position of the spar (m)
i_root = int(np.argmin(np.abs(y - 0.63)))   # right root fitting: the station nearest the cabin top's 0.63 m half-width
i_strut = int(np.argmin(np.abs(y - 2.6)))   # right strut joint: the station nearest the Cessna 182's, 2.6 m outboard
strut_fitting = np.array([0.21, 0.85, -1.07])  # right strut's lower-fuselage fitting (m), from cessna182_no_wing.stl
num_strut_nodes = 9

# Both spars are numbered in increasing y: the left spar is VLM stations 0..num_nodes-1-i_root
# (its root fitting is its last node) and the right spar is stations i_root..num_nodes-1 (its
# root fitting is its first node). Lists below are ordered [left, right].
spar_nodes = [np.arange(0, num_nodes - i_root), np.arange(i_root, num_nodes)]
spar_roots = [-1, 0]
spar_mesh = np.stack([np.full(num_nodes, x_spar), y, np.zeros(num_nodes)], axis=1)
spar_meshes = [spar_mesh[nodes] for nodes in spar_nodes]
spar_elems = np.concatenate([nodes[:-1] for nodes in spar_nodes])  # spar elements, as VLM strips, in increasing y
num_spar_elems = len(spar_elems)
num_left_elems = len(spar_nodes[0]) - 1
y_spar_elem = 0.5 * (y[spar_elems] + y[spar_elems + 1])
root_dofs = [0, 1, 2, 4, 5]   # all but theta_x, the rotation in flapwise bending

strut_joints = [num_nodes - 1 - i_strut, i_strut - i_root]   # the spar node each strut joins, in that spar's numbering
strut_meshes = [np.linspace(fitting, mesh[joint], num_strut_nodes)
                for fitting, mesh, joint in zip([strut_fitting * [1, -1, 1], strut_fitting], spar_meshes, strut_joints)]
num_struts = len(strut_meshes)

_, _, chords0, _, _, _, _ = vlm_aero.vlm_geometry(mesh0_jnp, symmetry=False)
t_over_c = 0.1   # airfoil thickness-to-chord ratio, sets the spar tube radius
r = t_over_c * (chords0[:-1] + chords0[1:]) / 4   # tube radius of every VLM strip (m)
r_spar = r[spar_elems]
r_strut = 0.025  # strut tube radius (m)
# largest admissible wall thicknesses: the smallest spar tube radius and the strut radius. A thicker wall
# than the radius makes the tube model unphysical (its area pi t (2r - t) shrinks as t grows and goes
# negative past t = 2r), so the thickness variables are bounded above by these (inactive at the optimum)
thickness_max = float(np.min(np.asarray(r_spar)))
strut_thickness_max = r_strut
# smallest admissible wall thickness, for the spar control points and the struts: small but positive,
# since a zero wall has zero area and makes the frame's stiffness matrix singular (an SLSQP iterate
# clipped to t = 0 then stalls the structures block). Far below the 1 mm minimum gauge, and below the
# smallest optimal control point (0.12 mm), so it is inactive at the optimum.
thickness_min = 1e-5
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

# design variables: B-spline control points for the twist (rad) and the spar wall
# thickness (m), and the wall thickness of each strut (m)
num_cp_twist = 7
num_cp_thickness = 10
bspline_twist = jnp.asarray(get_bspline_mtx(num_cp_twist, np.linspace(0, 1, num_nodes)))
bspline_thickness = jnp.asarray(get_bspline_mtx(
    num_cp_thickness, (y_spar_elem - y_spar_elem[0]) / (y_spar_elem[-1] - y_spar_elem[0])))

twist_cp0 = np.ones(num_cp_twist) * np.deg2rad(3)
thickness_cp0 = np.concatenate([np.linspace(min_gauge, 0.01, num_cp_thickness // 2),
                                np.linspace(0.01, min_gauge, num_cp_thickness // 2)])
strut_thickness0 = np.full(num_struts, 0.002)


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


def build_frame(aero_loads, thickness_cp, strut_thickness):
    """The jframe model of the spars and struts, [left spar, right spar, left strut, right strut],
    loaded by the design-load aero loads."""
    thickness = bspline_thickness @ thickness_cp   # spar wall thickness, (num_spar_elems,)

    F = jnp.zeros((num_nodes, 6))
    F = F.at[:, 0].set(aero_loads[:num_nodes] * load_factor * safety_factor)  # drag (chordwise)
    F = F.at[:, 2].set(aero_loads[num_nodes:] * load_factor * safety_factor)  # lift (vertical)

    spars = [Beam(mesh, E, G, rho_mat, CSTube(r_spar[elems], thickness[elems]), F=F[nodes],
                  fixed_nodes=[root], fixed_dofs=root_dofs)
             for mesh, nodes, root, elems in zip(spar_meshes, spar_nodes, spar_roots,
                                                 np.split(np.arange(num_spar_elems), [num_left_elems]))]
    # each strut is pinned at its fitting (translations held, rotations free), its first node
    strut_beams = [Beam(mesh, E, G, rho_mat, CSTube(r_strut, t), fixed_nodes=[0], fixed_dofs=[0, 1, 2])
                   for mesh, t in zip(strut_meshes, strut_thickness)]

    return Frame(spars + strut_beams,
                 joints=[[(spar, joint), (strut, -1)] for spar, joint, strut in zip(spars, strut_joints, strut_beams)])


def structures_model(aero_loads, thickness_cp, strut_thickness):
    """Return the KS max von Mises stress (MPa) in the spars and struts, the KS min wall
    thickness (mm) and the total weight."""
    frame = build_frame(aero_loads, thickness_cp, strut_thickness)
    u = frame.solve()

    # logsumexp(rho*x)/rho over-estimates max(x), and -logsumexp(-rho*x)/rho
    # under-estimates min(x), so constraining either one is conservative
    sigma_mpa = jnp.concatenate([beam.stress(u_beam) for beam, u_beam in zip(frame.beams, u)]) / 1e6
    max_sigma_mpa = logsumexp(rho_ks * sigma_mpa) / rho_ks
    thickness = jnp.concatenate([bspline_thickness @ thickness_cp, strut_thickness])
    min_thickness_mm = -logsumexp(-rho_ks_thick * thickness * 1e3) / rho_ks_thick

    weight = (frame.mass + m0) * 9.81

    return max_sigma_mpa, min_thickness_mm, weight
