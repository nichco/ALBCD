"""Vashon Ranger R7 wing setup and the aerodynamic (jvlm-oas VLM) and structural (jframe)
models, shared by ranger.py and monolithic.py so both solve exactly the same problem.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # modopt and the models work in float64
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from jframe import Beam, BSpline, CSRectTube, Frame
from jvlm import aerodynamics as vlm_aero
from jvlm import geometry as vlm_geom
from jvlm import generate_mesh, solve_aero
from jvlm.bsplines import get_bspline_mtx

# aero setup
b = 9.0                       # span (m)
c = 1.4                       # chord (m), constant: the Ranger R7 wing is rectangular
v_inf = 50.0                  # freestream velocity (m/s), about the Ranger R7's cruise speed
rho_atm = 1.0                 # density (kg/m^3), about 2 km altitude
q = 0.5 * rho_atm * v_inf**2  # dynamic pressure (Pa)

# Full-span rectangular VLM mesh whose 41 spanwise stations double as the wing box nodes
mesh0 = generate_mesh({"num_x": 8, "num_y": 41, "wing_type": "rect", "span": b,
                       "root_chord": c, "symmetry": False})
mesh0_jnp = jnp.asarray(mesh0)
y = mesh0[0, :, 1]   # spanwise station coordinates

# structures setup
# The wing box is a square tube along the quarter chord, clamped at the centerline where
# the wing carries through the cabin roof (the Ranger R7 wing is a cantilever, with no struts).
# One wall thickness applies to all four walls, the two caps and the two webs.
num_nodes = len(y)
num_elems = num_nodes - 1
x_box = 0.25 * c   # chordwise position of the wing box (m)
struct_mesh = np.stack([np.full(num_nodes, x_box), y, np.zeros(num_nodes)], axis=1)
fixed_nodes = [num_nodes // 2]   # the center node is fixed in all dofs
y_elem = 0.5 * (y[:-1] + y[1:])

box_t_over_c = 0.16        # wing box depth as a fraction of the chord
h_box = box_t_over_c * c   # outer side of the square wing box (m), the same at every station
# largest admissible wall thickness: half the box side, where the box is solid. A thicker wall makes the
# inner side negative and the section properties unphysical, so the thickness control points are bounded
# above by this (inactive at the optimum)
thickness_max = h_box / 2
E = 73e9          # aluminum 2024-T3
G = 28e9
rho_mat = 2780
sigma_yield_mpa = 345.0
m0 = 550.0           # mass of the aircraft without the wing box (kg), for a ~600 kg (1320 lb) LSA gross weight
load_factor = 4      # ASTM F2245 limit load factor for light sport aircraft
safety_factor = 1.5
rho_ks = 2e-1         # KS aggregation parameter for the max stress (1/MPa)
# minimum wall thickness (m), a lower bound on the thickness control points. The thickness B-spline is a
# convex combination of its control points, so this bounds the wall thickness of every element as well
min_gauge = 0.001

# design variables: B-spline control points for the twist (rad) and the wall thickness (m). The twist is
# interpolated at the VLM stations with jvlm's B-spline. The wall thickness is symmetric: a jframe cubic
# B-spline over the semispan, from the root (|y| = 0) to the tip, evaluated at |y| of the element midpoints
# of both semispans. A spline across the whole span would have to stay smooth through the root, and could
# not follow the optimal thickness, which peaks there with a corner.
num_cp_twist = 7
num_cp_thickness = 8
bspline_twist = jnp.asarray(get_bspline_mtx(num_cp_twist, np.linspace(0, 1, num_nodes)))
bspline_thickness = BSpline(np.abs(y_elem), num_cp_thickness, domain=(0., b / 2)).basis   # (num_elems, num_cp_thickness)

twist_cp0 = np.ones(num_cp_twist) * np.deg2rad(3)
thickness_cp0 = np.linspace(0.01, min_gauge, num_cp_thickness)   # root to tip


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


def build_frame(aero_loads, thickness_cp):
    """The jframe model of the wing box, loaded by the design-load aero loads."""
    thickness = bspline_thickness @ thickness_cp   # wall thickness of every element, (num_elems,)

    F = jnp.zeros((num_nodes, 6))
    F = F.at[:, 0].set(aero_loads[:num_nodes] * load_factor * safety_factor)  # drag (chordwise)
    F = F.at[:, 2].set(aero_loads[num_nodes:] * load_factor * safety_factor)  # lift (vertical)

    box = Beam(struct_mesh, E, G, rho_mat, CSRectTube(h_box, h_box, thickness), F=F, fixed_nodes=fixed_nodes)
    return Frame([box])


def structures_model(aero_loads, thickness_cp):
    """Return the KS max von Mises stress (MPa) and the total weight."""
    frame = build_frame(aero_loads, thickness_cp)
    u = frame.solve()

    # logsumexp(rho*x)/rho over-estimates max(x), so constraining it is conservative
    sigma_mpa = frame.beams[0].stress(u[0]) / 1e6
    max_sigma_mpa = logsumexp(rho_ks * sigma_mpa) / rho_ks

    weight = (frame.mass + m0) * 9.81

    return max_sigma_mpa, weight
