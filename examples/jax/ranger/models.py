import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # modopt and the models work in float64
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from jframe import Beam, BSpline, CSTube, Frame
from jvlm import aerodynamics as vlm_aero
from jvlm import solve_aero

from ranger_geometry_parameterization import twist_ffd

# aero setup
b = 9.0                       # span (m)
c = 1.4                       # chord (m), constant: the Ranger R7 wing is rectangular
v_inf = 50.0                  # freestream velocity (m/s), about the Ranger R7's cruise speed
rho_atm = 1.0                 # density (kg/m^3), about 2 km altitude
q = 0.5 * rho_atm * v_inf**2  # dynamic pressure (Pa)


def vlm_mesh(twist_cp):
    """The full-span VLM camber mesh (chordwise, spanwise, 3) of the wing, generated from its STEP geometry and
    twisted by the jeometry FFD in ranger_geometry_parameterization.py about the spar axis by the twist B-spline of ``twist_cp`` (rad)."""
    return twist_ffd(twist_cp)[0]


# Untwisted VLM mesh and spar nodes: the spar has a node at each of the mesh's 41 spanwise stations
mesh0, struct_mesh = (np.asarray(m) for m in twist_ffd(np.zeros(twist_ffd.twist_basis.shape[1])))
y = mesh0[0, :, 1]   # spanwise station coordinates

# structures setup
# The wing spar is a circular tube along the quarter chord, at the wing's mid-depth there (the FFD's
# beam mesh, which the twist leaves in place), clamped at the centerline where the wing carries
# through the cabin roof (the Ranger R7 wing is a cantilever, with no struts).
num_nodes = len(y)
num_elems = num_nodes - 1
fixed_nodes = [num_nodes // 2]   # the center node is fixed in all dofs
y_elem = 0.5 * (y[:-1] + y[1:])

tube_t_over_c = 0.18          # tube outer diameter as a fraction of the chord: the wing's thickness-to-chord ratio
d_tube = tube_t_over_c * c    # outer diameter of the tube (m), the same at every station
# largest admissible wall thickness: the outer radius, where the tube is solid. A thicker wall makes the
# inner radius negative and the section properties unphysical, so the thickness control points are bounded
# above by this (inactive at the optimum)
thickness_max = d_tube / 2
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
# the FFD's cubic B-spline across the span, evaluated at the VLM stations. The wall thickness is symmetric:
# a jframe cubic B-spline over the semispan, from the root (|y| = 0) to the tip, evaluated at |y| of the
# element midpoints of both semispans. A spline across the whole span would have to stay smooth through the
# root, and could not follow the optimal thickness, which peaks there with a corner.
bspline_twist = twist_ffd.twist_basis   # (num_nodes, num_cp_twist)
num_cp_twist = bspline_twist.shape[1]
num_cp_thickness = 8
bspline_thickness = BSpline(np.abs(y_elem), num_cp_thickness, domain=(0., b / 2)).basis   # (num_elems, num_cp_thickness)

twist_cp0 = np.ones(num_cp_twist) * np.deg2rad(3)
thickness_cp0 = np.linspace(0.01, min_gauge, num_cp_thickness)   # root to tip


def aero_model(twist_cp):
    """Return the drag coefficient, the nodal aero loads [drag, lift] and the total lift."""
    out = solve_aero(vlm_mesh(twist_cp), v_inf, rho_atm, symmetry=False, with_viscous=False)
    # (nx-1, ny-1, 3) panel forces -> one net force per spanwise strip -> nodal.
    # Keep drag (x) and lift (z) separately, stacked into one flat vector.
    strip_forces = vlm_aero.sectional_forces(out["sec_forces"])
    nodal_forces = vlm_aero.lump_to_nodes(strip_forces)   # (num_nodes, 3)
    aero_loads = jnp.concatenate([nodal_forces[:, 0], nodal_forces[:, 2]])
    lift = out["CL"] * q * out["S_ref"]

    return out["CD"], aero_loads, lift


def build_frame(aero_loads, thickness_cp):
    """The jframe model of the tube spar, loaded by the design-load aero loads."""
    thickness = bspline_thickness @ thickness_cp   # wall thickness of every element, (num_elems,)

    F = jnp.zeros((num_nodes, 6))
    F = F.at[:, 0].set(aero_loads[:num_nodes] * load_factor * safety_factor)  # drag (chordwise)
    F = F.at[:, 2].set(aero_loads[num_nodes:] * load_factor * safety_factor)  # lift (vertical)

    spar = Beam(struct_mesh, E, G, rho_mat, CSTube(d_tube / 2, thickness), F=F, fixed_nodes=fixed_nodes)
    return Frame([spar])


def structures_model(aero_loads, thickness_cp):
    """Return the KS max von Mises stress (MPa) and the total weight."""
    frame = build_frame(aero_loads, thickness_cp)
    u = frame.solve()

    # logsumexp(rho*x)/rho over-estimates max(x), so constraining it is conservative
    sigma_mpa = frame.beams[0].stress(u[0]) / 1e6
    max_sigma_mpa = logsumexp(rho_ks * sigma_mpa) / rho_ks

    weight = (frame.mass + m0) * 9.81

    return max_sigma_mpa, weight
