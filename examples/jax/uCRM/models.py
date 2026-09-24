"""Setup of the multipoint uCRM problem, shared by multipoint.py and monolithic.py.

N missions, with payload and range sampled with a Latin hypercube, are flown by
the same uCRM wing (vortex-lattice aerodynamics with viscous drag). The design
variables are the wing's twist control points and each mission's angle of attack.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # modopt and vlm_jax work in float64
import jax.numpy as jnp
from scipy.stats import qmc

from vlm_jax.analysis import build_problem, evaluate
from vlm_jax.functionals import GRAV_CONSTANT
from vlm_jax.meshing import generate_mesh

N = 2  # number of missions

mesh_dict = {"num_y": 15, "num_x": 8, "wing_type": "CRM", "symmetry": True, "num_twist_cp": 5}
mesh0, twist_cp0 = generate_mesh(mesh_dict)  # half-wing mesh and jig twist (deg)
num_twist_cp = mesh_dict["num_twist_cp"]
alpha0 = 5.0  # initial angle of attack (deg)

# payload and range of each mission
payload_bounds = (10000.0, 50000.0)  # kg
range_bounds = (5.0e6, 20.0e6)        # m
sample = qmc.LatinHypercube(d=2, seed=0).random(N)
payloads, ranges = qmc.scale(sample,
                             [payload_bounds[0], range_bounds[0]],
                             [payload_bounds[1], range_bounds[1]]).T
OEW = 90000.0  # kg, operating empty weight

# design variable bounds (deg)
twist_bounds = (-10.0, 15.0)
alpha_bounds = (-10.0, 10.0)

# one problem per mission: the same wing and flight condition, with that mission's range R and weight W0
problems = [
    build_problem(
        mesh0, num_twist_cp,
        symmetry=True, CL0=0.0, CD0=0.015, k_lam=0.05, c_max_t=0.303, with_viscous=True,
        v=248.136,                   # freestream speed, m/s
        beta=0.0,                    # sideslip, deg
        Mach_number=0.84,
        re=1.0e6,                    # Reynolds number per unit length, 1/m
        rho=0.38,                    # air density at cruise altitude, kg/m^3
        speed_of_sound=295.4,        # m/s
        CT=GRAV_CONSTANT * 17.0e-6,  # thrust-specific fuel consumption, 1/s
        R=float(R), load_factor=1.0, W0=float(OEW + payload), wing_structural_mass=0.0,
    )
    for payload, R in zip(payloads, ranges)
]

# each mission's fuel burn, jitted once so that logging it after every solve doesn't rerun the
# VLM op by op. problems[i] is a constant of the compiled function, so it's bound as a default
_fuelburn = [jax.jit(lambda twist_cp, alpha, p=p: evaluate(p, twist_cp, alpha)["fuelburn"]) for p in problems]


def fuelburn(i, twist_cp, alpha):
    """Breguet fuel burn (kg) of mission i."""
    return float(_fuelburn[i](jnp.asarray(twist_cp, dtype=float), jnp.asarray(alpha, dtype=float)))


def memoize_last(fn):
    """Cache fn's result at the most recent argument vector.

    modopt asks for the objective and the constraints (and the gradient and the
    Jacobian) at the same point through separate callbacks, so this lets one
    evaluation serve both.
    """
    cache = {}

    def wrapper(v):
        key = np.ascontiguousarray(v, dtype=float).tobytes()
        if cache.get("key") != key:
            cache["key"], cache["value"] = key, fn(v)
        return cache["value"]

    return wrapper
