"""Setup of the multipoint uCRM problem, shared by multipoint.py and monolithic.py.

N missions, with payload and range sampled with a Latin hypercube, are flown by
the same uCRM wing (vortex-lattice aerodynamics with viscous drag). The design
variables are the wing's twist control points and each mission's angle of attack.
"""

import numpy as np
import torch
torch.set_default_dtype(torch.float64)  # modopt and vlm_torch work in float64
from scipy.stats import qmc

from vlm_torch.analysis import build_problem, evaluate
from vlm_torch.functionals import GRAV_CONSTANT
from vlm_torch.meshing import generate_mesh

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


def fuelburn(i, twist_cp, alpha):
    """Breguet fuel burn (kg) of mission i."""
    out = evaluate(problems[i], torch.as_tensor(np.asarray(twist_cp, dtype=float)), torch.as_tensor(float(alpha)))
    return float(out["fuelburn"])


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
