"""Minimum-fuel trajectory of each flight in the fleet, with no block-time budget.

Each flight is solved on its own with SLSQP, so the result is the fleet's unconstrained optimum,
for comparison with the budget-constrained solution in fig_trajectories.py. Saves the flights'
variables to minimum_fuel_N{N}.npz. Pass the number of flights as an argument, e.g.
`python minimum_fuel.py 8`.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import numpy as np
import jax
import jax.numpy as jnp
import modopt as mo
from scipy.stats import qmc
import warnings
warnings.filterwarnings("ignore")

from models import cl, cu, xl, xu, x_scaler, setup_flight, flight_outputs, initial_guess

HERE = os.path.dirname(os.path.abspath(__file__))

# the same fleet as airliner_fleet.py
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
ranges, masses = qmc.scale(qmc.LatinHypercube(d=2, seed=0).random(N), [2000e3, 22000.0], [5000e3, 28000.0]).T

outputs_fn = jax.jit(flight_outputs)
jac_fn = jax.jit(jax.jacfwd(flight_outputs))

x, fuel = [], []
for rf, m0 in zip(ranges, masses):
    fl = setup_flight(rf, m0)
    outputs = lambda v: np.asarray(outputs_fn(jnp.asarray(v), fl))
    jac = lambda v: np.asarray(jac_fn(jnp.asarray(v), fl))
    prob = mo.ProblemLite(x0=initial_guess(fl), obj=lambda v: outputs(v)[-1], grad=lambda v: jac(v)[-1],
                          con=lambda v: outputs(v)[:-1], jac=lambda v: jac(v)[:-1],
                          xl=xl, xu=xu, cl=cl, cu=cu, x_scaler=x_scaler)
    optimizer = mo.SLSQP(prob, solver_options={"maxiter": 500, "ftol": 1e-9}, turn_off_outputs=True)
    optimizer.solve()
    x.append(optimizer.results["x"] / x_scaler)
    fuel.append(1e3 * optimizer.results["fun"])
    print(f"{rf / 1e3:.0f} km, {m0:.0f} kg: success {optimizer.results['success']}, "
          f"fuel {fuel[-1]:.1f} kg, flight time {x[-1][-1]:.3f} ks")

np.savez(os.path.join(HERE, f"minimum_fuel_N{N}.npz"), x=np.concatenate(x), fuel=np.array(fuel),
         ranges=ranges, masses=masses)
