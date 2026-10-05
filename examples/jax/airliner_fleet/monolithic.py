"""Solve the airliner fleet problem as one optimization problem with ModOpt SLSQP.

Each flight contributes its trajectory variables, local path constraints, and fuel use.
The only fleet-wide constraint is the total block-time budget. Pass the number of flights
as an argument, e.g. ``python monolithic.py 8``.
"""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import modopt as mo
from scipy.stats import qmc

from models import (
    nvar, cl, cu, xl, xu, x_scaler, setup_flight, flight_outputs,
    initial_guess,
)

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4

# Fleet definition and block-time budget, matching airliner_fleet.py.
ranges, masses = qmc.scale(
    qmc.LatinHypercube(d=2, seed=0).random(N),
    [2000e3, 22000.0], [5000e3, 28000.0],
).T
flights = [setup_flight(rf, m0) for rf, m0 in zip(ranges, masses)]
T_budget = 0.97 * np.sum(0.298 + 5.488e-6 * ranges)

# z = [flight 0 variables, ..., flight N-1 variables]. Each flight's variables are
# [altitude coefficients, speed coefficients, block-time allocation].
fleet = jax.tree.map(lambda *arrays: jnp.stack(arrays), *flights)
blocks = lambda z: z.reshape(N, nvar)
tile = lambda values: np.tile(values, N)

def flight_outputs_all(z):
    """Local constraints and fuel for every flight, with shape (N, nc + 1)."""
    return jax.vmap(flight_outputs)(blocks(z), fleet)


def objective(z):
    """Total fleet fuel in tonnes."""
    return jnp.sum(flight_outputs_all(z)[:, -1])


def constraints(z):
    """All flight-local constraints followed by the fleet block-time budget."""
    return jnp.concatenate(
        (
            flight_outputs_all(z)[:, :-1].reshape(-1),
            jnp.array([jnp.sum(blocks(z)[:, -1]) - T_budget]),
        )
    )


z0 = np.concatenate([initial_guess(flight) for flight in flights])
problem = mo.JaxProblem(
    x0=z0,
    jax_obj=objective,
    jax_con=constraints,
    xl=tile(xl),
    xu=tile(xu),
    cl=np.append(tile(cl), 0.0),
    cu=np.append(tile(cu), 0.0),
    x_scaler=tile(x_scaler),
    order=1,
)
optimizer = mo.SLSQP(
    problem,
    solver_options={"maxiter": 2000, "ftol": 1e-9},
    turn_off_outputs=True,
)
optimizer.solve()
optimizer.print_results()

# ModOpt's SLSQP multiplier uses L = f - lambda*c. The fleet budget cost index is
# reported with the convention L = f + y*c, hence the sign change.
z = optimizer.results["x"] / tile(x_scaler)
cost_index = -float(optimizer.results["multipliers"][N])
elapsed = optimizer.total_time
nit = optimizer.results["nit"]
success = optimizer.results["success"]

fuel_kg = 1e3 * np.asarray(flight_outputs_all(jnp.asarray(z)))[:, -1]
tau = blocks(z)[:, -1]
print(f"slsqp: {'converged' if success else 'did not converge'} in {nit} iterations, "
      f"{elapsed:.1f} s | fleet fuel {fuel_kg.sum():.1f} kg, "
      f"cost index {cost_index:.4f} kg/s, budget violation {tau.sum() - T_budget:.2e} ks")
for i in range(N):
    print(f"Flight {i} ({ranges[i] / 1e3:.0f} km, {masses[i]:.0f} kg): "
          f"fuel {fuel_kg[i]:.1f} kg, block time {tau[i]:.3f} ks")

# np.savez(
#     os.path.join(HERE, f"monolithic_solution_N{N}.npz"),
#     z=z, fuel=fuel_kg, tau=tau, cost_index=cost_index,
#     time=elapsed, nit=nit, success=success,
# )
