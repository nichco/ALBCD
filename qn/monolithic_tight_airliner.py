"""Tight monolithic reference for the airliner fleet (Task 5).

The problem of examples/jax/airliner_fleet/monolithic.py (copied from its lines 25-78), solved with SLSQP at
ftol = 1e-12. Usage: ``python qn/monolithic_tight_airliner.py N``. Saves
qn/results/monolithic_solution_tight_airliner_N{N}.npz and compares with the example's reference.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from airliner_blocks import import_example_models, EXAMPLE  # noqa: E402

models = import_example_models(EXAMPLE, "airliner_fleet_models")

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import modopt as mo  # noqa: E402
from scipy.stats import qmc  # noqa: E402

nvar, cl, cu, xl, xu, x_scaler = models.nvar, models.cl, models.cu, models.xl, models.xu, models.x_scaler
setup_flight, flight_outputs, initial_guess = models.setup_flight, models.flight_outputs, models.initial_guess

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4

# Flights and block-time budget, matching airliner_fleet.py.
ranges, masses = qmc.scale(
    qmc.LatinHypercube(d=2, seed=0).random(N),
    [2000e3, 22000.0], [5000e3, 28000.0],
).T
flights = [setup_flight(rf, m0) for rf, m0 in zip(ranges, masses)]
T_budget = 0.9 * np.sum(0.276 + 5.468e-6 * ranges)

fleet = jax.tree.map(lambda *arrays: jnp.stack(arrays), *flights)
blocks = lambda z: z.reshape(N, nvar)
tile = lambda values: np.tile(values, N)


def flight_outputs_all(z):
    """Local constraints and fuel for every flight, with shape (N, nc + 1)."""
    return jax.vmap(flight_outputs)(blocks(z), fleet)


def objective(z):
    """Total fuel over all flights in tonnes."""
    return jnp.sum(flight_outputs_all(z)[:, -1])


def constraints(z):
    """All flight-local constraints followed by the total block-time budget."""
    return jnp.concatenate(
        (
            flight_outputs_all(z)[:, :-1].reshape(-1),
            jnp.array([jnp.sum(blocks(z)[:, -1]) - T_budget]),
        )
    )


if __name__ == "__main__":
    z0 = np.concatenate([initial_guess(flight) for flight in flights])
    problem = mo.JaxProblem(
        x0=z0,
        nc=N * len(cl) + 1,
        jax_obj=objective,
        jax_con=constraints,
        xl=tile(xl),
        xu=tile(xu),
        cl=np.append(tile(cl), 0.0),
        cu=np.append(tile(cu), 0.0),
        x_scaler=tile(x_scaler),
        order=1,
    )
    problem.obj(z0), problem.grad(z0), problem.con(z0), problem.jac(z0)  # compile before the timed solve

    optimizer = mo.SLSQP(problem, solver_options={"maxiter": 2000, "ftol": 1e-12}, turn_off_outputs=True)
    t0 = time.perf_counter()
    optimizer.solve()
    elapsed = time.perf_counter() - t0
    optimizer.print_results()

    z = optimizer.results["x"] / tile(x_scaler)
    m = np.asarray(optimizer.results["multipliers"])
    cost_index = -float(m[N])
    fuel_kg = 1e3 * np.asarray(flight_outputs_all(jnp.asarray(z)))[:, -1]
    tau = blocks(z)[:, -1]

    ref_file = EXAMPLE / f"monolithic_solution_N{N}.npz"
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    extra = {}
    if ref_file.exists():
        ref = np.load(ref_file)
        dist = float(np.linalg.norm((z - ref["z"]) * tile(x_scaler)))
        print(f"fuel tight {fuel_kg.sum():.6f} kg, example reference {ref['fuel'].sum():.6f} kg, "
              f"difference {fuel_kg.sum() - ref['fuel'].sum():+.3e} kg; cost index {cost_index:.8f} vs "
              f"{float(ref['cost_index']):.8f}; ||z - z_ref|| (scaled) {dist:.3e}")
        extra = dict(fuel_example=ref["fuel"], cost_index_example=float(ref["cost_index"]), dist_example=dist)
    print(f"time {elapsed:.1f} s, nit {optimizer.results.get('nit')}, success {optimizer.results['success']}")
    np.savez(out / f"monolithic_solution_tight_airliner_N{N}.npz", z=z, fuel=fuel_kg, tau=tau, cost_index=cost_index,
             multipliers=m, time=elapsed, nit=optimizer.results.get("nit", -1), success=optimizer.results["success"],
             message=str(optimizer.results["message"]), **extra)
