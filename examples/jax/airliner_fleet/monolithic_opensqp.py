"""Solve the airliner fleet problem as a single optimization problem with modOpt's OpenSQP.

The same problem as monolithic.py (the same flights, objective, constraints, bounds and
scaling), solved with OpenSQP instead of SLSQP. The design and the wall time at every major
iteration are recorded, and each recorded design is then scored with fleet_kkt.kkt_history(),
the KKT residual of this monolithic problem, which albcd_timing.py also applies to the ALBCD
iterates. Pass the number of flights as an argument (default 8), e.g.
``python monolithic_opensqp.py 8``.

Writes monolithic_opensqp_history_N{N}.npz, which compare_monolithic_and_albcd.py plots.
"""

import os

# single-threaded BLAS and XLA, as in examples/comp/cessna_comparison.py, so the timings are
# reproducible and both methods run with the same resources. Must be set before numpy loads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
import modopt as mo
import warnings
warnings.filterwarnings("ignore")

from models import nvar, cl, cu, xl, xu, x_scaler, flight_outputs, initial_guess
from fleet_kkt import make_fleet, kkt_history_reusing, load_previous_kkt

HERE = os.path.dirname(os.path.abspath(__file__))


class RecordingOpenSQP(mo.OpenSQP):
    """OpenSQP that keeps the wall time, the design and OpenSQP's own optimality and feasibility
    at every major iteration (iteration 0 is the initial point), even with turn_off_outputs."""

    def __init__(self, *args, x_scaler, **kwargs):
        self.x_scaler = x_scaler
        super().__init__(*args, **kwargs)

    def solve(self):
        self.history = []
        self.t_start = time.perf_counter()
        return super().solve()

    def update_outputs(self, **kwargs):
        t = time.perf_counter() - self.t_start
        self.history.append((t, np.array(kwargs["x"]) / self.x_scaler, kwargs["opt"], kwargs["feas"]))
        print(f"iter {kwargs['major']:4d} | {t:8.2f} s | obj {kwargs['obj']:.8f} | "
              f"opt {kwargs['opt']:.2e} | feas {kwargs['feas']:.2e}", flush=True)
        super().update_outputs(**kwargs)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    ranges, masses, flights, T_budget = make_fleet(N)

    # z = [flight 0 variables, ..., flight N-1 variables], as in monolithic.py
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
        return jnp.concatenate((flight_outputs_all(z)[:, :-1].reshape(-1),
                                jnp.array([jnp.sum(blocks(z)[:, -1]) - T_budget])))

    z0 = np.concatenate([initial_guess(flight) for flight in flights])
    problem = mo.JaxProblem(
        x0=z0,
        nc=N * len(cl) + 1,  # more constraints than variables, so JaxProblem uses forward mode for the Jacobian
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

    optimizer = RecordingOpenSQP(problem, x_scaler=tile(x_scaler), maxiter=2000, turn_off_outputs=True)
    optimizer.solve()
    optimizer.print_results()

    times, xs, opt_native, feas_native = (np.array(a) for a in zip(*optimizer.history))
    z = xs[-1]
    fuel_kg = 1e3 * np.asarray(flight_outputs_all(jnp.asarray(z)))[:, -1]
    print(f"Fleet fuel: {fuel_kg.sum():.1f} kg, wall time {times[-1]:.1f} s, "
          f"success {optimizer.results['success']}", flush=True)

    # save the run before the (slower) KKT post-processing, then again with the KKT measure. The
    # measure is reused for the designs an earlier run shares exactly with this one
    path = os.path.join(HERE, f"monolithic_opensqp_history_N{N}.npz")
    previous = load_previous_kkt(path)  # loaded before the save below overwrites it
    run = dict(time=times, x=xs, opt_native=opt_native, feas_native=feas_native, fuel=fuel_kg,
               success=bool(optimizer.results["success"]), niter=int(optimizer.results["niter"]))
    np.savez(path, **run)

    optimality, feasibility, budget_multiplier = kkt_history_reusing(xs, N, previous)
    print(f"KKT {optimality[-1]:.2e}   feasibility {feasibility[-1]:.2e}   "
          f"budget multiplier {budget_multiplier[-1]:.4f}")
    np.savez(path, **run, optimality=optimality, feasibility=feasibility, budget_multiplier=budget_multiplier)
