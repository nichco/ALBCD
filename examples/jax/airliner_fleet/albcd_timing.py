"""Wall-time history of the ALBCD solve of the airliner fleet problem.

Runs ALBCD with the settings of airliner_fleet.py, except for a tighter final optimality tolerance
(1e-6 instead of 1e-4 by default), and records the wall time after every block (flight) solve. Each design is then scored with fleet_kkt.kkt_history(), the KKT residual of the
monolithic problem, so the ALBCD and OpenSQP curves measure the same thing. ALBCD's own
optimality residual (opt_history) is saved too, with the time at the end of each sweep.

airliner_fleet.py runs its solve and its figures at import, so FlightSubproblem and the ALBCD
settings are repeated here. Pass the number of flights (default 8) and optionally the final
optimality tolerance (default 1e-6), e.g. ``python albcd_timing.py 8 1e-6``.

Writes albcd_timing_history_N{N}.npz, which compare_monolithic_and_albcd.py plots.
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
from albcd import ALBCD, Subproblem
import warnings
warnings.filterwarnings("ignore")

from models import nvar, cl, cu, xl, xu, x_scaler, flight_outputs, initial_guess, last_call_cache
from fleet_kkt import make_fleet, kkt_history_reusing, load_previous_kkt

HERE = os.path.dirname(os.path.abspath(__file__))


class TimedList(list):
    """A list that also records time.perf_counter() whenever an item is appended."""

    def __init__(self, items=()):
        super().__init__(items)
        self.times = [time.perf_counter()] * len(self)

    def append(self, item):
        super().append(item)
        self.times.append(time.perf_counter())


# ---------------------------------------------------------------------------------------------
# the subproblem of airliner_fleet.py, unchanged except that the fleet is passed in
# ---------------------------------------------------------------------------------------------

taus = lambda x: x[nvar - 1::nvar]

# compiled once and shared by every block (the flight's data is an argument)
outputs_fn = jax.jit(flight_outputs)
jac_fn = jax.jit(jax.jacfwd(flight_outputs))


def kkt_rows(J):
    """The rows of the local constraint Jacobian J in the order and sign of SLSQP's multipliers:
    equalities, then lower bounds, then upper bounds (negated), so that grad f = rows^T multipliers."""
    eq = cl == cu
    return np.concatenate((J[eq], J[(cl != -np.inf) & ~eq], -J[(cu != np.inf) & ~eq]))


class FlightSubproblem(Subproblem):
    """Flight i: owns its trajectory and its block-time allocation tau_i."""

    def __init__(self, i, flights, T_budget):
        fl = flights[i]
        self.T_budget = T_budget
        # this flight's [local constraints, fuel] and their Jacobian, each computed once per point
        self.outputs = last_call_cache(lambda v: np.asarray(outputs_fn(jnp.asarray(v), fl)))
        self.outputs_jac = last_call_cache(lambda v: np.asarray(jac_fn(jnp.asarray(v), fl)))
        super().__init__(slice(i * nvar, (i + 1) * nvar))

    def budget(self, v, x):
        """The budget constraint phi, with this flight's allocation taken from v and the others' from x."""
        return v[-1] + taus(x).sum() - x[self.index][-1] - self.T_budget

    def objective(self, v, x, y, mu):
        """Augmented Lagrangian: this flight's fuel plus the budget's multiplier and penalty terms."""
        phi = self.budget(v, x)
        return self.outputs(v)[-1] + y[0] * phi + 0.5 * mu[0] * phi**2

    def gradient(self, v, x, y, mu):
        grad = self.outputs_jac(v)[-1].copy()  # d(fuel)/dv
        grad[-1] += y[0] + mu[0] * self.budget(v, x)
        return grad

    def solve(self, x, y, mu, data, outputs) -> None:
        prob = mo.ProblemLite(x0=np.array(self.decompose(x)),
                              obj=lambda v: self.objective(v, x, y, mu),
                              grad=lambda v: self.gradient(v, x, y, mu),
                              con=lambda v: self.outputs(v)[:-1],
                              jac=lambda v: self.outputs_jac(v)[:-1],
                              xl=xl, xu=xu, cl=cl, cu=cu, x_scaler=x_scaler)
        optimizer = mo.SLSQP(prob, solver_options={"maxiter": 500, "ftol": 1e-9}, turn_off_outputs=True)
        optimizer.solve()
        v_new = optimizer.results["x"] / x_scaler

        # cache SLSQP's multipliers and the matching rows of the local constraints' Jacobian for residual()
        self._multipliers = np.asarray(optimizer.results["multipliers"])
        self._rows = kkt_rows(self.outputs_jac(v_new)[:-1])

        outputs["x"] = self.recompose(x, v_new)
        outputs["phi"] = np.array([taus(outputs["x"]).sum() - self.T_budget])

    def residual(self, x, y, mu, data) -> float:
        v = self.decompose(x)

        # Lagrangian gradient (with SLSQP's multipliers for the local constraints) projected onto
        # the box bounds: zero iff v is a KKT point of this block's subproblem
        grad_L = self.gradient(v, x, y, mu) - self._rows.T @ self._multipliers
        return float(np.max(np.abs(v - np.clip(v - grad_L, xl, xu))))


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    # final-phase optimality tolerance (ALBCD's own block residual); airliner_fleet.py uses 1e-4
    final_opt_tol = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-6
    ranges, masses, flights, T_budget = make_fleet(N)
    x0 = np.concatenate([initial_guess(fl) for fl in flights])

    # compile before the timed solve, as airliner_fleet.py does
    outputs_fn(jnp.asarray(x0[:nvar]), flights[0]), jac_fn(jnp.asarray(x0[:nvar]), flights[0])

    # the settings of airliner_fleet.py
    opt = ALBCD(subproblems=[FlightSubproblem(i, flights, T_budget) for i in range(N)],
                x0=x0,
                mu0=np.array([0.01]),
                max_mu=1e3,
                rho=2.0,
                tau=0.5,
                feas_tol=1e-4,
                opt_tol=[1e-2, final_opt_tol],
                max_y=1e6,
                max_outer_iter=100,
                max_inner_iter=20)
    opt.x_history = TimedList(opt.x_history)  # x0, then x after every block solve
    opt.opt_history = TimedList()             # ALBCD's optimality residual after every sweep

    t_start = time.perf_counter()
    opt.x_history.times[0] = t_start
    opt.solve()

    time_x = np.array(opt.x_history.times) - t_start
    time_sweep = np.array(opt.opt_history.times) - t_start
    xs = np.array(opt.x_history)

    blocks = opt.x.reshape(N, nvar)
    fuel_kg = np.array([1e3 * float(outputs_fn(jnp.asarray(v), fl)[-1]) for v, fl in zip(blocks, flights)])
    print(f"{len(opt.feas_history)} sweeps, wall time {opt.tf:.1f} s, success {opt.success}, "
          f"fleet fuel {fuel_kg.sum():.1f} kg, cost index {opt.y[0]:.4f}", flush=True)

    # save the run before the (slower) KKT post-processing, then again with the KKT measure.
    # time and x: one entry for x0 and one after every block solve; time_sweep, opt_native and
    # feas_native: one entry per sweep
    path = os.path.join(HERE, f"albcd_timing_history_N{N}.npz")
    previous = load_previous_kkt(path)  # loaded before the save below overwrites it
    run = dict(time=time_x, x=xs, time_sweep=time_sweep, opt_native=np.array(opt.opt_history),
               feas_native=np.array(opt.feas_history), fuel=fuel_kg, cost_index=opt.y[0],
               success=opt.success, opt_tol=opt.opt_tol, feas_tol=opt.feas_tol)
    np.savez(path, **run)

    optimality, feasibility, budget_multiplier = kkt_history_reusing(xs, N, previous)
    print(f"KKT {optimality[-1]:.2e}   feasibility {feasibility[-1]:.2e}   "
          f"budget multiplier {budget_multiplier[-1]:.4f}")
    np.savez(path, **run, optimality=optimality, feasibility=feasibility, budget_multiplier=budget_multiplier)
