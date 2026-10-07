"""Run the Cessna ALBCD problem in cessna.py for a fixed wall-time budget, for several inner-loop limits.

Same comparison as inner_iter_sweep.py, but every run continues until the budget is spent instead
of stopping at ALBCD's convergence test, so the runs can be compared at equal wall time.
Writes cessna_inner_iter_budget.npz, for fig_inner_iter_budget.py.

ALBCD itself is unchanged:
- ALBCD stops when it is feasible in the last opt_tol phase. In any earlier phase, becoming
  feasible updates the multipliers and penalties and starts the next phase, which is the same as
  another outer iteration. Repeating the final tolerance many times therefore runs the same
  algorithm without the final stop.
- Each subproblem solve checks the clock first and raises BudgetSpent once the budget is used up.
  The histories are appended as the solve goes, so they are complete up to that point.
"""

import os
import time
import numpy as np

from cessna import HERE, make_albcd, design_error

inner_iters = [1, 2, 10, 100]
budget = 60.0  # wall time for each run, in seconds


class BudgetSpent(Exception):
    pass


def timed(method, times, deadline=None):
    """Wrap method so that it raises BudgetSpent if called after deadline, and appends the time at
    which each call returns to times."""
    def wrapper(*args, **kwargs):
        if deadline is not None and time.perf_counter() > deadline[0]:
            raise BudgetSpent
        out = method(*args, **kwargs)
        times.append(time.perf_counter())
        return out
    return wrapper


# the histories have different lengths for each run, so they are saved as object arrays
opt_history, feas_history, error, t_sweep, t_solve = [], [], [], [], []
for n_inner in inner_iters:
    print(f"\n=== max_inner_iter = {n_inner} ===")
    opt = make_albcd(max_inner_iter=n_inner)
    opt.opt_tol = np.concatenate([opt.opt_tol, np.full(10_000, opt.opt_tol[-1])])  # no final stop (see above)
    opt.max_outer_iter = 10**9                                   # the budget ends the run instead
    opt.verbose = False

    # each new subproblem jit-compiles its functions on their first call; call them once
    # here, untimed, so that compilation is not counted in the wall time
    for sub in opt.subproblems:
        sub.solve(opt.x, opt.y, opt.mu, opt.data, {})
        sub.residual(opt.x, opt.y, opt.mu, opt.data)

    # ALBCD calls every subproblem's residual() after each sweep, in order, so the last
    # subproblem's residual() marks the end of a sweep
    deadline = [np.inf]
    solve_times, residual_times = [], []
    for sub in opt.subproblems:
        sub.solve = timed(sub.solve, solve_times, deadline)
    opt.subproblems[-1].residual = timed(opt.subproblems[-1].residual, residual_times)

    t0 = time.perf_counter()
    deadline[0] = t0 + budget
    try:
        opt.solve()
    except BudgetSpent:
        pass

    # a sweep cut off by the budget may have one block solve without the sweep's residuals
    n_sweeps = len(residual_times)
    opt_history.append(np.array(opt.opt_history[:n_sweeps]))
    feas_history.append(np.array(opt.feas_history[:n_sweeps]))
    error.append(design_error(opt.x_history))
    t_sweep.append(np.array(residual_times) - t0)
    t_solve.append(np.concatenate([[0.0], np.array(solve_times) - t0]))  # x0 at t = 0
    print(f"max_inner_iter = {n_inner}: {n_sweeps} sweeps in {t_sweep[-1][-1]:.1f} s, "
          f"optimality {opt_history[-1][-1]:.2e}, feasibility {feas_history[-1][-1]:.2e}, "
          f"relative error {error[-1][-1]:.2e}")


def ragged(arrays):
    out = np.empty(len(arrays), dtype=object)
    out[:] = arrays
    return out


# opt_history, feas_history and t_sweep have one entry per sweep; error and t_solve have one per
# subproblem solve, and a leading entry for x0. Times are in seconds from the start of opt.solve()
np.savez(os.path.join(HERE, "cessna_inner_iter_budget.npz"), max_inner_iter=np.array(inner_iters),
         budget=budget, opt_history=ragged(opt_history), feas_history=ragged(feas_history),
         error=ragged(error), t_sweep=ragged(t_sweep), t_solve=ragged(t_solve))
