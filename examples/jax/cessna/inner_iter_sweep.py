"""Solve the Cessna ALBCD problem in cessna.py for a sweep of inner-loop limits max_inner_iter.

max_inner_iter = 1 is one sweep per multiplier update, as in ADMM; 100 solves each outer
iteration's block problem to the optimality tolerance. Every block solve and every sweep is
timestamped, so the convergence can be compared against wall time. Writes cessna_inner_iter_sweep.npz,
for fig_inner_iter_sweep.py.
"""

import os
import time
import numpy as np

from cessna import HERE, make_albcd, design_error

inner_iters = [1, 2, 100]
max_outer_iter = 200  # enough for one sweep per outer iteration; only reached if a run stalls


def timestamped(method, times):
    """Wrap method so that it appends the time at which each call returns to times."""
    def wrapper(*args, **kwargs):
        out = method(*args, **kwargs)
        times.append(time.perf_counter())
        return out
    return wrapper


# the histories have different lengths for each run, so they are saved as object arrays
opt_history, feas_history, error, t_sweep, t_solve, success = [], [], [], [], [], []
for n_inner in inner_iters:
    print(f"\n=== max_inner_iter = {n_inner} ===")
    opt = make_albcd(max_inner_iter=n_inner, max_outer_iter=max_outer_iter)

    # each new subproblem jit-compiles its functions on their first call; call them once
    # here, untimed, so that compilation is not counted in the wall time
    for sub in opt.subproblems:
        sub.solve(opt.x, opt.y, opt.mu, opt.data, {})
        sub.residual(opt.x, opt.y, opt.mu, opt.data)

    # ALBCD calls every subproblem's residual() after each sweep, in order, so the last
    # subproblem's residual() marks the end of a sweep
    solve_times, residual_times = [], []
    for sub in opt.subproblems:
        sub.solve = timestamped(sub.solve, solve_times)
    opt.subproblems[-1].residual = timestamped(opt.subproblems[-1].residual, residual_times)

    t0 = time.perf_counter()
    opt.solve()

    opt_history.append(np.array(opt.opt_history))
    feas_history.append(np.array(opt.feas_history))
    error.append(design_error(opt.x_history))
    t_sweep.append(np.array(residual_times) - t0)
    t_solve.append(np.concatenate([[0.0], np.array(solve_times) - t0]))  # x0 at t = 0
    success.append(opt.success)
    print(f"max_inner_iter = {n_inner}: {len(opt.feas_history)} sweeps, {t_sweep[-1][-1]:.1f} s, "
          f"success {opt.success}, relative error {error[-1][-1]:.2e}")


def ragged(arrays):
    out = np.empty(len(arrays), dtype=object)
    out[:] = arrays
    return out


# opt_history, feas_history and t_sweep have one entry per sweep; error and t_solve have one per
# subproblem solve, and a leading entry for x0. Times are in seconds from the start of opt.solve()
np.savez(os.path.join(HERE, "cessna_inner_iter_sweep.npz"), max_inner_iter=np.array(inner_iters),
         opt_history=ragged(opt_history), feas_history=ragged(feas_history), error=ragged(error),
         t_sweep=ragged(t_sweep), t_solve=ragged(t_solve), success=np.array(success))
