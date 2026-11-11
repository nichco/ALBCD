"""Wall-time history of the ALBCD solve of the Cessna wing problem.

Runs ALBCD with the example's settings (make_albcd in cessna.py) and records the wall time
after every block solve. Each design is then scored with kkt_measure() from
monolithic_opensqp.py, the KKT residual of the monolithic problem, so the ALBCD and OpenSQP
curves measure the same thing. ALBCD's own optimality residual (the blocks' augmented-
Lagrangian stationarity, opt_history) is saved too, with the time at the end of each sweep.

Writes albcd_timing_history.npz, which compare_monolithic_and_albcd.py plots.
"""

import os

# single-threaded BLAS and XLA, as in examples/comp/cessna_comparison.py, so the timings are
# reproducible and both methods run with the same resources
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import time
import numpy as np

import cessna as C  # the example; its __main__ block does not run on import
from monolithic_opensqp import kkt_history, kkt_measure, x0 as x0_monolithic

HERE = os.path.dirname(os.path.abspath(__file__))


class TimedList(list):
    """A list that also records time.perf_counter() whenever an item is appended."""

    def __init__(self, items=()):
        super().__init__(items)
        self.times = [time.perf_counter()] * len(self)

    def append(self, item):
        super().append(item)
        self.times.append(time.perf_counter())


if __name__ == "__main__":
    # warm-up: one sweep with throwaway settings compiles each block's jitted functions. The timed
    # solve reuses the same subproblem objects, and so the same compiled code, so its times measure
    # the optimization and not JAX's compile time
    warm = C.make_albcd(max_outer_iter=1, max_inner_iter=1)
    warm.verbose = False
    warm.solve()
    kkt_measure(x0_monolithic)

    opt = C.make_albcd()
    opt.subproblems = warm.subproblems
    opt.x_history = TimedList(opt.x_history)  # x0, then x after every block solve
    opt.opt_history = TimedList()             # ALBCD's optimality residual after every sweep

    t_start = time.perf_counter()
    opt.x_history.times[0] = t_start
    opt.solve()

    time_x = np.array(opt.x_history.times) - t_start
    time_sweep = np.array(opt.opt_history.times) - t_start

    # the monolithic design variables [twist_cp, thickness_cp] of every iterate; the
    # monolithic measure evaluates the aero loads and weight from them, not from the copies
    xs = np.array([np.concatenate([x[C.TWIST_SLICE], x[C.THICKNESS_SLICE]]) for x in opt.x_history])
    optimality, feasibility = kkt_history(xs)

    print(f'{len(opt.feas_history)} sweeps, wall time {opt.tf:.2f} s, success {opt.success}')
    print(f'CD: {opt.data["CD"]:.6e}   KKT {optimality[-1]:.2e}   feasibility {feasibility[-1]:.2e}')

    # time, optimality, feasibility and x: one entry for x0 and one after every block solve.
    # time_sweep, opt_native and feas_native: one entry per sweep
    np.savez(os.path.join(HERE, 'albcd_timing_history.npz'),
             time=time_x, optimality=optimality, feasibility=feasibility, x=xs,
             time_sweep=time_sweep, opt_native=np.array(opt.opt_history),
             feas_native=np.array(opt.feas_history), CD=opt.data["CD"], success=opt.success)
