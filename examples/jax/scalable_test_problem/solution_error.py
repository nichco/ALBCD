import os
# one BLAS thread, as in scalable_test_problem_scaling.py: the thread count changes SLSQP's
# rounding, and with it ALBCD's path to the solution
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import io
import runpy
import contextlib
import functools
import numpy as np
import matplotlib.pyplot as plt
import albcd

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = np.array([1e-2, 1e-3, 1e-4, 1e-5]) # solution errors to solve to

def name(prefix, sizes):
    """File name for the sizes, e.g. monolithic_solution_N_4_n0_10_ni_20.npz."""
    return f"{prefix}_N_{sizes['N']}_n0_{sizes['n0']}_ni_{sizes['ni']}.npz"


class TargetReached(Exception):
    """Raised to stop ErrorTerminatedALBCD once the solution error reaches its target."""


class ErrorTerminatedALBCD(albcd.ALBCD):
    """ALBCD that stops as soon as the solution error is at most ``target``, checked after every
    subproblem solve. Its iterates are ALBCD's until then. ALBCD's own stopping test still
    applies, so ``success`` is True only if the target was reached before it.
    """

    def __init__(self, *args, target, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = target
        self.solves = 0

        # the problem's sizes: N blocks of n0 + ni variables, and (N - 1) * n0 + 1 coupling
        # constraints (so N must be at least 2)
        N = len(self.subproblems)
        m = self.subproblems[0].index.stop - self.subproblems[0].index.start
        n0 = (self.mu.size - 1) // (N - 1)
        self.sizes = dict(N=N, n0=n0, ni=m - n0)

        # the monolithic solution for these sizes, written by scalable_test_problem.py
        path = os.path.join(HERE, name("monolithic_solution", self.sizes))
        if not os.path.exists(path):
            raise FileNotFoundError(f"{os.path.basename(path)} not found; run scalable_test_problem.py "
                                    f"with N={N}, n0={n0}, ni={m - n0} first")
        solution = np.load(path)
        self.exact = np.hstack([np.tile(solution["x0"], (N, 1)), solution["xs"]]).ravel() # [z_1, x_1, ..., z_N, x_N]

        for sub in self.subproblems:
            sub.solve = self.checked(sub.solve)

    def checked(self, solve):
        """solve(), then stop once the solution error of the x it returns is at most the target."""
        def wrapper(inputs, outputs):
            solve(inputs, outputs)
            self.solves += 1
            if np.max(np.abs(outputs["x"][:self.exact.size] - self.exact)) <= self.target:
                raise TargetReached
        return wrapper

    def solve(self) -> None:
        try:
            super().solve()
            self.success = False # ALBCD stopped on its own tolerances first
        except TargetReached:
            self.success = True


def solve_to(target):
    """Subproblem solves, whether the target was reached, and the problem sizes, of the ALBCD
    example solved to a solution error of target."""
    original = albcd.ALBCD
    albcd.ALBCD = functools.partial(ErrorTerminatedALBCD, target=target)
    try:
        with contextlib.redirect_stdout(io.StringIO()): # the example's printed output
            opt = runpy.run_path(os.path.join(HERE, "scalable_test_problem_albcd.py"))["opt"]
    finally:
        albcd.ALBCD = original
    return opt.solves, opt.success, opt.sizes


results = [solve_to(target) for target in TARGETS]
solves = np.array([s for s, _, _ in results])
reached = np.array([r for _, r, _ in results])
sizes = results[0][2]
print(f"N={sizes['N']}, n0={sizes['n0']}, ni={sizes['ni']}")
for target, s, r in zip(TARGETS, solves, reached):
    print(f"Solution error {target:.0e}: {s} subproblem solves" + ("" if r else " (not reached)"))

np.savez(os.path.join(HERE, name("itr_v_error", sizes)), target=TARGETS[reached], solves=solves[reached])

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(TARGETS[reached], solves[reached], "o-", linewidth=2, markersize=7)
ax.set_xscale("log")
ax.invert_xaxis() # tighter errors to the right
ax.set_ylim(0, 1.1 * solves.max())
ax.set_xlabel("Solution error")
ax.set_ylabel("Iterations")

ax.grid(color='lavender')
plt.tight_layout()
plt.show()
