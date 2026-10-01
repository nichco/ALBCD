"""Scaling study for the distributed MDO ALBCD example.

Each case is rerun for a sequence of solution-error targets. The custom
ALBCD class stops immediately after a subproblem solve reaches its target and
writes one ``.npz`` file per N for ``plot_mdo_test_prob_scaling.py``.
"""
import contextlib
import functools
import io
import os
import runpy

import albcd
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "mdo_test_prob_albcd.py")
TARGETS = np.array([1e-2, 1e-3, 1e-4, 1e-5])
N_CASES = (2, 4, 8, 16, 32)
N0 = 10
NI = 20


class TargetReached(Exception):
    """Stop an ALBCD solve after its requested solution error is reached."""


class ErrorTerminatedALBCD(albcd.ALBCD):
    """ALBCD that stops after a subproblem solve reaches a known exact solution."""

    def __init__(self, *args, target, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = target
        self.solves = 0

        self.N = len(self.subproblems)
        self.m = self.subproblems[0].index.stop - self.subproblems[0].index.start
        self.n0 = self.mu.size // (self.N - 1)
        self.ni = self.m - self.n0
        drv = (np.arange(self.ni) * self.n0) // self.ni
        counts = np.bincount(drv, minlength=self.n0)
        scale = np.sqrt(counts * self.N)
        c = np.roots([400.0, 0.0, -198.0, -2.0]).real.max()
        exact_block = np.concatenate([c * scale, np.full(self.ni, 0.5)])
        self.exact = np.tile(exact_block, self.N)

        for sub in self.subproblems:
            sub.solve = self.checked(sub.solve)

    def checked(self, solve):
        def wrapper(x, y, mu, data, outputs):
            solve(x, y, mu, data, outputs)
            self.solves += 1
            error = np.max(np.abs(outputs["x"] - self.exact))
            if error <= self.target:
                raise TargetReached
        return wrapper

    def solve(self):
        try:
            super().solve()
        except TargetReached:
            self.success = True
        else:
            self.success = False


def solve_to(N, n0, ni, target):
    """Run one case until its target error is reached or ALBCD stops normally."""
    original = albcd.ALBCD
    albcd.ALBCD = functools.partial(ErrorTerminatedALBCD, target=target)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            module = runpy.run_path(
                SCRIPT,
                init_globals={"N": N, "n0": n0, "ni": ni},
            )
            solve_case = module["solve_case"]
            opt, _, _, _ = solve_case(N=N, n0=n0, ni=ni)
    finally:
        albcd.ALBCD = original
    return opt.solves, opt.success


def run_case(N, n0=N0, ni=NI):
    solves = []
    reached = []
    for target in TARGETS:
        solve_count, success = solve_to(N, n0, ni, target)
        solves.append(solve_count)
        reached.append(success)
        print(f"N={N:3d} target={target:.0e} | solves={solve_count:5d} | reached={success}")

    path = os.path.join(HERE, f"mdo_itr_v_error_N_{N}_n0_{n0}_ni_{ni}.npz")
    np.savez(path, N=N, n0=n0, ni=ni, target=TARGETS,
             solves=np.asarray(solves), reached=np.asarray(reached))
    return path


if __name__ == "__main__":
    for N in N_CASES:
        run_case(N)
