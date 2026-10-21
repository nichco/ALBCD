"""Subproblem solves ALBCD needs to reach each solution error, on the airliner allocation problem.

Runs airliner_fleet.py with ALBCD replaced by ErrorTerminatedALBCD, which stops as soon as the
relative error in x (as defined in airliner_fleet.py: the x_scaler-scaled 2-norm, relative to
the monolithic solution) is at most the tightest of TARGETS. ALBCD is deterministic, so a single
run gives every looser target too: the solves to reach it are those at which the run first
crossed it, the same count a run stopped at that target would report. The example is stopped
right after its solve, so its own output files (fleet_albcd_N*.npz, the gifs) are not written.

For each N, the results are saved to solves_v_error_N{N}.npz, and the figure of every saved N
to fig_solution_error.pdf. Needs monolithic_solution_N{N}.npz (from monolithic.py). Pass the
numbers of flights as arguments, e.g. `python solution_error.py 2 4 6`, or `--plot` to only
replot the saved results.
"""

import os
# as in airliner_fleet.py; must be set before numpy/scipy load OpenBLAS
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import io
import sys
import glob
import runpy
import contextlib
import functools
import numpy as np
import albcd

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = np.array([1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]) # relative errors in x to solve to
SIZES = [2, 3, 4, 5, 6, 7, 8, 9, 10] # numbers of flights, when none are passed


class TargetReached(Exception):
    """Raised to stop ErrorTerminatedALBCD once the solution error reaches its tightest target."""


class ExampleStopped(Exception):
    """Raised after the solve, to stop airliner_fleet.py before it writes its own output files."""


class ErrorTerminatedALBCD(albcd.ALBCD):
    """ALBCD that stops as soon as the solution error is at most ``min(targets)``, checked after
    every subproblem solve, and records the solves at which it first reached each target. Its
    iterates are ALBCD's until then. ALBCD's own stopping test still applies, so a target it
    stops before reaching is recorded as not reached.

    The airliner example's subproblems are set up for x_scaler, so the error is measured as in
    airliner_fleet.py: ||(x - x*) * scale|| / ||x* * scale||, with x* the monolithic solution.
    """

    def __init__(self, *args, targets, exact, scale, **kwargs):
        super().__init__(*args, **kwargs)
        self.targets = np.sort(np.asarray(targets, dtype=float))[::-1] # loosest first
        self.exact = exact * scale
        self.scale = scale
        self.solves = 0
        self.solves_to_target = np.full(self.targets.size, -1) # -1 until reached
        self.errors = [self.error(self.x)] # x0, then after every solve

        for sub in self.subproblems:
            sub.solve = self.checked(sub.solve)

    def error(self, x):
        return float(np.linalg.norm(x * self.scale - self.exact) / np.linalg.norm(self.exact))

    def checked(self, solve):
        """solve(), then record which targets the x it returns reaches, and stop at the tightest."""
        def wrapper(x, y, mu, data, outputs):
            solve(x, y, mu, data, outputs)
            self.solves += 1
            e = self.error(outputs["x"])
            self.errors.append(e)
            self.solves_to_target[(self.solves_to_target < 0) & (e <= self.targets)] = self.solves
            if e <= self.targets[-1]:
                raise TargetReached
        return wrapper

    def solve(self):
        try:
            super().solve()
            self.success = False # ALBCD stopped on its own tolerances first
        except TargetReached:
            self.success = True
        raise ExampleStopped


def sweep(N):
    """Solve the N-flight example to the tightest target; return the ErrorTerminatedALBCD."""
    sys.path.insert(0, HERE) # for the example's `from models import ...`
    from models import nvar, x_scaler
    reference = os.path.join(HERE, f"monolithic_solution_N{N}.npz")
    if not os.path.exists(reference):
        raise FileNotFoundError(f"{os.path.basename(reference)} not found; run monolithic.py {N} first")

    created = []
    def make(*args, **kwargs):
        opt = ErrorTerminatedALBCD(*args, targets=TARGETS, exact=np.load(reference)["z"],
                                   scale=np.tile(x_scaler, N), **kwargs)
        created.append(opt)
        return opt

    original, argv = albcd.ALBCD, sys.argv
    albcd.ALBCD, sys.argv = make, [os.path.join(HERE, "airliner_fleet.py"), str(N)]
    try:
        with contextlib.redirect_stdout(io.StringIO()): # the example's printed output
            runpy.run_path(sys.argv[0], run_name="__main__")
    except ExampleStopped:
        pass
    finally:
        albcd.ALBCD, sys.argv = original, argv
    return created[0]


def plot():
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 8,
                         "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True,
                         "ytick.major.size": 3, "pdf.fonttype": 42})

    files = glob.glob(os.path.join(HERE, "solves_v_error_N*.npz"))
    cases = sorted((int(os.path.basename(f)[len("solves_v_error_N"):-4]), np.load(f)) for f in files)
    if not cases:
        raise FileNotFoundError("no solves_v_error_N*.npz files; run the sweep first")

    # one sequential hue, light to dark with N
    colors = plt.cm.Blues(np.linspace(0.35, 1.0, len(cases)))
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for (N, d), c in zip(cases, colors):
        ok = d["reached"]
        ax.plot(d["target"][ok], d["solves"][ok], "o-", color=c, lw=1.2, ms=4, mfc="w", mew=1.0, label=f"$N = {N}$")
    ax.set_xscale("log")
    ax.invert_xaxis() # tighter errors to the right
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Relative error in $x$")
    ax.set_ylabel("Subproblem solves")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    path = os.path.join(HERE, "fig_solution_error.pdf")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path[:-4] + ".png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {os.path.basename(path)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args != ["--plot"]:
        for N in [int(n) for n in args] or SIZES:
            opt = sweep(N)
            reached = opt.solves_to_target >= 0
            for target, s in zip(opt.targets, opt.solves_to_target):
                print(f"N = {N}: error {target:.0e} after {s} subproblem solves" if s >= 0
                      else f"N = {N}: error {target:.0e} not reached ({opt.solves} solves)", flush=True)
            np.savez(os.path.join(HERE, f"solves_v_error_N{N}.npz"),
                     target=opt.targets, solves=opt.solves_to_target, reached=reached,
                     error=np.array(opt.errors), N=N)
    plot()
