"""Bar plots of the ALBCD scaling study in scalable_test_problem_scaling.npz.

Each column is one sweep around the baseline (N, n0, ni) = (4, 10, 20), with the other two
sizes held at the baseline, and each row one result. Cases that didn't meet ALBCD's
tolerances are marked, since their costs aren't comparable with the others. Run
scalable_test_problem_scaling.py first.
"""

import os
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
results = np.load(os.path.join(HERE, "scalable_test_problem_scaling.npz"))
settings = ast.literal_eval(str(results["settings"]))

BLUE, RED = "#2a78d6", "#d03b3b" # met the tolerances, didn't
INK, MUTED, GRID, AXIS = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

BASELINE = dict(N=4, n0=10, ni=20)
SWEEPS = [("N", "Subproblems N"), ("n0", "Global variables n0"), ("ni", "Local variables ni")]

# (result, label, tolerance drawn as a line)
ROWS = [("optimality", "Optimality", settings["opt_tol"][-1]),
        ("feasibility", "Feasibility", settings["feas_tol"]),
        ("solves", "Block solves", None),
        ("analyses", "Model analyses", None),
        ("derivatives", "Model derivatives", None),
        ("wall_time", "Wall time (s)", None)]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(len(ROWS), len(SWEEPS), figsize=(11, 12.5), sharex="col")

for j, (var, name) in enumerate(SWEEPS):
    # the cases of this sweep: the other two sizes at the baseline, ordered by this one
    fixed = [results[k] == v for k, v in BASELINE.items() if k != var]
    cases = np.flatnonzero(np.all(fixed, axis=0))
    cases = cases[np.argsort(results[var][cases])]
    x = np.arange(len(cases))
    colors = [BLUE if results["success"][c] else RED for c in cases]

    for i, (key, label, tol) in enumerate(ROWS):
        ax = axes[i, j]
        values = results[key][cases]
        ax.bar(x, values, width=0.35, color=colors, zorder=2)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(which="both", length=0)

        # the tolerance rows span decades, so they're on a log scale shared along the row
        if tol is not None:
            ax.set_yscale("log")
            row = results[key]
            ax.set_ylim(min(row.min(), tol) / 5, max(row.max(), tol) * 5)
            ax.axhline(tol, color=INK, linestyle="--", linewidth=1, zorder=3)
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 10 else f"{v:g}"))

        for xi, c in zip(x, cases):
            if not results["success"][c]:
                ax.annotate("✗", (xi, results[key][c]), xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", color=INK)

        if j == 0:
            ax.set_ylabel(label)
        if i == 0:
            others = ", ".join(f"{k}={v}" for k, v in BASELINE.items() if k != var)
            ax.set_title(f"{name} ({others})", fontsize=10)

    ax.set_xticks(x, [str(results[var][c]) for c in cases])
    ax.set_xlabel(name)
    for tick, c in zip(ax.get_xticklabels(), cases):
        if results[var][c] == BASELINE[var]:
            tick.set_fontweight("bold") # the baseline case, the same in every column

opt_tol, feas_tol = settings["opt_tol"][-1], settings["feas_tol"]
fig.suptitle("ALBCD scaling on the scalable test problem", fontsize=13, x=0.02, ha="left")
fig.text(0.02, 0.955, f"Same settings for every case: final opt_tol={opt_tol:g}, feas_tol={feas_tol:g}, "
         f"max_outer_iter={settings['max_outer_iter']}, max_inner_iter={settings['max_inner_iter']}. "
         "Baseline case in bold.", color=MUTED)
fig.legend(handles=[Patch(color=BLUE, label="Met the tolerances"),
                    Patch(color=RED, label="✗ Didn't meet the tolerances (costs not comparable)"),
                    Line2D([], [], color=INK, linestyle="--", linewidth=1, label="Tolerance")],
           loc="upper left", bbox_to_anchor=(0.02, 0.945), ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.925))
fig.savefig(os.path.join(HERE, "scalable_test_problem_scaling.png"), dpi=150)
plt.show()
