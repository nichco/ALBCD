"""Optimality, feasibility and distance from the solution against wall time, for pure ALBCD, monolithic OpenSQP
and hybrid ALBCD.

    python fig_convergence.py ranger fleet_N4 fleet_N8        (one column per problem; default: every convergence_*.npz)
    python fig_convergence.py ranger fleet_N4 fleet_N8 linear (a linear time axis instead of a logarithmic one)

Reads convergence_<problem>.npz (convergence_runs.py) and distance_<problem>.npz (distances.py) and writes
fig_convergence.png. The optimality and feasibility are the same measure for every method, OpenSQP's of the
full problem; the distance is in its scaled variables. The times exclude setup and compilation. Edit RUNS to
choose the curves (convergence_runs.py also saves monolithic_hessian, the monolithic solve started from the
exact Hessian of the objective).
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LINEAR = "linear" in sys.argv
tags = [a for a in sys.argv[1:] if a != "linear"] or sorted(
    os.path.basename(p)[len("convergence_"):-4] for p in glob.glob(os.path.join(HERE, "convergence_*.npz")))
titles = {"ranger": "Aerostructural problem"}

# run: (label, color, line style)
RUNS = {"albcd": ("ALBCD", "tab:blue", "-"),
        "monolithic": ("Monolithic (OpenSQP)", "tab:red", "-"),
        "hybrid": ("Hybrid ALBCD", "k", "-"),
        "hybrid_hessian": ("Hybrid ALBCD, Hessian warm start", "tab:green", "-")}
ROWS = [("optimality", "Optimality"), ("feasibility", "Feasibility"), ("distance", r"$\Vert z - z^*\Vert$")]
floor = 1e-13  # values below this (e.g. exact feasibility) are drawn here on the log axis

fig, axs = plt.subplots(len(ROWS), len(tags), figsize=(4.8 * len(tags), 8.6), squeeze=False, sharex="col")
for col, tag in enumerate(tags):
    d = dict(np.load(os.path.join(HERE, f"convergence_{tag}.npz")))
    distance_file = os.path.join(HERE, f"distance_{tag}.npz")
    if os.path.exists(distance_file):
        d.update(np.load(distance_file))
    runs = [run for run in RUNS if f"{run}_time" in d]
    t_max = max(d[f"{run}_time"][-1] for run in runs)
    for row, (key, ylabel) in enumerate(ROWS):
        ax = axs[row, col]
        for run in runs:
            if f"{run}_{key}" not in d:
                continue
            label, color, ls = RUNS[run]
            t, v = d[f"{run}_time"], np.maximum(d[f"{run}_{key}"], floor)
            ax.semilogy(t, v, color=color, ls=ls, lw=1.5, label=label, zorder=3 if run.startswith("hybrid") else 2)
            ax.plot(t[-1], v[-1], "o", color=color, ms=4, zorder=4)
            if run.startswith("hybrid"):  # where the hybrid hands over to OpenSQP
                k = int(d[f"{run}_switch_sweep"])
                ax.plot(t[k], v[k], "D", color=color, ms=6, mfc="w", zorder=5,
                        label=f"Switch: {str(d[f'{run}_switch_label'])}")
        if key in ("optimality", "feasibility"):
            ax.axhline(float(d["sqp_opt_tol" if key == "optimality" else "sqp_feas_tol"]), color="0.6", lw=0.8, ls=":")
        if not LINEAR:
            ax.set_xscale("symlog", linthresh=max(1.0, 0.02 * t_max))
        ax.set_xlim(left=0)
        ax.grid(color="0.92")
        ax.set_axisbelow(True)
        if col == 0:
            ax.set_ylabel(ylabel)
    axs[0, col].set_title(titles.get(tag, tag.replace("fleet_N", "Airliner fleet, N = ")))
    axs[-1, col].set_xlabel("Wall time (s)")
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.06, 1, 1))
path = os.path.join(HERE, "fig_convergence_linear.png" if LINEAR else "fig_convergence.png")
fig.savefig(path, dpi=200)
print("saved", path)
