"""Plot the results of switching_study.py: ``python fig_switching_study.py ranger`` or ``fleet_N4``.

Left: ALBCD's merit, max(optimality / opt_tol, feasibility / feas_tol), after every sweep. Right: the time
of the hybrid solve against the sweep at which it switches, split into its two phases. Sweep 0 is the
monolithic solve. The labels mark where one setting of each criterion switches.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1] if len(sys.argv) > 1 else "ranger"
d = np.load(os.path.join(HERE, f"switching_study_{tag}.npz"))

sweeps, t_albcd, t_sqp, ok = d["sweeps"], d["t_albcd"], d["t_sqp"], d["success"]
shown = ["merit_below(10)", "merit_below(100)", "feasibility_below(10)", "stagnation(0.7)",
         "multipliers_settled(0.1)", "step_below(1e-3)"]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.4))
k = np.arange(1, len(d["merit"]) + 1)
ax.semilogy(k, d["optimality"], label="Optimality", color="tab:blue")
ax.semilogy(k, d["feasibility"], label="Feasibility", color="tab:orange")
ax.semilogy(k, d["merit"], label="Merit", color="k", lw=1, ls="--")
ax.set(xlabel="ALBCD sweep", title="Phase 1 alone")
ax.legend(frameon=False)
ax.grid(color="0.92")

bx.bar(sweeps, t_albcd, width=0.8, color="tab:blue", label="ALBCD")
bx.bar(sweeps, t_sqp, width=0.8, bottom=t_albcd, color="tab:red", label="OpenSQP")
for s, t, success in zip(sweeps, t_albcd + t_sqp, ok):
    if not success:
        bx.text(s, t, "x", ha="center", va="bottom", color="tab:red")
top = (t_albcd + t_sqp).max()
for i, label in enumerate(shown):
    s = int(d["switch"][list(d["labels"]).index(label)])
    bx.annotate(label, (s, (t_albcd + t_sqp)[list(sweeps).index(s)]), (s, top * (1.05 + 0.07 * i)), fontsize=7,
                ha="center", arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6))
bx.set(xlabel="Switch after sweep", ylabel="Wall time (s)", ylim=(0, top * 1.5), title="Hybrid solve")
bx.legend(frameon=False, loc="upper right", fontsize=8)
bx.grid(color="0.92", axis="y")
bx.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(HERE, f"fig_switching_study_{tag}.png"), dpi=200)
print("saved", f"fig_switching_study_{tag}.png")
