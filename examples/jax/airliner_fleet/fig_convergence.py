"""Figure 3: ALBCD convergence for several fleet sizes.

Budget violation, block optimality residual and cost index against sweep, and the relative
error against the monolithic solution against block solves per flight. Reads
fleet_albcd_N{N}.npz (run airliner_fleet.py, and monolithic.py for the error) and writes
fig_convergence.pdf. Pass the fleet sizes as arguments (default 2 6 10).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "0.92", "legend.frameon": False})

HERE = os.path.dirname(os.path.abspath(__file__))
sizes = [int(n) for n in sys.argv[1:]] or [2, 6, 10]

fig, ax = plt.subplots(2, 2, figsize=(6.5, 4.6))
(feas, opt), (err, ci) = ax
for color, n in zip(plt.cm.viridis(np.linspace(0, 0.8, len(sizes))), sizes):
    d = np.load(os.path.join(HERE, f"fleet_albcd_N{n}.npz"))
    sweeps = np.arange(1, len(d["feas_history"]) + 1)
    feas.semilogy(sweeps, d["feas_history"], color=color, lw=1.3, label=f"$N$ = {n}")
    opt.semilogy(sweeps, d["opt_history"], color=color, lw=1.3)
    ci.plot(sweeps, d["y_history"], color=color, lw=1.3)
    if len(d["error"]):
        err.semilogy(np.arange(len(d["error"])) / n, d["error"], color=color, lw=1.3)
feas.axhline(float(d["feas_tol"]), color="0.5", ls="--", lw=0.8)
opt.axhline(float(d["opt_tol"][-1]), color="0.5", ls="--", lw=0.8)

feas.set_ylabel("Budget violation (ks)")
opt.set_ylabel("Optimality residual")
err.set_ylabel("Relative error")
ci.set_ylabel("Cost index $y$ (kg/s)")
for a in (feas, opt, ci):
    a.set_xlabel("Sweep")
err.set_xlabel("Block solves per flight")
feas.legend()
fig.align_ylabels()
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_convergence.pdf"), bbox_inches="tight")
plt.show()
