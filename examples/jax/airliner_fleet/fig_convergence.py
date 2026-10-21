import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 8,
                     "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True,
                     "ytick.major.size": 3, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "pdf.fonttype": 42})

HERE = os.path.dirname(os.path.abspath(__file__))
sizes = [int(n) for n in sys.argv[1:]] or [2, 6, 10]

colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]  # Okabe-Ito blue, vermillion, green, purple
linestyles = ["-", (0, (5, 2)), (0, (1, 1.2)), (0, (5, 1.5, 1, 1.5))]
markers = ["o", "s", "^", "D"]

fig, (opt, feas, err) = plt.subplots(1, 3, figsize=(7, 2), sharex=True)
for k, n in enumerate(sizes):
    d = np.load(os.path.join(HERE, f"fleet_albcd_N{n}.npz"))
    sweeps = np.arange(1, len(d["feas_history"]) + 1)
    style = dict(color=colors[k % 4], ls=linestyles[k % 4], marker=markers[k % 4], lw=1.2, ms=3, mfc="w",
                 mew=0.8, markevery=4)
    opt.plot(sweeps, d["opt_history"], label=f"$N = {n}$", **style)
    feas.plot(sweeps, d["feas_history"], **style)
    if len(d["error"]):
        err.plot(np.arange(len(d["error"])) / n, d["error"], **{**style, "markevery": 4 * n})


for a, label in ((opt, "Optimality residual"), (feas, "Feasibility"),
                 (err, "Relative error in $x$")):
    a.set_yscale("log")
    a.xaxis.set_minor_locator(NullLocator())
    a.yaxis.set_minor_locator(NullLocator())
    a.set_ylabel(label)
    a.set_xlabel("Iteration")

err.legend(*opt.get_legend_handles_labels(), loc="upper right", handlelength=2.6)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_convergence.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.show()
