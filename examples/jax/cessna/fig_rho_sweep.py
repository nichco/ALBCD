import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 8,
                     "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True,
                     "ytick.major.size": 3, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "pdf.fonttype": 42})

HERE = os.path.dirname(os.path.abspath(__file__))
n = 2  # subproblems (aero, structures), so error has n entries per sweep

# Okabe-Ito blue, vermillion, green, purple, orange, sky blue
colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
linestyles = ["-", (0, (5, 2)), (0, (1, 1.2)), (0, (5, 1.5, 1, 1.5)), (0, (3, 1)), (0, (8, 2, 1, 2, 1, 2))]
markers = ["o", "s", "^", "D", "v", "p"]

d = np.load(os.path.join(HERE, "cessna_rho_sweep.npz"), allow_pickle=True)

fig, (opt, feas, err) = plt.subplots(1, 3, figsize=(7, 2), sharex=True)
for k, rho in enumerate(d["rho"]):
    sweeps = np.arange(1, len(d["feas_history"][k]) + 1)
    style = dict(color=colors[k % 6], ls=linestyles[k % 6], marker=markers[k % 6], lw=1.2, ms=3, mfc="w",
                 mew=0.8, markevery=4)
    opt.plot(sweeps, d["opt_history"][k], label=rf"$\rho = {rho:.2f}$", **style)
    feas.plot(sweeps, d["feas_history"][k], **style)
    err.plot(np.arange(len(d["error"][k])) / n, d["error"][k], **{**style, "markevery": 4 * n})

for a, label, tag in ((opt, "Optimality residual", "a"), (feas, "Feasibility", "b"),
                      (err, "Relative error in $x$", "c")):
    a.set_yscale("log")
    a.xaxis.set_minor_locator(NullLocator())
    a.yaxis.set_minor_locator(NullLocator())
    a.set_ylabel(label)
    a.set_xlabel("Iteration")
    a.set_title(f"({tag})", fontsize=10, loc="left")

fig.legend(*opt.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=6,
           handlelength=2.6, frameon=False)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_rho_sweep.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.show()
