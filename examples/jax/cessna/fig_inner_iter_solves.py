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

colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]  # Okabe-Ito blue, vermillion, green, purple
linestyles = ["-", (0, (5, 2)), (0, (1, 1.2)), (0, (5, 1.5, 1, 1.5))]
markers = ["o", "s", "^", "D"]

d = np.load(os.path.join(HERE, "cessna_inner_iter_budget.npz"), allow_pickle=True)

fig, (opt, feas, err) = plt.subplots(1, 3, figsize=(7, 2), sharex=True)
for k, n_inner in enumerate(d["max_inner_iter"]):
    style = dict(color=colors[k % 4], ls=linestyles[k % 4], marker=markers[k % 4], lw=1.2, ms=3, mfc="w",
                 mew=0.8, markevery=16)
    label = rf"$n_\mathrm{{inner}} = {n_inner}$" + (" (ADMM)" if n_inner == 1 else "")
    # optimality and feasibility at the end of each sweep (every n subproblem solves), error after each subproblem solve
    solves = n * np.arange(1, len(d["feas_history"][k]) + 1)
    opt.plot(solves, d["opt_history"][k], label=label, **style)
    feas.plot(solves, d["feas_history"][k], **style)
    err.plot(np.arange(len(d["error"][k])), d["error"][k], **{**style, "markevery": 16 * n})

for a, label, tag in ((opt, "Optimality residual", "a"), (feas, "Feasibility", "b"),
                      (err, "Relative error in $x$", "c")):
    a.set_yscale("log")
    a.xaxis.set_minor_locator(NullLocator())
    a.yaxis.set_minor_locator(NullLocator())
    a.set_ylabel(label)
    a.set_xlabel("Subproblem solves")
    a.set_title(f"({tag})", fontsize=10, loc="left")

fig.legend(*opt.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
           handlelength=2.6, frameon=False)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_inner_iter_solves.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.show()
