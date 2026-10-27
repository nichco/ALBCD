"""Plot the ALBCD / ALC / ADMM comparison on the Cessna problem from cessna_comparison.npz
(run cessna_comparison.py first). Writes fig_cessna_comparison.pdf and .png."""

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
d = np.load(os.path.join(HERE, "cessna_comparison.npz"))

# Okabe-Ito blue, vermillion, green
methods = [("albcd", "ALBCD", "#0072B2", "-", "o"),
           ("alc", "ALC", "#D55E00", (0, (5, 2)), "s"),
           ("admm", r"ADMM ($\mu = 10$)", "#009E73", (0, (1, 1.2)), "^")]

fig, (opt, feas, err) = plt.subplots(1, 3, figsize=(7, 2), sharex=True)
for k, (name, label, color, ls, marker) in enumerate(methods):
    sweeps = np.arange(1, int(d[f"{name}_sweeps"]) + 1)
    style = dict(color=color, ls=ls, marker=marker, lw=1.2, ms=3, mfc="w", mew=0.8,
                 markevery=(0.02 * k, 0.1))  # every 10% of the axes diagonal, staggered by line
    opt.plot(sweeps, d[f"{name}_opt_history"], label=label, **style)
    feas.plot(sweeps, d[f"{name}_feas_history"], **style)
    err.plot(sweeps, d[f"{name}_error"], **style)
    # where each method stops
    for a, key in ((opt, "opt_history"), (feas, "feas_history"), (err, "error")):
        a.plot(sweeps[-1], d[f"{name}_{key}"][-1], marker=marker, color=color, ms=4.5, ls="none")

for a, label in ((opt, "Optimality residual"), (feas, "Feasibility"), (err, "Relative error in $x$")):
    a.set_yscale("log")
    a.xaxis.set_minor_locator(NullLocator())
    a.yaxis.set_minor_locator(NullLocator())
    a.set_ylabel(label)
    a.set_xlabel("Iteration")

fig.legend(*opt.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
           handlelength=2.6, frameon=False)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_cessna_comparison.pdf"), bbox_inches="tight", pad_inches=0.02)
fig.savefig(os.path.join(HERE, "fig_cessna_comparison.png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
plt.show()
