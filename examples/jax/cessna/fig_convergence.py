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

d = np.load(os.path.join(HERE, "cessna_albcd.npz"))
sweeps = np.arange(1, len(d["feas_history"]) + 1)
style = dict(color="#0072B2", ls="-", marker="o", lw=1.2, ms=3, mfc="w", mew=0.8, markevery=0.1)  # a marker every 10% of the axes diagonal

fig, (opt, feas, err) = plt.subplots(1, 3, figsize=(7, 2), sharex=True)
opt.plot(sweeps, d["opt_history"], **style)
feas.plot(sweeps, d["feas_history"], **style)
err.plot(np.arange(len(d["error"])) / n, d["error"], **style)

for a, label in ((opt, "Optimality residual"), (feas, "Feasibility"), (err, "Relative error in $x$")):
    a.set_yscale("log")
    a.xaxis.set_minor_locator(NullLocator())
    a.yaxis.set_minor_locator(NullLocator())
    a.set_ylabel(label)
    a.set_xlabel("Iteration")

fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_convergence.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.show()
