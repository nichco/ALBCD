"""Plot the tight-tolerance ALBCD / ALC / ADMM comparison on the Cessna problem from
cessna_comparison_tight.npz (run cessna_comparison_tight.py first). Writes
fig_cessna_comparison_tight.pdf and .png.

x axis: BCD sweeps, one solve of each block (aero, then structures), so a unit on the axis is the
same work for every method. Markers sit on the sweeps after which ALBCD and ALC update their
multipliers (each after an inner loop of one or more sweeps); ADMM updates after every sweep, so its
line has none. Grey lines: the feasibility and optimality tolerances.
"""

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
d = np.load(os.path.join(HERE, "cessna_comparison_tight.npz"))
FEAS_TOL, OPT_TOL = 1e-5, 3e-5  # as in cessna_comparison_tight.py

# Okabe-Ito blue, vermillion, green; (name, label, color, line style, marker, mark the updates)
methods = [("albcd", rf"ALBCD ($\mu_0 = {float(d['albcd_mu0']):.2g}$)", "#0072B2", "-", "o", True),
           ("alc", rf"ALC ($\mu_0 = {float(d['alc_mu0']):.2g}$)", "#D55E00", (0, (5, 2)), "s", True),
           ("admm", rf"ADMM ($\mu = {float(d['admm_mu0']):.2g}$)", "#009E73", (0, (1, 1.2)), "^", False)]

fig, (opt, feas, err) = plt.subplots(1, 3, figsize=(7, 2.1), sharex=True)
for name, label, color, ls, marker, mark_updates in methods:
    sweeps = np.arange(1, int(d[f"{name}_sweeps"]) + 1)
    every = list(d[f"{name}_updates"] - 1) if mark_updates else []  # indices of the update sweeps
    style = dict(color=color, ls=ls, lw=1.2, marker=marker, ms=3, mfc="w", mew=0.8, markevery=every)
    for a, key in ((opt, "opt_history"), (feas, "feas_history"), (err, "error")):
        a.plot(sweeps, d[f"{name}_{key}"], label=label if a is opt else None, **style)
        a.plot(sweeps[-1], d[f"{name}_{key}"][-1], marker=marker, color=color, ms=5, ls="none")  # where it stops

for a, label, tol in ((opt, "Optimality residual", OPT_TOL), (feas, "Feasibility", FEAS_TOL),
                      (err, "Relative error in $x$", None)):
    a.set_xscale("log")
    a.set_yscale("log")
    a.xaxis.set_minor_locator(NullLocator())
    a.yaxis.set_minor_locator(NullLocator())
    if tol:
        a.axhline(tol, color="0.6", lw=0.6, zorder=0)
    a.set_ylabel(label)
    a.set_xlabel("BCD sweep")

fig.legend(*opt.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
           handlelength=2.6, frameon=False)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_cessna_comparison_tight.pdf"), bbox_inches="tight", pad_inches=0.02)
fig.savefig(os.path.join(HERE, "fig_cessna_comparison_tight.png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
plt.show()
