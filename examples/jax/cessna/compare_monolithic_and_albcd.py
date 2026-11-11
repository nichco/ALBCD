"""Plot optimality against wall time for the monolithic OpenSQP solve and the ALBCD solve of the
Cessna wing problem, from monolithic_opensqp_history.npz and albcd_timing_history.npz (run
monolithic_opensqp.py and albcd_timing.py first). Writes fig_monolithic_vs_albcd.pdf and .png.

Both curves use the same measure, the KKT residual of the monolithic problem (kkt_measure() in
monolithic_opensqp.py), at every OpenSQP major iteration and after every ALBCD block solve.
The times exclude JAX's compile time.
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

# Okabe-Ito blue, vermillion
methods = [("albcd_timing_history.npz", "ALBCD", "#0072B2", "-", "o"),
           ("monolithic_opensqp_history.npz", "Monolithic (OpenSQP)", "#D55E00", (0, (5, 2)), "s")]

fig, ax = plt.subplots(figsize=(3.5, 2.4))
for k, (file, label, color, ls, marker) in enumerate(methods):
    d = np.load(os.path.join(HERE, file))
    t, optimality = d["time"], d["optimality"]
    ax.plot(t, optimality, label=label, color=color, ls=ls, marker=marker, lw=1.2, ms=3, mfc="w", mew=0.8,
            markevery=(0.02 * k, 0.1))  # every 10% of the axes diagonal, staggered by line
    ax.plot(t[-1], optimality[-1], marker=marker, color=color, ms=4.5, ls="none")  # where each method stops
    print(f"{label}: {t[-1]:.2f} s, final optimality {optimality[-1]:.2e}, "
          f"final feasibility {d['feasibility'][-1]:.2e}")

ax.set_yscale("log")
ax.set_xlim(left=0)
ax.yaxis.set_minor_locator(NullLocator())
ax.set_xlabel("Wall time (s)")
ax.set_ylabel("Optimality")
ax.legend(frameon=False, handlelength=2.6)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_monolithic_vs_albcd.pdf"), bbox_inches="tight", pad_inches=0.02)
fig.savefig(os.path.join(HERE, "fig_monolithic_vs_albcd.png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
plt.show()
