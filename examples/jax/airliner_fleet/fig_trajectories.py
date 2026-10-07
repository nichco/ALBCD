"""Figure: the airliner's altitude profiles on each flight against time, with the block-time allocations against the budget.

Solid lines are the budget-constrained solution (fleet_albcd_N{N}.npz, from airliner_fleet.py) and dashed
lines each flight's minimum-fuel trajectory (minimum_fuel_N{N}.npz, from minimum_fuel.py), colored by range.
The lower panel stacks the flights' block-time allocations against the budget. Writes fig_trajectories.pdf.
"""

import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable

from models import nvar, setup_flight, simulate

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 8,
                     "pdf.fonttype": 42, "axes.grid": True,
                     "axes.grid.axis": "y", "grid.color": "0.92", "axes.axisbelow": True})

HERE = os.path.dirname(os.path.abspath(__file__))
N = 8

data = np.load(os.path.join(HERE, f"fleet_albcd_N{N}.npz"))
ranges, masses, blocks = data["ranges"], data["masses"], data["x"].reshape(N, nvar)
blocks_min_fuel = np.load(os.path.join(HERE, f"minimum_fuel_N{N}.npz"))["x"].reshape(N, nvar)
T_budget = 0.9 * np.sum(0.276 + 5.468e-6 * ranges)  # ks, as in airliner_fleet.py

cmap = ListedColormap(plt.cm.plasma(np.linspace(0, 0.85, 256)))
norm = Normalize(ranges.min() / 1e6, ranges.max() / 1e6)
dashed = dict(lw=0.9, ls=(0, (3, 2)), alpha=0.7)

fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.5, 2.75), gridspec_kw={"height_ratios": [3, 0.35], "hspace": 0.55})
left, t_max = 0.0, 0.0  # left: block-time allocations stacked against the budget in the lower panel
for i in np.argsort(ranges):
    color = cmap(norm(ranges[i] / 1e6))
    fl = setup_flight(ranges[i], masses[i])
    for v, style in ((blocks_min_fuel[i], dashed), (blocks[i], dict(lw=1.4))):
        out = simulate(jnp.asarray(v), fl)
        ax.plot(np.asarray(out["t"]) / 1e3, np.asarray(out["h"]) / 1e3, color=color, **style)
        t_max = max(t_max, float(out["t"][-1]) / 1e3)
    bx.barh(0, blocks[i][-1], left=left, height=0.7, color=color, edgecolor="w", lw=0.6)
    left += blocks[i][-1]

ax.plot([], [], "k-", lw=1.4, label="Time limit")
ax.plot([], [], "k", label="Minimum fuel", **dashed)
ax.legend(loc="lower center", ncol=2, framealpha=0.9)
ax.set(xlabel="Time (ks)", ylabel="Altitude (km)", xlim=(0, 1.01 * t_max), ylim=(3, 15))

bx.axvline(T_budget, color="k", ls="--", lw=1)
bx.text(T_budget, 1.02, "Time limit", transform=bx.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
bx.set(xlabel="Cumulative block time (ks)", yticks=[], xlim=(0, 1.1 * max(left, T_budget)))
bx.grid(False)
# bx.spines["left"].set_visible(False)

cb = fig.colorbar(ScalarMappable(norm, cmap), ax=[ax, bx], pad=0.04, aspect=30)
cb.set_label("Range (1000 km)")
cb.outline.set_visible(False)

fig.savefig(os.path.join(HERE, "fig_trajectories.pdf"), bbox_inches="tight")
plt.show()
