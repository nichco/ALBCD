import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from models import nvar, n_h, h0, hf, setup_flight, simulate

# plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.color": "0.92", "legend.frameon": False})

HERE = os.path.dirname(os.path.abspath(__file__))
N = 8

data = np.load(os.path.join(HERE, f"convergence_N{N}.npz"))
ranges, masses, blocks = data["ranges"], data["masses"], data["x"].reshape(N, nvar)

fig, ax = plt.subplots(figsize=(6.5, 2.25))
for color, i in zip(plt.cm.plasma(np.linspace(0, 0.9, N)), np.argsort(ranges)):  # colored by range
    fl = setup_flight(ranges[i], masses[i])
    out = simulate(jnp.asarray(blocks[i]), fl)
    ax.plot(np.asarray(fl["r_nodes"]) / 1e3, np.asarray(out["h"]) / 1e3, color=color, lw=1.5,
            label=f"{ranges[i] / 1e3:,.0f} km, {masses[i] / 1e3:.1f} t")
    # altitude control points (the two fixed ones at each end included) at their Greville abscissae
    ch = np.concatenate(([h0, h0], blocks[i][:n_h], [hf, hf]))
    ax.plot(np.asarray(fl["greville"]) / 1e3, ch / 1e3, "o", color=color, ms=2.5, alpha=0.6)
    # ax.plot(np.asarray(fl["r_nodes"]) / 1e3, np.asarray(out["h"]) / 1e3, lw=2,
    #             label=f"{ranges[i] / 1e3:,.0f} km, {masses[i] / 1e3:.1f} t")

ax.set_xlabel("Range (km)")
ax.set_ylabel("Altitude (km)")
ax.set_xlim(0, ranges.max() / 1e3)
ax.set_ylim(3, 15)
ax.legend(loc="lower center", ncol=2, fontsize=8, title_fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_trajectories.pdf"), bbox_inches="tight")
plt.show()
