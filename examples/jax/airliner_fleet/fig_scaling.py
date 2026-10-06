"""Figure 2: solve time and peak memory against the number of flights.

Reads monolithic_solution_N{N}.npz and fleet_albcd_N{N}.npz (run monolithic.py and airliner_fleet.py)
for every N that has both, and writes fig_scaling.pdf. Solve times exclude JAX compilation; peak memory
is the process's peak resident memory, imports and compilation included.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "0.92", "legend.frameon": False})

HERE = os.path.dirname(os.path.abspath(__file__))
sizes = sorted(int(f.split("_N")[-1][:-4]) for f in glob.glob(os.path.join(HERE, "fleet_albcd_N*.npz"))
               if os.path.exists(os.path.join(HERE, f"monolithic_solution_N{f.split('_N')[-1]}")))
results = {method: [np.load(os.path.join(HERE, f"{prefix}_N{n}.npz")) for n in sizes]
           for method, prefix in (("monolithic", "monolithic_solution"), ("albcd", "fleet_albcd"))}

fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.6))
for method, label, color in (("monolithic", "Monolithic SLSQP", "tab:blue"), ("albcd", "ALBCD", "tab:orange")):
    ax[0].plot(sizes, [float(d["time"]) for d in results[method]], "o-", color=color, ms=4, lw=1.3, label=label)
    ax[1].plot(sizes, [float(d["peak_memory"]) / 1024 for d in results[method]], "o-", color=color, ms=4, lw=1.3)

ax[0].set_yscale("log")
ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda t, _: f"{t:g}"))
ax[0].set_ylabel("Solve time (s)")
ax[0].legend(loc="upper left")
ax[1].set_ylabel("Peak memory (GB)")
ax[1].set_ylim(0, None)
for a in ax:
    a.set_xticks(sizes)
    a.set_xlabel("Number of flights $N$")

fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_scaling.pdf"), bbox_inches="tight")
plt.show()
