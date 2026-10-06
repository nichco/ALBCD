"""Figure 2: solve time and ALBCD block solves per flight against the number of flights.

Reads scaling_runs.json (run scaling.py first) and writes fig_scaling.pdf. Markers show the
median over the repeated runs and the bars their range; open triangles are runs stopped at
the time limit, so lower bounds.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "0.92", "legend.frameon": False})

HERE = os.path.dirname(os.path.abspath(__file__))
runs = json.load(open(os.path.join(HERE, "scaling_runs.json")))
sizes = sorted({r["N"] for r in runs})

fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.6))
for method, label, color in (("monolithic", "Monolithic SLSQP", "tab:blue"), ("albcd", "ALBCD", "tab:orange")):
    done = [n for n in sizes if any(r["method"] == method and r["N"] == n and r["finished"] for r in runs)]
    times = [np.array([r["time"] for r in runs if r["method"] == method and r["N"] == n]) for n in done]
    median = np.array([np.median(t) for t in times])
    spread = [median - [t.min() for t in times], [t.max() for t in times] - median]
    ax[0].errorbar(done, median, yerr=spread, fmt="o-", color=color, ms=4, lw=1.3, capsize=2, label=label)
    stopped = sorted({r["N"] for r in runs if r["method"] == method and not r["finished"]})
    if stopped:
        limit = max(r["time"] for r in runs if r["method"] == method and not r["finished"])
        ax[0].plot(stopped, [limit] * len(stopped), "^", color=color, mfc="none", ms=6, mew=1.3)

    if method == "albcd":
        per_flight = [np.median([r["count"] for r in runs if r["method"] == method and r["N"] == n]) / n for n in done]
        ax[1].plot(done, per_flight, "o-", color=color, ms=4, lw=1.3)

ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xticks(sizes, [str(n) for n in sizes], minor=False)
ax[0].xaxis.set_minor_locator(plt.NullLocator())
ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda t, _: f"{t:g}"))
ax[0].yaxis.set_minor_formatter(plt.FuncFormatter(lambda t, _: f"{t:g}" if f"{t:g}"[0] in "25" else ""))
ax[0].set_xlabel("Number of flights $N$")
ax[0].set_ylabel("Solve time (s)")
ax[0].legend()

ax[1].set_xticks(sizes)
ax[1].set_ylim(0, 1.2 * ax[1].get_ylim()[1])
ax[1].set_xlabel("Number of flights $N$")
ax[1].set_ylabel("ALBCD block solves per flight")

fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_scaling.pdf"), bbox_inches="tight")
plt.show()
