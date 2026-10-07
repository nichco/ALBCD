import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 8,
                     "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True,
                     "ytick.major.size": 3, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "pdf.fonttype": 42})

HERE = os.path.dirname(os.path.abspath(__file__))
sizes = [n for n in (2, 4, 6, 8, 10) if os.path.exists(os.path.join(HERE, f"fleet_albcd_N{n}.npz"))
         and os.path.exists(os.path.join(HERE, f"monolithic_solution_N{n}.npz"))]
results = {method: [np.load(os.path.join(HERE, f"{prefix}_N{n}.npz")) for n in sizes]
           for method, prefix in (("monolithic", "monolithic_solution"), ("albcd", "fleet_albcd"))}

fig, ax = plt.subplots(1, 2, figsize=(4.5, 2))
# Okabe-Ito blue and vermillion, with line style and marker so the figure also reads in grayscale
for method, label, color, ls, marker in (("monolithic", "Monolithic", "#0072B2", "-", "o"),
                                         ("albcd", "ALBCD", "#D55E00", (0, (5, 2)), "s")):
    style = dict(color=color, ls=ls, marker=marker, lw=1.2, ms=3, mfc="w", mew=0.8)
    ax[0].plot(sizes, [float(d["time"]) for d in results[method]], label=label, **style)
    ax[1].plot(sizes, [float(d["peak_memory"]) / 1024 for d in results[method]], **style)

ax[0].set_yscale("log")
ax[0].yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda t, _: f"{t:g}"))
ax[0].set_ylabel("Solve time (s)")
ax[0].legend(loc="upper left", handlelength=2.6)
ax[1].set_ylabel("Peak memory (GB)")
ax[1].set_ylim(0, None)

for a, tag in zip(ax, "ab"):
    a.set_xticks(sizes)
    a.xaxis.set_minor_locator(NullLocator())
    a.yaxis.set_minor_locator(NullLocator())
    a.set_xlabel("Number of flights $N$")
    a.set_title(f"({tag})", fontsize=10, loc="left")

fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_scaling.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.show()
