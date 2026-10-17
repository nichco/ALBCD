# Distribution Statement A. Approved for public release: distribution is unlimited. Approved AFRL-2026-1671 28-09-2026.

"""Run quadratic_global_failure.py and no_feasible_solution.py, and plot their BCD
iterates side by side in bcd_failure_demo.pdf, with the lines where each block is
optimal and the infeasible regions of the coupling constraints.
"""

import os
import runpy
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def run(name):
    """Run an example script without showing its own plot, and return its ALBCD instance."""
    show, plt.show = plt.show, lambda *args, **kwargs: None
    try:
        opt = runpy.run_path(os.path.join(HERE, name), run_name="__main__")["opt"]
    finally:
        plt.show = show
        plt.close("all")
    return opt


global_failure = np.array(run("quadratic_global_failure.py").x_history)  # (x1, x2)
no_feasible = np.array(run("no_feasible_solution.py").x_history)  # (x1, x2)

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
                     "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "xtick.major.size": 3, "ytick.major.size": 3,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})

BLUE, PURPLE, GREEN = "#0072B2", "#CC79A7", "#009E73"  # Okabe-Ito, colorblind safe
LIM = 1.5

t = np.linspace(-LIM, LIM, 400)
X, Y = np.meshgrid(t, t)
Z = X**2 + Y**2 - 1.5 * X * Y


def background(ax):
    ax.contour(X, Y, Z, levels=(0.2 * np.arange(1, 15))**2, colors="0.6", linewidths=0.4, zorder=1)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")


def block_optimal(ax):
    """Zero level sets of the partial derivatives: the lines where block 1 and block 2 are optimal."""
    ax.contour(X, Y, 2 * X - 1.5 * Y, levels=[0], colors=PURPLE, linewidths=1.0, linestyles="--", zorder=2)
    ax.contour(X, Y, 2 * Y - 1.5 * X, levels=[0], colors=GREEN, linewidths=1.0, linestyles="-.", zorder=2)
    ax.plot([], [], color=PURPLE, lw=1.0, ls="--", label=r"$\partial f / \partial x = 0$")  # legend proxies
    ax.plot([], [], color=GREEN, lw=1.0, ls="-.", label=r"$\partial f / \partial y = 0$")


def infeasible(ax, y_line, y_edge):
    """Infeasible region between the constraint line y_line and the plot edge y_edge."""
    ax.fill_between(t, y_line, y_edge, color="0.85", lw=0, zorder=0)
    ax.plot(t, y_line, color="k", lw=0.8, zorder=2)


def path(ax, x, y):
    ax.plot(x, y, color=BLUE, marker="o", lw=1.0, ms=3.5, mfc="w", mew=0.8, label="Iterates", zorder=4)
    ax.plot(x[0], y[0], color=BLUE, marker="o", ms=3.5, mew=0.8, ls="none", zorder=5)  # filled start point


# square panels placed in inches, as in fig_bcd_demo.py
# (with room for the tick and axis labels; bbox_inches="tight" trims the rest)
side, left, gap, bottom = 2.0, 0.55, 1.0, 0.45
width, height = left + 2 * side + gap + 0.1, bottom + side + 0.1
fig = plt.figure(figsize=(width, height))
axes = [fig.add_axes([(left + i * (side + gap)) / width, bottom / height, side / width, side / height])
        for i in range(2)]

# x - 0.5 y >= 0, as in quadratic_global_failure.py: BCD stalls on the constraint line
ax = axes[0]
background(ax)
infeasible(ax, 2 * t, LIM)
block_optimal(ax)
path(ax, global_failure[:, 0], global_failure[:, 1])

# x - 0.5 y >= 0 and x + 0.5 y >= 0, as in no_feasible_solution.py: BCD zigzags to the origin
ax = axes[1]
background(ax)
infeasible(ax, 2 * t, LIM)
infeasible(ax, -2 * t, -LIM)
block_optimal(ax)
path(ax, no_feasible[:, 0], no_feasible[:, 1])

for ax in axes:
    ax.legend(loc="upper left")  # over the infeasible region, clear of the iterates

fig.savefig(os.path.join(HERE, "bcd_failure_demo.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.show()
