# Distribution Statement A. Approved for public release: distribution is unlimited. Approved AFRL-2026-1671 28-09-2026.

"""Run quadratic_consensus_linear.py, quadratic_consensus_circle.py and
quadratic_global_linear.py, and plot their iterates side by side in
fig_quadratic_iterates.pdf.
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


consensus_linear = np.array(run("quadratic_consensus_linear.py").x_history)  # (x1_1, x2_1, x1_2, x2_2)
consensus_circle = np.array(run("quadratic_consensus_circle.py").x_history)  # (x1_1, x2_1, x1_2, x2_2)
global_linear = np.array(run("quadratic_global_linear.py").x_history)  # (x1, s, x2)

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
                     "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "xtick.major.size": 3, "ytick.major.size": 3,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})

BLUE, VERMILLION = "#0072B2", "#D55E00"  # Okabe-Ito, colorblind safe
LIM = 1.5

t = np.linspace(-LIM, LIM, 400)
X, Y = np.meshgrid(t, t)
Z = X**2 + Y**2 - 1.5 * X * Y


def background(ax):
    ax.contour(X, Y, Z, levels=(0.2 * np.arange(1, 15))**2, colors="0.6", linewidths=0.4, zorder=1)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel("$x$")


def linear_constraint(ax):
    """Infeasible region of x - 0.5 y >= 0, i.e. above y = 2x."""
    ax.fill_between(t, 2 * t, LIM, color="0.85", lw=0, zorder=0)
    ax.plot(t, 2 * t, color="k", lw=0.8, zorder=2)


def circle_constraint(ax):
    """Infeasible region of x^2 + y^2 >= 0.25, i.e. inside the circle of radius 0.5."""
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.fill(0.5 * np.cos(theta), 0.5 * np.sin(theta), color="0.85", lw=0, zorder=0)
    ax.plot(0.5 * np.cos(theta), 0.5 * np.sin(theta), color="k", lw=0.8, zorder=2)


def path(ax, x, y, color, marker, label):
    ax.plot(x, y, color=color, marker=marker, lw=1.0, ms=3.5, mfc="w", mew=0.8, label=label, zorder=4)
    ax.plot(x[0], y[0], color=color, marker=marker, ms=3.5, mew=0.8, ls="none", zorder=5)  # filled start point


fig, axes = plt.subplots(1, 3, figsize=(7, 2.6), sharey=True)

ax = axes[0]
background(ax)
linear_constraint(ax)
path(ax, consensus_linear[:, 0], consensus_linear[:, 3], VERMILLION, "s", "$(x_1, y_2)$")
path(ax, consensus_linear[:, 2], consensus_linear[:, 1], BLUE, "o", "$(x_2, y_1)$")

ax = axes[1]
background(ax)
circle_constraint(ax)
path(ax, consensus_circle[:, 0], consensus_circle[:, 3], VERMILLION, "s", "$(x_1, y_2)$")
path(ax, consensus_circle[:, 2], consensus_circle[:, 1], BLUE, "o", "$(x_2, y_1)$")

ax = axes[2]
background(ax)
linear_constraint(ax)
path(ax, global_linear[:, 0], global_linear[:, 2], BLUE, "o", "$(x, y)$")

axes[0].set_ylabel("$y$")
for ax in axes:
    ax.legend(loc="upper right")

fig.tight_layout(w_pad=1.0)
fig.savefig(os.path.join(HERE, "fig_albcd_demo.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.show()
