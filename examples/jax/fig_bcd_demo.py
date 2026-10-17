# Distribution Statement A. Approved for public release: distribution is unlimited. Approved AFRL-2026-1671 28-09-2026.

"""Run 2d_rosenbrock.py and wing_design.py, and plot their BCD iterates side by side in
fig_bcd_demo.pdf, with the curves where each block is optimal.
"""

import os
import runpy
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def run(name):
    """Run an example script without showing its own plot, and return its globals."""
    show, plt.show = plt.show, lambda *args, **kwargs: None
    try:
        script = runpy.run_path(os.path.join(HERE, name), run_name="__main__")
    finally:
        plt.show = show
        plt.close("all")
    return script


rosenbrock = np.array(run("2d_rosenbrock.py")["opt"].x_history)  # (x1, x2)
wing = run("wing_design.py")
wing_history = np.array(wing["opt"].x_history)  # (span, chord)
power_required = wing["power_required"]

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
                     "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "xtick.major.size": 3, "ytick.major.size": 3,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})

BLUE, PURPLE, GREEN = "#0072B2", "#CC79A7", "#009E73"  # Okabe-Ito, colorblind safe


def contours(ax, X, Y, Z, levels):
    ax.contour(X, Y, Z, levels=levels, colors="0.6", linewidths=0.4, zorder=1)


def block_optimal(ax, X, Y, dZ_dx, dZ_dy, x_label, y_label):
    """Zero level sets of the partial derivatives: the curves where block 1 and block 2 are optimal."""
    ax.contour(X, Y, dZ_dx, levels=[0], colors=PURPLE, linewidths=1.0, linestyles="--", zorder=2)
    ax.contour(X, Y, dZ_dy, levels=[0], colors=GREEN, linewidths=1.0, linestyles="-.", zorder=2)
    ax.plot([], [], color=PURPLE, lw=1.0, ls="--", label=x_label)  # legend proxies for the contours
    ax.plot([], [], color=GREEN, lw=1.0, ls="-.", label=y_label)


def path(ax, x, y):
    ax.plot(x, y, color=BLUE, marker="o", lw=1.0, ms=3.5, mfc="w", mew=0.8, label="Iterates", zorder=4)
    ax.plot(x[0], y[0], color=BLUE, marker="o", ms=3.5, mew=0.8, ls="none", zorder=5)  # filled start point


# square panels the same size as in fig_quadratic_iterates.pdf, placed in inches
# (with room for the tick and axis labels; bbox_inches="tight" trims the rest)
side, left, gap, bottom = 2.0, 0.55, 1.0, 0.45
width, height = left + 2 * side + gap + 0.1, bottom + side + 0.1
fig = plt.figure(figsize=(width, height))
axes = [fig.add_axes([(left + i * (side + gap)) / width, bottom / height, side / width, side / height])
        for i in range(2)]

# Rosenbrock function with coefficient 1, as in 2d_rosenbrock.py
ax = axes[0]
t = np.linspace(-1.5, 1.5, 400)
X, Y = np.meshgrid(t, t)
Z = (1 - X)**2 + (Y - X**2)**2
contours(ax, X, Y, Z, (0.3 * np.arange(1, 16))**2)
block_optimal(ax, X, Y, -2 * (1 - X) - 4 * X * (Y - X**2), 2 * (Y - X**2),
              r"$\partial f / \partial x = 0$", r"$\partial f / \partial y = 0$")
path(ax, rosenbrock[:, 0], rosenbrock[:, 1])
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

# power required by the wing, as in wing_design.py
ax = axes[1]
b = np.linspace(5, 35, 400)
c = np.linspace(0.3, 1.5, 400)
B, C = np.meshgrid(b, c)
P = np.asarray(power_required(B, C))
dP_dc, dP_db = np.gradient(P, c, b)
contours(ax, B, C, P, P.min() + (3000 - P.min()) * np.linspace(0, 1, 16)[1:]**2)
block_optimal(ax, B, C, dP_db, dP_dc, r"$\partial P / \partial b = 0$", r"$\partial P / \partial c = 0$")
path(ax, wing_history[:, 0], wing_history[:, 1])
ax.set_xlim(5, 35)
ax.set_ylim(0.3, 1.5)
ax.set_xlabel("Span $b$ (m)")
ax.set_ylabel("Chord $c$ (m)")

for ax, loc in zip(axes, ("lower right", "upper right")):  # clear of the iterates
    ax.legend(loc=loc)

fig.savefig(os.path.join(HERE, "fig_bcd_demo.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.show()
