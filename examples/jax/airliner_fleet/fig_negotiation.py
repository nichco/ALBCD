"""Variants of a figure showing how ALBCD allocates the block-time budget among the flights (data from
negotiation_data.py, negotiation_N{N}.npz). Writes, as PDF and PNG:

    fig_negotiation_price     each flight's block time after every flight solve, against the time the other
                              flights leave it, above the price of block time (the multiplier y and the
                              effective price y + mu phi that each flight responds to)
    fig_negotiation_stacked   the block times stacked against the budget, and each flight's cut
    fig_negotiation_tradeoff  each flight's fuel against block time: its trade-off curve, the iterates on it,
                              and the tangent at the solution, whose slope is the cost index for every flight
    fig_negotiation_demand    each flight's block time against the price of block time (its demand curve), and
                              the fleet's total against the budget: the outer loop searches for the price
                              at which the flights' demands fit the budget

Flights are colored by range, as in fig_trajectories.py. Pass the number of flights as an argument (default 4).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import NullLocator, MaxNLocator
from scipy.interpolate import CubicHermiteSpline

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 8,
                     "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True,
                     "ytick.major.size": 3, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "pdf.fonttype": 42})

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
d = np.load(os.path.join(HERE, f"negotiation_N{N}.npz"))

flight, tau, fuel = d["flight"], d["tau"], 1e3 * d["fuel"]  # per flight solve; fuel in kg
y, mu, phi = d["y"], d["mu"], d["phi_after"]
T, tau0 = float(d["T_budget"]), d["tau0"]
tau_mono, ci_mono = d["tau_monolithic"], float(d["cost_index_monolithic"])
ranges = d["ranges"]
price = y + mu * phi  # the marginal price of block time each flight sees after its solve (kg/s)

# block times at every flight solve, with x0 first; the x axis counts sweeps, so solve k ends at (k + 1) / N
s = np.arange(len(flight) + 1) / N
tau_all = np.vstack((tau0, tau))
sweeps = len(d["feas_history"])

cmap = ListedColormap(plt.cm.plasma(np.linspace(0, 0.85, 256)))
norm = Normalize(ranges.min() / 1e6, ranges.max() / 1e6)
colors = [cmap(norm(r / 1e6)) for r in ranges]
names = [f"Flight {j} ({ranges[j] / 1e3:,.0f} km)" for j in range(N)]


def finish(fig, name, ncol=None):
    handles, labels = {}, []
    for a in fig.axes:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in handles:
                handles[l] = h
                labels.append(l)
    if labels:
        fig.legend([handles[l] for l in labels], labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                   ncol=ncol or len(labels), frameon=False, handlelength=2.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02, dpi=200)


def tag(a, text):
    a.set_title(text, fontsize=10, loc="left")


# (1) block times at the resolution of each flight solve, above the price of block time
fig = plt.figure(figsize=(7, 3.9))
gs = fig.add_gridspec(2, N, height_ratios=(1.25, 1), hspace=0.6, wspace=0.45)
for j in range(N):
    a = fig.add_subplot(gs[0, j])
    left = T - (tau_all.sum(axis=1) - tau_all[:, j])
    a.step(s, left, where="post", color="0.65", lw=0.9, label="Time left by the other flights")
    a.step(s, tau_all[:, j], where="post", color=colors[j], lw=1.4)
    if j == 0:
        a.plot([], [], color="0.2", lw=1.4, marker="o", ms=2.2, label="Own block time (dots: own solves)")
    turns = np.flatnonzero(flight == j) + 1  # this flight's own solves
    a.plot(s[turns], tau_all[turns, j], ls="none", marker="o", ms=2.2, color=colors[j])
    a.axhline(tau_mono[j], color="k", lw=0.6, ls=":", label="Monolithic")
    tag(a, f"({'abcdefgh'[j]}) Flight {j}, {ranges[j] / 1e3:,.0f} km")
    a.set_xlabel("Sweep")
    a.set_xlim(0, sweeps)
    a.xaxis.set_minor_locator(NullLocator())
    if j == 0:
        a.set_ylabel("Block time (ks)")
b = fig.add_subplot(gs[1, :])
b.plot(s[1:], price, color="0.35", lw=1.0, label=r"Effective price $y + \mu\,\phi$")
b.step(s[1:], y, where="post", color="k", lw=1.4, label="Multiplier $y$ (cost index)")
b.axhline(ci_mono, color="k", lw=0.6, ls=":")
b.text(sweeps, ci_mono, " monolithic", va="center", ha="left", fontsize=8)
tag(b, f"({'abcdefgh'[N]}) Price of block time")
b.set(xlabel="Sweep", ylabel="Price (kg/s)", xlim=(0, sweeps), ylim=(0, None))
b.xaxis.set_minor_locator(NullLocator())
finish(fig, "fig_negotiation_price", ncol=3)


# (2) the block times stacked against the budget, and each flight's cut from its initial block time
fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.5), gridspec_kw={"width_ratios": (1.2, 1), "wspace": 0.3})
a.stackplot(s, tau_all.T, colors=colors, edgecolor="w", lw=0.4, labels=names)
a.axhline(T, color="k", ls="--", lw=1)
a.text(sweeps, T, "Budget", va="bottom", ha="right", fontsize=8)
a.set(xlabel="Sweep", ylabel="Cumulative block time (ks)", xlim=(0, sweeps), ylim=(0, 1.08 * tau_all.sum(axis=1).max()))
tag(a, "(a)")
for j in range(N):
    b.step(s, tau0[j] - tau_all[:, j], where="post", color=colors[j], lw=1.4)
    b.axhline(tau0[j] - tau_mono[j], color=colors[j], lw=0.6, ls=":")
b.set(xlabel="Sweep", ylabel="Cut from initial block time (ks)", xlim=(0, sweeps))
tag(b, "(b)")
for x in (a, b):
    x.xaxis.set_minor_locator(NullLocator())
finish(fig, "fig_negotiation_stacked", ncol=2)


# (3) each flight's fuel against its block time, interpolated with the exact slopes d(fuel)/d(block time)
curve_tau, curve_fuel = d["curve_tau"], 1e3 * d["curve_fuel"]
splines = [CubicHermiteSpline(t, f, 1e3 * g) for t, f, g in zip(curve_tau, curve_fuel, d["curve_slope"])]
min_fuel = d["min_fuel"]
sweep_cmap = plt.cm.Greys
fig, ax = plt.subplots(1, N, figsize=(7, 2.3), gridspec_kw={"wspace": 0.5})
for j, a in enumerate(ax):
    t_fine = np.linspace(curve_tau[j][0], curve_tau[j][-1], 300)
    a.plot(t_fine, splines[j](t_fine), color=colors[j], lw=1.4, zorder=1)
    if j == 0:
        a.plot([], [], color="0.2", lw=1.4, label="Minimum fuel at fixed block time")
    turns = np.flatnonzero(flight == j)
    a.scatter(tau[turns, j], fuel[turns, j], c=(turns + 1) / N, cmap=sweep_cmap, vmin=-0.3 * sweeps, vmax=sweeps,
              s=12, edgecolors="k", linewidths=0.4, zorder=3, label="Iterates")
    # tangent at the monolithic block time, with slope -cost index (kg/s, i.e. 1000 kg per ks)
    t = tau_mono[j] + np.array([-0.3, 0.3])
    f_mono = float(splines[j](tau_mono[j]))
    a.plot(t, f_mono - 1e3 * ci_mono * (t - tau_mono[j]), color="k", lw=0.8, ls=(0, (4, 2)), zorder=2,
           label=f"Slope $-${ci_mono:.3f} kg/s (cost index)")
    a.plot(min_fuel[j, 0], 1e3 * min_fuel[j, 1], ls="none", marker="v", color="k", ms=4, label="Minimum fuel", zorder=4)
    tag(a, f"({'abcdefgh'[j]}) Flight {j}")
    a.set_xlabel("Block time (ks)")
    lo, hi = min(tau[turns, j].min(), tau_mono[j]) - 0.25, max(tau[turns, j].max(), min_fuel[j, 0]) + 0.25
    a.set_xlim(lo, hi)
    a.xaxis.set_major_locator(MaxNLocator(3))
    inside = (curve_tau[j] >= lo) & (curve_tau[j] <= hi)
    span = curve_fuel[j][inside].max() - curve_fuel[j][inside].min()
    a.set_ylim(curve_fuel[j][inside].min() - 0.1 * span, curve_fuel[j][inside].max() + 0.1 * span)
    a.xaxis.set_minor_locator(NullLocator())
ax[0].set_ylabel("Fuel (kg)")
cb = fig.colorbar(ScalarMappable(Normalize(-0.3 * sweeps, sweeps), sweep_cmap), ax=ax, pad=0.02, aspect=25)
cb.set_label("Sweep")
cb.ax.set_ylim(0, sweeps)
cb.outline.set_visible(False)
finish(fig, "fig_negotiation_tradeoff", ncol=2)


# (4) demand curves: each flight's block time against the price of block time, and the fleet's total
p_curve = -d["curve_slope"]  # -d(fuel)/d(block time) (kg/s), decreasing in block time
fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.6), gridspec_kw={"wspace": 0.3})
p_grid = np.linspace(0, 1.4 * max(ci_mono, price.max()), 200)
total = np.zeros_like(p_grid)
for j in range(N):
    order = np.argsort(p_curve[j])
    demand = np.interp(p_grid, p_curve[j][order], curve_tau[j][order], left=np.nan, right=np.nan)
    total += demand
    a.plot(p_grid, demand, color=colors[j], lw=1.4, label=names[j])
    turns = np.flatnonzero(flight == j)
    a.plot(price[turns], tau[turns, j], ls="none", marker="o", ms=3, mfc="w", mew=0.8, color=colors[j])
for x in (a, b):
    x.axvline(ci_mono, color="k", lw=0.6, ls=":")
a.set(xlabel="Price of block time (kg/s)", ylabel="Block time (ks)")
tag(a, "(a) Each flight")
b.plot(p_grid, total, color="k", lw=1.4, label="Total of the flights' block times")
b.axhline(T, color="k", ls="--", lw=1, label="Budget")
ends = np.arange(1, sweeps + 1) * N - 1  # last solve of each sweep
b.plot(price[ends], tau[ends].sum(axis=1), ls="none", marker="o", ms=3, mfc="w", mew=0.8, color="0.3", label="End of each sweep")
b.set(xlabel="Price of block time (kg/s)", ylabel="Total block time (ks)")
tag(b, "(b) Fleet")
for x in (a, b):
    x.set_xlim(0, p_grid[-1])
finish(fig, "fig_negotiation_demand", ncol=3)

plt.show()
