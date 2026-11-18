"""Task 7 figures: theta of albcd, v2 and v3 against cumulative sweeps and against wall time, per problem.

Reads qn/results/qn_results_airliner.npz and qn_results_ranger.npz; writes qn/results/fig_theta_<problem>.png.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent / "results"
STYLE = {"albcd": dict(color="0.35", marker="s"), "v2": dict(color="tab:blue", marker="o"),
         "v3": dict(color="tab:orange", marker="^")}


def curves(d, prefix, name):
    ev = d[f"{prefix}{name}__event"]
    th = d[f"{prefix}{name}__theta"]
    t = d[f"{prefix}{name}__time"]
    sweeps = np.cumsum(ev == "sweep")
    ok = np.isfinite(th)
    return sweeps[ok], t[ok], th[ok], ev[ok]


def plot(d, prefix, title, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    for name, style in STYLE.items():
        if f"{prefix}{name}__event" not in d.files:
            continue
        s, t, th, ev = curves(d, prefix, name)
        for ax, x in ((a, s), (b, t)):
            ax.semilogy(x, th, lw=1.3, ms=3.5, label=name, **style)
    for ax, xl in ((a, "Cumulative sweeps"), (b, "Wall time (s)")):
        for level in (1e-4, 1e-8):
            ax.axhline(level, color="k", lw=0.6, ls="--")
        ax.set_xlabel(xl)
        ax.set_ylabel(r"KKT measure $\theta$")
        ax.grid(alpha=0.3)
    a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    air = RESULTS / "qn_results_airliner.npz"
    if air.exists():
        d = np.load(air)
        for N in (4, 8):
            if f"N{N}/names" in d.files:
                plot(d, f"N{N}/", f"Airliner fleet, N = {N}", RESULTS / f"fig_theta_airliner_N{N}.png")
    rng = RESULTS / "qn_results_ranger.npz"
    if rng.exists():
        plot(np.load(rng), "", "Ranger", RESULTS / "fig_theta_ranger.png")
