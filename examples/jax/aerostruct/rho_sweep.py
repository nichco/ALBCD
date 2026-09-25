"""Plot the saved ALBCD convergence of the aerostructural problem for several penalty growth factors rho.

Reads aerostruct_albcd_rho{rho}.npz, written by aerostruct.py with ALBCD's rho set to
that value, so the solves don't need to be rerun. Saves the figure to rho_sweep.pdf.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

RHOS = (1.05, 1.24, 1.43, 1.62, 1.81, 2.0)
COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 2.5))

for i, (rho, color) in enumerate(zip(RHOS, COLORS)):
    path = os.path.join(HERE, f"aerostruct_albcd_rho{rho}.npz")
    if not os.path.exists(path):
        print(f"No convergence data for rho = {rho}; run aerostruct.py with rho={rho} to create it.")
        continue
    data = np.load(path)

    # opt_history and feas_history describe x after each subproblem solve, while error
    # also starts at x0, so error[i] and the other histories' entry i - 1 share an iteration
    iterations = np.arange(1, len(data["feas_history"]) + 1)
    # larger rho drawn further back, so its wider optimality oscillations don't hide the smaller
    # rho curves; every zorder stays above the default line zorder of 2, and so above the grid
    ax1.semilogy(iterations, data["opt_history"], color=color, linewidth=2, label=rf"$\rho$ = {rho}", alpha=0.8,
                 zorder=2 + len(RHOS) - i)
    ax2.semilogy(iterations, data["feas_history"], color=color, linewidth=2)
    ax3.semilogy(np.arange(len(data["error"])), data["error"], color=color, linewidth=2)

for ax, label in zip((ax1, ax2, ax3), ("Optimality", "Feasibility", "Relative error")):
    ax.set_xlabel("Iteration")
    ax.set_ylabel(label)
    ax.grid(color="lavender", alpha=0.5, axis="y")
# one row above the panels, since the optimality oscillations fill every corner of ax1
fig.legend(*ax1.get_legend_handles_labels(), loc="upper center", ncol=len(RHOS), fontsize=9, frameon=False)

plt.tight_layout(rect=(0, 0, 1, 0.9))
plt.savefig(os.path.join(HERE, "aerostruct_rho_sweep.pdf"))
plt.show()
