"""Plot the saved ALBCD convergence of the uncertain cart-pole problem for N = 2, 3 and 4.

Reads convergence_N{N}.npz, written by cart_pole.py (run `python cart_pole.py N` to
create one), so the solves don't need to be rerun. Saves the figure to
cart_pole_convergence.pdf.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 2.5))

for N, color in zip((2, 3, 4), ("tab:blue", "tab:orange", "tab:green")):
    path = os.path.join(HERE, f"convergence_N{N}.npz")
    if not os.path.exists(path):
        print(f"No convergence data for N = {N}; run `python cart_pole.py {N}` to create it.")
        continue
    data = np.load(path)

    # opt_history and feas_history describe x after each subproblem solve, while error
    # also starts at x0, so error[i] and the other histories' entry i - 1 share an iteration
    iterations = np.arange(1, len(data["feas_history"]) + 1)
    ax1.semilogy(iterations, data["opt_history"], color=color, linewidth=2, label=f"N = {N}")
    ax2.semilogy(iterations, data["feas_history"], color=color, linewidth=2)
    ax3.semilogy(np.arange(len(data["error"])), data["error"], color=color, linewidth=2)

# final tolerances, the same for every N
ax1.axhline(data["opt_tol"][-1], color="gray", linewidth=1, linestyle="--", alpha=0.8)
ax2.axhline(data["feas_tol"], color="gray", linewidth=1, linestyle="--", alpha=0.8)

for ax, label in zip((ax1, ax2, ax3), ("Optimality", "Feasibility", "Relative error")):
    ax.set_xlabel("Subproblem solve")
    ax.set_ylabel(label)
    ax.grid(color="lavender", alpha=0.5, axis="y")
ax1.legend(fontsize=10, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(HERE, "cart_pole_convergence.pdf"))
plt.show()
