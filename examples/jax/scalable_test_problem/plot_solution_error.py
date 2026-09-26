"""Plots the subproblem solves ALBCD needs to reach each solution error, for every N.

Loads the itr_v_error_N_*_n0_10_ni_20.npz files written by solution_error.py.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

files = glob.glob(os.path.join(HERE, "itr_v_error_N_*_n0_10_ni_20.npz"))
cases = sorted((int(os.path.basename(f).split("_")[4]), np.load(f)) for f in files)

fig, ax = plt.subplots(figsize=(5.5, 3.5))
for N, data in cases:
    ax.plot(data["target"], data["solves"], "o-", linewidth=2, markersize=6, label=f"N={N}")
ax.set_xscale("log")
ax.invert_xaxis() # tighter errors to the right
ax.set_ylim(bottom=0)
ax.set_xlabel("Solution error")
ax.set_ylabel("Iterations")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1, 1)) # largest N on top, as plotted
ax.grid(color="lavender")
plt.tight_layout()
plt.show()
