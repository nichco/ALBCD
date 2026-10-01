"""Plot subproblem solves needed to reach each MDO solution-error target."""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
PATTERN = os.path.join(HERE, "mdo_itr_v_error_N_*_n0_*_ni_*.npz")
FILE_RE = re.compile(r"mdo_itr_v_error_N_(\d+)_n0_(\d+)_ni_(\d+)\.npz$")


files = []
for path in glob.glob(PATTERN):
    match = FILE_RE.match(os.path.basename(path))
    if match:
        N, n0, ni = (int(value) for value in match.groups())
        files.append((N, n0, ni, path))

if not files:
    raise FileNotFoundError(f"No scaling results found matching {PATTERN}")

files.sort()
fig, ax = plt.subplots(figsize=(5.5, 3.5))
for N, n0, ni, path in files:
    data = np.load(path)
    reached = data["reached"].astype(bool)
    if np.any(reached):
        ax.plot(data["target"][reached], data["solves"][reached], "o-",
                linewidth=2, markersize=6, label=f"N={N}")

ax.set_xscale("log")
ax.invert_xaxis()
ax.set_ylim(bottom=0)
ax.set_xlabel("Solution error")
ax.set_ylabel("Subproblem solves")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
ax.grid(color="lavender")
plt.tight_layout()
plt.show()
