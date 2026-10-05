"""Scaling study: ALBCD vs. monolithic SLSQP solve time as the fleet grows.

For each fleet size N, runs monolithic.py and then airliner_fleet.py (one after the other,
so the timings don't compete for the CPU) and reads the solve times they save, which
exclude JAX compilation. Results already saved (monolithic_solution_N{N}.npz,
convergence_N{N}.npz) are reused; delete them to rerun. A run that exceeds `time_limit` is
stopped and its time recorded in scaling.npz as a lower bound, and isn't retried while that
record exists. Writes scaling.npz and scaling.pdf.
"""

import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sizes = [1, 2, 4, 8, 16]
time_limit = 1800.0  # s per run

path = lambda name, n: os.path.join(HERE, f"{name}_N{n}.npz")
times = {}  # (name, N) -> (solve time, finished)

# runs that timed out before, recorded in scaling.npz, aren't retried (delete scaling.npz to retry them)
timed_out = set()
if os.path.exists(os.path.join(HERE, "scaling.npz")):
    saved = np.load(os.path.join(HERE, "scaling.npz"))
    for name, key in (("monolithic_solution", "monolithic"), ("convergence", "albcd")):
        for n, t, done in zip(saved["N"], saved[f"{key}_time"], saved[f"{key}_finished"]):
            if not done:
                timed_out.add((name, int(n)))
                times[name, int(n)] = (float(t), False)

for n in sizes:
    for script, name in (("monolithic.py", "monolithic_solution"), ("airliner_fleet.py", "convergence")):
        if (name, n) in timed_out:
            continue
        if not os.path.exists(path(name, n)):
            print(f"N = {n}: {script}", flush=True)
            try:
                subprocess.run([sys.executable, os.path.join(HERE, script), str(n)], check=True, timeout=time_limit,
                               env=dict(os.environ, MPLBACKEND="Agg"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                times[name, n] = (time_limit, False)
                continue
        times[name, n] = (float(np.load(path(name, n))["time"]), True)

monolithic_time, monolithic_finished = np.array([times["monolithic_solution", n] for n in sizes]).T
albcd_time, albcd_finished = np.array([times["convergence", n] for n in sizes]).T
np.savez(os.path.join(HERE, "scaling.npz"), N=sizes, monolithic_time=monolithic_time, albcd_time=albcd_time,
         monolithic_finished=monolithic_finished, albcd_finished=albcd_finished)

for n, tm, fm, ta, fa in zip(sizes, monolithic_time, monolithic_finished, albcd_time, albcd_finished):
    show = lambda t, done: f"{t:7.1f} s" if done else f"> {t:.0f} s"
    print(f"N = {n:2d}: monolithic {show(tm, fm)}, ALBCD {show(ta, fa)}")

fig, ax = plt.subplots(figsize=(4, 3))
for t, done, label, color in ((monolithic_time, monolithic_finished, "Monolithic SLSQP", "tab:blue"),
                              (albcd_time, albcd_finished, "ALBCD", "tab:orange")):
    ax.loglog(sizes, t, "-", linewidth=2, color=color, label=label)
    ax.loglog(np.array(sizes)[done == 1], t[done == 1], "o", color=color)
    ax.loglog(np.array(sizes)[done == 0], t[done == 0], "^", color=color, mfc="none", mew=2)  # lower bound
ax.set_xticks(sizes, [str(n) for n in sizes])
ax.set_xlabel("Flights N")
ax.set_ylabel("Solve time (s)")
ax.legend()
ax.grid(color="lavender", which="both")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "scaling.pdf"), bbox_inches="tight")
plt.show()
