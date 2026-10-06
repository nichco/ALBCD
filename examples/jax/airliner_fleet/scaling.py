"""Scaling study: ALBCD vs. monolithic SLSQP solve time as the number of flights grows.

For each number of flights, runs monolithic.py and then airliner_fleet.py `repeats` times, one run at
a time so the timings don't compete for the CPU, and appends each run's solve time (JAX
compilation excluded) to scaling_runs.json. Runs already recorded there are skipped, so the
study can be stopped and resumed, or extended by adding sizes. A run that exceeds
`time_limit` is stopped and recorded as a lower bound, and that method isn't run for more
flights. Plot the results with fig_scaling.py.
"""

import os
import sys
import json
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sizes = [2, 4, 6, 8, 10, 12]
repeats = 3
time_limit = 3600.0  # s per run

# method: (script, result file); monolithic first, since ALBCD compares against its solution
methods = {"monolithic": ("monolithic.py", "monolithic_solution"), "albcd": ("airliner_fleet.py", "convergence")}

log = os.path.join(HERE, "scaling_runs.json")
runs = json.load(open(log)) if os.path.exists(log) else []

for n in sizes:
    for method, (script, result) in methods.items():
        for k in range(repeats):
            if any(r["method"] == method and r["N"] == n and r["repeat"] == k for r in runs):
                continue
            if any(r["method"] == method and r["N"] <= n and not r["finished"] for r in runs):
                continue  # timed out at this size or a smaller one
            print(f"N = {n}: {method}, run {k + 1} of {repeats}", flush=True)
            try:
                subprocess.run([sys.executable, os.path.join(HERE, script), str(n)], check=True, timeout=time_limit,
                               env=dict(os.environ, MPLBACKEND="Agg"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                saved = np.load(os.path.join(HERE, f"{result}_N{n}.npz"))
                run = dict(time=float(saved["time"]), finished=True, success=bool(saved["success"]),
                           count=int(saved["solves"] if method == "albcd" else saved["nit"]))  # block solves or SLSQP iterations
            except subprocess.TimeoutExpired:
                run = dict(time=time_limit, finished=False)
            runs.append(dict(method=method, N=n, repeat=k, **run))
            json.dump(runs, open(log, "w"), indent=1)

for method in methods:
    for n in sizes:
        times = [r["time"] for r in runs if r["method"] == method and r["N"] == n]
        if times:
            finished = all(r["finished"] for r in runs if r["method"] == method and r["N"] == n)
            print(f"{method:10s} N = {n:2d}: median {np.median(times):7.1f} s over {len(times)} runs"
                  + ("" if finished else f" (stopped at {time_limit:.0f} s)"))
