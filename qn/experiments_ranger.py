"""Task 6 experiments on ranger: every configuration of Task 6's table.

Usage: ``python qn/experiments_ranger.py [configuration ...]`` (default: all). Saves
qn/results/qn_results_ranger.npz after every configuration and writes the summary table to
qn/results/qn_summary_ranger.txt. Distances use only the twist and thickness entries, scaled by the blocks'
X_SCALER (10 and 100), against qn/results/monolithic_solution_tight_ranger.npz (Task 5).
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import ranger_blocks as rb  # noqa: E402
from experiments_common import run_config, configurations, summary_table, save_results  # noqa: E402

RESULTS = HERE / "results"


def main(names=None):
    ref = np.load(RESULTS / "monolithic_solution_tight_ranger.npz")
    x_star = np.concatenate([ref["twist_cp"], ref["thickness_cp"]])
    scale = np.concatenate([np.full(rb.num_cp_twist, 10.0), np.full(rb.num_cp_thickness, 100.0)])

    def dist(x):
        x = np.asarray(x)
        return float(np.linalg.norm((np.concatenate([x[rb.TWIST], x[rb.THICKNESS]]) - x_star) * scale))

    print(f"tight reference: KKT stationarity {float(ref['stationarity']):.2e}, feasibility "
          f"{float(ref['feasibility']):.2e}", flush=True)
    results = []
    time_limit = np.inf
    mu0 = np.full(2 * rb.num_nodes + 1, 10.0)
    for name, cls, settings in configurations(mu0, names):
        build = (lambda cls=cls, settings=settings: rb.make_solver(cls=cls, **settings))
        r = run_config(name, build, rb.compile_models, dist, time_limit, log=lambda s: print(s, flush=True))
        if name == "albcd":
            time_limit = 3 * r["total_time"]
        results.append(r)
        save_results(results, RESULTS / "qn_results_ranger.npz",
                     reference_stationarity=float(ref["stationarity"]), reference_feasibility=float(ref["feasibility"]))
    table = summary_table(results, f"Ranger (tight reference KKT stationarity {float(ref['stationarity']):.2e}; "
                                   f"albcd time limit for the others: {time_limit:.0f} s)")
    print(table, flush=True)
    (RESULTS / "qn_summary_ranger.txt").write_text(table + "\n")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
