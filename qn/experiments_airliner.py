"""Task 6 experiments on the airliner fleet: Task 6's configurations at N = 4, and albcd, v2, v3 at N = 8.

Usage: ``python qn/experiments_airliner.py [N ...]`` (default: 4 8). Saves qn/results/qn_results_airliner.npz
(keys "N{N}/<configuration>__<field>") after every configuration, and writes the summary tables to
qn/results/qn_summary_airliner.txt. Distances are in the scaled variables (x_scaler), against
qn/results/monolithic_solution_tight_airliner_N{N}.npz (Task 5).
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import airliner_blocks as ab  # noqa: E402
import qn_core as qc  # noqa: E402
from experiments_common import run_config, configurations, summary_table, flatten  # noqa: E402

RESULTS = HERE / "results"
N8_CONFIGS = ("albcd", "v2", "v3")


def reference(N):
    return np.load(RESULTS / f"monolithic_solution_tight_airliner_N{N}.npz")


def reference_theta(N, ref):
    """theta (§1.6) of the tight monolithic solution, with its SLSQP multipliers mapped to (lam, kappa_i)."""
    opt = ab.make_solver(N, verbose=False)
    z, m = ref["z"], ref["multipliers"]
    cl, cu = ab.cl, ab.cu
    ncon = len(cl)
    big_cl = np.append(np.tile(cl, N), 0.0)
    big_cu = np.append(np.tile(cu, N), 0.0)
    eq = big_cl == big_cu
    lower = np.isfinite(big_cl) & ~eq
    upper = np.isfinite(big_cu) & ~eq
    n_eq, n_lo = int(eq.sum()), int(lower.sum())
    kappa_all = np.zeros(len(big_cl))
    kappa_all[eq] = -m[:n_eq]
    kappa_all[lower] += -m[n_eq:n_eq + n_lo]
    kappa_all[upper] += m[n_eq + n_lo:]
    lam = np.array([kappa_all[-1]])
    lins, kappas, frees = [], [], []
    phi = np.zeros(1)
    for i, sp in enumerate(opt.subproblems):
        lin = dict(sp.grad_phi(z, {}))
        lin["v"] = z[sp.index]
        k = kappa_all[i * ncon:(i + 1) * ncon]
        r = qc.stationarity(lin, lam, k)
        free, _, _ = qc.active_sets(lin, k, r)
        lins.append(lin)
        kappas.append(k)
        frees.append(free)
        phi = phi + lin["c"]
    return qc.kkt_measure(lins, lam, kappas, frees, phi, parts=True)


def main(Ns):
    all_results, tables = {}, []
    out_file = RESULTS / "qn_results_airliner.npz"
    for N in Ns:
        ref = reference(N)
        scale = np.tile(ab.x_scaler, N)
        z_star = ref["z"]
        dist = lambda x: float(np.linalg.norm((np.asarray(x) - z_star) * scale))
        th_ref = reference_theta(N, ref)
        print(f"N = {N}: tight reference theta {th_ref[0]:.2e} (stationarity {th_ref[1]:.2e}, feasibility "
              f"{th_ref[2]:.2e}, local {th_ref[3]:.2e})", flush=True)

        names = None if N == 4 else N8_CONFIGS
        results = []
        time_limit = np.inf
        for name, cls, settings in configurations(np.array([0.01]), names):
            build = (lambda cls=cls, settings=settings: ab.make_solver(N, cls=cls, **settings))
            r = run_config(name, build, ab.compile_models, dist, time_limit,
                           log=lambda s: print(f"N={N} {s}", flush=True))
            if name == "albcd":
                time_limit = 3 * r["total_time"]
            results.append(r)
            all_results[N] = (results, th_ref)
            arrays = {}
            for n_, (res, th) in all_results.items():
                arrays.update(flatten(res, prefix=f"N{n_}/"))
                arrays[f"N{n_}/reference_theta"] = np.asarray(th)  # theta, stationarity, feasibility, local
            np.savez(out_file, **arrays)
        tables.append(summary_table(results, f"Airliner fleet, N = {N} (reference theta {th_ref[0]:.2e}, "
                                             f"albcd time limit for the others: {time_limit:.0f} s)"))
        print(tables[-1], flush=True)
    (RESULTS / "qn_summary_airliner.txt").write_text("\n\n".join(tables) + "\n")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [4, 8])
