"""Does a Hessian warm start shorten phase 2?

Phase 2 (OpenSQP on the full problem) is run from ALBCD states saved by convergence_runs.py, once with
OpenSQP's BFGS matrix started from the identity and once from the exact Hessian of the full problem's
Lagrangian at the ALBCD iterate (the full model's ``warm_hessian`` option), made positive definite.

    python hessian_warm_start.py ranger
    python hessian_warm_start.py fleet 8

Reads convergence_<problem>.pkl and writes hessian_warm_start_<problem>.npz and .txt. The states are the
merit_below(10) switch, the end of ALBCD, and two earlier sweeps. (The initial point is not one of them:
without block solves there are no multipliers to build the Lagrangian with.)
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import sys
import time
import pickle
from types import SimpleNamespace
import numpy as np
import hybrid_albcd as H

HERE = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1] if len(sys.argv) > 1 else "ranger"
if name == "ranger":
    import ranger as problem
    tag = "ranger"
    make_solver = problem.make_solver
else:
    import airliner_fleet as problem
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    tag = f"fleet_N{N}"
    make_solver = lambda **kwargs: problem.make_solver(N, **kwargs)

with open(os.path.join(HERE, f"convergence_{tag}.pkl"), "rb") as f:
    h = pickle.load(f)["albcd"]
K = len(h["optimality"])

solvers = {False: make_solver(verbose=False), True: make_solver(warm_hessian=True, verbose=False)}
for opt in solvers.values():
    problem.compile_models(opt)

# the sweep at which merit_below(10) switches
view = lambda k: SimpleNamespace(history={key: h[key][:k] for key in ("optimality", "feasibility")},
                                 opt_tol=solvers[False].opt_tol, feas_tol=solvers[False].feas_tol)
k_switch = next((k for k in range(1, K + 1) if H.merit_below(10.0)(view(k))), K)
sweeps = sorted({max(1, k_switch // 4), max(1, k_switch // 2), k_switch, K})

lines = [f"{tag}: phase 2 from ALBCD sweep k (merit_below(10) switches at sweep {k_switch}; ALBCD ends at {K})",
         f"{'sweep':>5s} {'ALBCD s':>8s} | {'identity: its':>13s} {'s':>7s} | {'Hessian: its':>12s} {'s':>7s} {'setup s':>8s}  ok"]
results = {}
for k in sweeps:
    row = {}
    for warm, opt in solvers.items():
        opt.x, opt.y, opt.data = h["x_sweep"][k - 1].copy(), h["y"][k - 1].copy(), dict(h["data"][k - 1])
        t = time.perf_counter()
        opt._monolithic()
        total = time.perf_counter() - t
        r = opt.sqp.results
        hist = np.array([(a[0], a[2], a[3], a[4]) for a in opt.sqp.history])
        # total includes building the problem, multipliers and Hessian; r["time"] is OpenSQP's own time
        row[warm] = dict(niter=int(r["niter"]), time=total, setup=total - r["time"], success=bool(r["success"]),
                         objective=float(r["objective"]), history=hist)
        print(f"sweep {k} warm_hessian {warm}: {r['niter']} its, {total:.1f} s, success {r['success']}, "
              f"objective {r['objective']:.10f}", flush=True)
    results[k] = row
    a, b = row[False], row[True]
    lines.append(f"{k:5d} {h['time'][k - 1]:8.1f} | {a['niter']:13d} {a['time']:7.1f} | {b['niter']:12d} {b['time']:7.1f} "
                 f"{b['setup']:8.1f}  {a['success']}/{b['success']}")
text = "\n".join(lines)
print("\n" + text)
with open(os.path.join(HERE, f"hessian_warm_start_{tag}.txt"), "w") as f:
    f.write(text + "\n")
np.savez(os.path.join(HERE, f"hessian_warm_start_{tag}.npz"), sweeps=sweeps, t_albcd=[h["time"][k - 1] for k in sweeps],
         **{f"{key}_{'hessian' if warm else 'identity'}": np.array([results[k][warm][key] for k in sweeps])
            for warm in (False, True) for key in ("niter", "time", "setup", "success", "objective")},
         **{f"history_{'hessian' if warm else 'identity'}_{k}": results[k][warm]["history"]
            for k in sweeps for warm in (False, True)})
