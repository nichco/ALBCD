"""Monolithic OpenSQP with an exact Hessian injected partway through: does it match the hybrid?

The monolithic solve of convergence_runs.py is repeated with the BFGS matrix replaced, once, at major
iteration k, by the exact Hessian of the Lagrangian at OpenSQP's own iterate and multipliers, made positive
definite (RecordingOpenSQP.inject). That is the information the hybrid's warm start gives OpenSQP, without
ALBCD. Every k in the sweep is run, so the best k is chosen in hindsight, which favors the monolithic solve.
A periodic variant injects every PERIOD iterations (off: PERIODIC = False).

    python hessian_injection.py ranger
    python hessian_injection.py fleet 8

No setup or compilation is timed (a throwaway solve with an injection runs first). Reads distance_<problem>.npz
for the reference solution, and writes injection_<problem>.npz and .txt:

    k                      the injection iterations tried (-PERIOD for the periodic variant)
    time_<k>, optimality_<k>, feasibility_<k>, distance_<k>, x_<k>    the run's history (iterate 0 = x0)
    niter, total_time, success                                        one entry per k
    best                   the k with the smallest total time among the successful runs
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import sys
import time
import numpy as np
import hybrid_albcd as H

HERE = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1] if len(sys.argv) > 1 else "ranger"
PERIOD = 10
PERIODIC = False  # also run the periodic variant (injection every PERIOD iterations)

if name == "ranger":
    import ranger as problem
    tag = "ranger"
    make_solver = problem.make_solver
    scale = problem.Z_SCALER
    KS = [1, 2, 3, 4, 6, 8, 10, 15, 20, 25]
else:
    import airliner_fleet as problem
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    tag = f"fleet_N{N}"
    make_solver = lambda **kwargs: problem.make_solver(N, **kwargs)
    scale = np.tile(problem.x_scaler, N)
    KS = [1, 2, 3, 5, 10, 20, 30, 40, 60, 80]

z_ref = np.load(os.path.join(HERE, f"distance_{tag}.npz"))["z_ref"]


def run(opt, inject):
    """The monolithic solve from the initial point, with the given injection rule."""
    model = opt.full_model(opt.x, opt.y, opt.data)
    problem_ = model[0] if isinstance(model, tuple) else model
    t = time.perf_counter()
    sqp = H.RecordingOpenSQP(problem_, turn_off_outputs=True, **opt.sqp_options)
    sqp.inject, sqp.hessian = inject, opt.full_model.hessian
    sqp.t_model = time.perf_counter() - t
    sqp.solve()
    return sqp


# warm-up (not timed): compile the models and the Hessian, and load OpenSQP's lazy imports
opt = make_solver(verbose=False, sqp_options={**problem.SQP_OPTIONS, "maxiter": 3})
problem.compile_models(opt)
run(opt, lambda k: k == 1)

opt = make_solver(verbose=False)
problem.compile_models(opt)
out, rows = {}, []
for k in KS + ([-PERIOD] if PERIODIC else []):
    rule = (lambda j, k=k: j == k) if k > 0 else (lambda j: j > 0 and j % PERIOD == 0)
    sqp = run(opt, rule)
    hist = sqp.history
    t = np.array([h[0] for h in hist])
    z = np.array([h[1] for h in hist]) / sqp.problem.x_scaler
    r = sqp.results
    out.update({f"time_{k}": t, f"optimality_{k}": np.array([h[2] for h in hist]),
                f"feasibility_{k}": np.array([h[3] for h in hist]), f"x_{k}": z,
                f"distance_{k}": np.linalg.norm((z - z_ref) * scale, axis=1), f"injected_{k}": np.array(sqp.injected)})
    rows.append((k, int(r["niter"]), t[-1], bool(r["success"])))
    label = f"every {PERIOD}" if k < 0 else f"at {k}"
    print(f"inject {label:9s}: {r['niter']:4d} iterations, {t[-1]:7.1f} s, success {r['success']}, "
          f"injected at {sqp.injected[:6]}{'...' if len(sqp.injected) > 6 else ''}", flush=True)

ok = [row for row in rows if row[3]]
best = min(ok, key=lambda row: row[2])[0]
c = np.load(os.path.join(HERE, f"convergence_{tag}.npz"))
lines = [f"{tag}: monolithic OpenSQP with the exact Hessian injected at iteration k (best: k = {best})",
         f"  no injection: {len(c['monolithic_time']) - 1} its, {c['monolithic_time'][-1]:.1f} s; hybrid with Hessian "
         f"warm start: {c['hybrid_hessian_time'][-1]:.1f} s (from convergence_runs.py, an earlier run)",
         f"{'k':>9s} {'its':>5s} {'time s':>8s}  ok"]
lines += [f"{(f'every {PERIOD}' if k < 0 else str(k)):>9s} {n:5d} {t:8.1f}  {s}" for k, n, t, s in rows]
text = "\n".join(lines)
print("\n" + text)
with open(os.path.join(HERE, f"injection_{tag}.txt"), "w") as f:
    f.write(text + "\n")
np.savez(os.path.join(HERE, f"injection_{tag}.npz"), k=[row[0] for row in rows], niter=[row[1] for row in rows],
         total_time=[row[2] for row in rows], success=[row[3] for row in rows], best=best, period=PERIOD, **out)
