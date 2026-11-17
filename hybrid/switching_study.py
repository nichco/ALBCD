"""Compare switching criteria for the hybrid ALBCD solver.

Phase 1 does not depend on the criterion until it fires, so ALBCD is run once, to its own tolerances,
keeping its state after every sweep. Each criterion is then replayed on that history to find the sweep at
which it would have switched, and phase 2 (OpenSQP on the full problem) is run from the states at those
sweeps, at a regular grid of sweeps, from the initial point (sweep 0, the monolithic solve) and from the
final ALBCD iterate. The cost of a criterion is the ALBCD time up to its switch plus the OpenSQP time
from there, to the tolerances in the example's SQP_OPTIONS.

    python switching_study.py ranger
    python switching_study.py fleet 4        (the number of flights)
    python switching_study.py fleet 8 hessian   (phase 2 with the Hessian warm start; tag fleet_N8_hessian)

Optional third argument: the spacing of the grid of sweeps (default 3). Writes
switching_study_<problem>.npz and .txt, which fig_switching_study.py plots.
"""

import os

# single-threaded BLAS and XLA, as in the timing scripts of the examples, so that the two phases run with
# the same resources and the timings are less sensitive to other jobs. Must be set before numpy loads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import sys
import time
from types import SimpleNamespace
import numpy as np

import hybrid_albcd as H

HERE = os.path.dirname(os.path.abspath(__file__))
HESSIAN = "hessian" in sys.argv  # phase 2 with the Hessian warm start (the full model's warm_hessian)
sys.argv = [a for a in sys.argv if a != "hessian"]
name = sys.argv[1] if len(sys.argv) > 1 else "ranger"

if name == "ranger":
    import ranger as problem
    tag = "ranger"
    make_solver = problem.make_solver
    x_scale = np.concatenate([problem.AeroSubproblem.X_SCALER, problem.StructSubproblem.X_SCALER])
else:
    import airliner_fleet as problem
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tag = f"fleet_N{N}"
    make_solver = lambda **kwargs: problem.make_solver(N, **kwargs)
    x_scale = np.tile(problem.x_scaler, N)
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 3
if HESSIAN:
    tag += "_hessian"

# a throwaway solve (one sweep, two OpenSQP iterations), so that no lazy import or compilation is timed
scratch = make_solver(warm_hessian=True, verbose=False, max_outer_iter=1, max_inner_iter=1,
                      sqp_options={**problem.SQP_OPTIONS, "maxiter": 2})
problem.compile_models(scratch)
scratch.solve()

opt = make_solver(warm_hessian=HESSIAN)
problem.compile_models(opt)
x0, y0, data0 = opt.x.copy(), opt.y.copy(), dict(opt.data)

# ---------------------------------------------------------------- phase 1, once, to ALBCD's own tolerances
opt._t0 = time.perf_counter()
opt._albcd()
h = opt.history
K = len(h["optimality"])
albcd_success = opt.success
print(f"ALBCD: {K} sweeps, {h['time'][-1]:.1f} s, success {albcd_success}, optimality {h['optimality'][-1]:.2e}, "
      f"feasibility {h['feasibility'][-1]:.2e}", flush=True)

# ---------------------------------------------------------------- the criteria, replayed on the history
criteria = {
    "after_sweeps(1)": H.after_sweeps(1),
    "after_sweeps(2)": H.after_sweeps(2),
    "after_sweeps(5)": H.after_sweeps(5),
    "after_sweeps(10)": H.after_sweeps(10),
    "feasibility_below(100)": H.feasibility_below(100.0),
    "feasibility_below(10)": H.feasibility_below(10.0),
    "feasibility_below(1)": H.feasibility_below(1.0),
    "merit_below(1000)": H.merit_below(1000.0),
    "merit_below(100)": H.merit_below(100.0),
    "merit_below(10)": H.merit_below(10.0),
    "stagnation(0.5)": H.stagnation(0.5),
    "stagnation(0.7)": H.stagnation(0.7),
    "stagnation(0.9)": H.stagnation(0.9),
    "multipliers_settled(0.5)": H.multipliers_settled(0.5),
    "multipliers_settled(0.1)": H.multipliers_settled(0.1),
    "multipliers_settled(0.01)": H.multipliers_settled(0.01),
    "step_below(1e-2)": H.step_below(1e-2, x_scale),
    "step_below(1e-3)": H.step_below(1e-3, x_scale),
    "step_below(1e-4)": H.step_below(1e-4, x_scale),
}


def switch_sweep(criterion):
    """The first sweep at which the criterion fires; K (the end of ALBCD) if it never does."""
    for k in range(1, K + 1):
        view = SimpleNamespace(history={key: h[key][:k] for key in ("optimality", "feasibility", "time", "x_sweep",
                                                                     "y", "mu", "outer")},
                               opt_tol=opt.opt_tol, feas_tol=opt.feas_tol)
        if criterion(view):
            return k
    return K


switch = {label: switch_sweep(c) for label, c in criteria.items()}
switch["monolithic (no ALBCD)"] = 0
switch["ALBCD to its tolerances"] = K
sweeps = sorted(set(switch.values()) | set(range(0, K + 1, stride)))
print("switch sweeps:", switch, flush=True)

# ---------------------------------------------------------------- phase 2 from every candidate sweep
opt_tol, feas_tol = opt.sqp_options["opt_tol"], opt.sqp_options["feas_tol"]
runs = {}
for k in sweeps:
    x, y, data = (x0, y0, data0) if k == 0 else (h["x_sweep"][k - 1], h["y"][k - 1], h["data"][k - 1])
    opt.x, opt.y, opt.data = x.copy(), y.copy(), dict(data)
    t = time.perf_counter()
    opt._monolithic()
    t_sqp = time.perf_counter() - t
    r = opt.sqp.results
    hist = np.array([(row[0], row[2], row[3], row[4]) for row in opt.sqp.history])  # time, opt, feas, objective
    runs[k] = dict(t_albcd=0.0 if k == 0 else h["time"][k - 1], t_sqp=t_sqp, niter=int(r["niter"]),
                   nfev=int(r["nfev"]), success=bool(r["success"]), objective=float(r["objective"]),
                   opt=float(r["optimality"]), feas=float(r["feasibility"]), kkt0=hist[0, 1], history=hist,
                   z=opt.z.copy())
    print(f"switch at sweep {k:3d}: ALBCD {runs[k]['t_albcd']:7.1f} s + OpenSQP {t_sqp:7.1f} s ({r['niter']:4d} its) "
          f"= {runs[k]['t_albcd'] + t_sqp:7.1f} s   success {r['success']}   objective {r['objective']:.10f}", flush=True)

# ---------------------------------------------------------------- results
total = {k: run["t_albcd"] + run["t_sqp"] for k, run in runs.items()}
best = min((k for k in runs if runs[k]["success"]), key=total.get)
lines = [f"{tag}: ALBCD alone {K} sweeps, {h['time'][-1]:.1f} s (success {albcd_success}); OpenSQP to opt_tol "
         f"{opt_tol:.0e}, feas_tol {feas_tol:.0e}",
         f"best switch (of the sweeps tried): sweep {best}, {total[best]:.1f} s",
         "",
         f"{'criterion':28s} {'sweep':>5s} {'ALBCD s':>8s} {'SQP its':>7s} {'SQP s':>8s} {'total s':>8s} {'/best':>6s}  ok"]
for label, k in sorted(switch.items(), key=lambda item: total[item[1]]):
    run = runs[k]
    lines.append(f"{label:28s} {k:5d} {run['t_albcd']:8.1f} {run['niter']:7d} {run['t_sqp']:8.1f} {total[k]:8.1f} "
                 f"{total[k] / total[best]:6.2f}  {run['success']}")
lines += ["", "all sweeps tried:", f"{'sweep':>5s} {'merit':>9s} {'ALBCD s':>8s} {'SQP its':>7s} {'SQP s':>8s} {'total s':>8s}  ok"]
m = H.merit(opt)
for k in sweeps:
    run = runs[k]
    lines.append(f"{k:5d} {(m[k - 1] if k else np.inf):9.2e} {run['t_albcd']:8.1f} {run['niter']:7d} {run['t_sqp']:8.1f} "
                 f"{total[k]:8.1f}  {run['success']}")
text = "\n".join(lines)
print("\n" + text)
with open(os.path.join(HERE, f"switching_study_{tag}.txt"), "w") as f:
    f.write(text + "\n")

np.savez(os.path.join(HERE, f"switching_study_{tag}.npz"),
         optimality=h["optimality"], feasibility=h["feasibility"], time=h["time"], outer=h["outer"], merit=m,
         albcd_success=albcd_success, sweeps=sweeps, labels=list(switch), switch=list(switch.values()),
         **{f"{key}": np.array([runs[k][key] for k in sweeps])
            for key in ("t_albcd", "t_sqp", "niter", "nfev", "success", "objective", "opt", "feas", "kkt0")},
         **{f"history_{k}": runs[k]["history"] for k in sweeps}, **{f"z_{k}": runs[k]["z"] for k in sweeps})
