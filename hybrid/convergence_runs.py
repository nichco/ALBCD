"""Timed runs of pure ALBCD, monolithic OpenSQP and hybrid ALBCD on one problem, for fig_convergence.py.

    python convergence_runs.py ranger
    python convergence_runs.py fleet 8        (the number of flights)

The runs start from the same point and are timed one after the other:

    albcd                phase 1 alone, to ALBCD's own tolerances
    monolithic           phase 2 alone: OpenSQP on the full problem from the initial point
    hybrid               HybridALBCD with the switching criterion SWITCH below
    monolithic_hessian   as monolithic, with OpenSQP's BFGS matrix started from the exact Hessian of the
                         objective at the initial point instead of the identity
    hybrid_hessian       HybridALBCD with the criterion SWITCH_HESSIAN and the Hessian warm start: the BFGS
                         matrix starts from the exact Hessian of the Lagrangian at the ALBCD iterate

No setup or compilation is timed. Before the timed runs, a throwaway solve (one ALBCD sweep and two OpenSQP
iterations) loads the solvers' lazy imports and compiles the JAX primitives, and every solver's jitted
functions are compiled before its run. ALBCD's clock covers its sweeps. OpenSQP's clock starts when the
full model is evaluated at its starting point; for a warm start that includes computing the Hessian.

All runs are scored with one measure, OpenSQP's optimality and feasibility of the full problem (the scaled
KKT residual it stops on). For OpenSQP's iterates these come from the solver. For the ALBCD iterates they
are evaluated after the timed run, at the state after every sweep, with the multipliers that the full
model's warm start builds from the blocks' multipliers and y (hybrid_albcd.init_multipliers).

Writes convergence_<problem>.npz, with every iterate, time and measure of the runs:

    <run>_time, <run>_optimality, <run>_feasibility    the common measure against wall time (s); iterate 0
                                                       is the initial point (for ALBCD, with zero multipliers)
    <run>_x                                            the iterates (ALBCD: x after every sweep; OpenSQP: z)
    albcd_/hybrid*_ native_optimality, native_feasibility, y, mu, outer     ALBCD's own per-sweep history
    hybrid*_switch_time, hybrid*_switch_sweep, hybrid*_switch_label         where the hybrid handed over
    hybrid*_z, monolithic*_/hybrid*_ objective         OpenSQP's iterates in a hybrid run, and its objective

and convergence_<problem>.pkl, the complete histories (including the shared data after every sweep).
"""

import os

# single-threaded BLAS and XLA, as in switching_study.py. Must be set before numpy loads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import sys
import time
import pickle
import numpy as np

import hybrid_albcd as H

HERE = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1] if len(sys.argv) > 1 else "ranger"

# the criteria that switching_study.py found best, without and with the Hessian warm start
if name == "ranger":
    import ranger as problem
    tag = "ranger"
    make_solver = problem.make_solver
    SWITCH, SWITCH_LABEL = H.merit_below(10.0), "merit_below(10)"
    SWITCH_HESSIAN, SWITCH_HESSIAN_LABEL = H.stagnation(0.5), "stagnation(0.5)"
else:
    import airliner_fleet as problem
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    tag = f"fleet_N{N}"
    make_solver = lambda **kwargs: problem.make_solver(N, **kwargs)
    SWITCH, SWITCH_LABEL = H.merit_below(10.0), "merit_below(10)"
    SWITCH_HESSIAN, SWITCH_HESSIAN_LABEL = H.stagnation(0.5), "stagnation(0.5)"


def new_solver(**kwargs):
    """A solver with its jitted functions compiled."""
    opt = make_solver(**kwargs)
    problem.compile_models(opt)
    return opt


def full_problem_kkt(opt, x, y, data):
    """OpenSQP's optimality and feasibility of the full problem at an ALBCD state (no iterations)."""
    model = plain_model(x, y, data)
    prob, pi0 = model if isinstance(model, tuple) else (model, None)
    sqp = H.RecordingOpenSQP(prob, init_multipliers=pi0, turn_off_outputs=True, **{**opt.sqp_options, "maxiter": 0})
    sqp.solve()
    return sqp.history[0][2], sqp.history[0][3]


def albcd_curve(opt, x0, y0, data0):
    """The common measure at the initial point and after every sweep of opt's phase 1."""
    h = opt.history
    states = [(x0, y0, data0)] + list(zip(h["x_sweep"], h["y"], h["data"]))
    kkt = np.array([full_problem_kkt(opt, *state) for state in states])
    return dict(time=np.concatenate([[0.0], h["time"]]), optimality=kkt[:, 0], feasibility=kkt[:, 1],
                x=np.array([s[0] for s in states]), native_optimality=np.array(h["optimality"]),
                native_feasibility=np.array(h["feasibility"]), y=np.array(h["y"]), mu=np.array(h["mu"]),
                outer=np.array(h["outer"]))


def sqp_curve(opt, t_start=0.0):
    """Wall time (from t_start), measure, objective and iterates of opt's phase 2."""
    hist = opt.sqp.history
    x_scaler = opt.sqp.problem.x_scaler
    return dict(time=t_start + np.array([row[0] for row in hist]), optimality=np.array([row[2] for row in hist]),
                feasibility=np.array([row[3] for row in hist]), objective=np.array([row[4] for row in hist]),
                x=np.array([row[1] / x_scaler for row in hist]))


def sqp_raw(opt):
    return dict(results={k: v for k, v in opt.sqp.results.items() if k != "exit_msg"}, history=opt.sqp.history)


runs, raw, success = {}, {}, {}

# ---------------------------------------------------------------- warm-up (not timed)
scratch = new_solver(warm_hessian=True, verbose=False, max_outer_iter=1, max_inner_iter=1,
                     sqp_options={**problem.SQP_OPTIONS, "maxiter": 2})
scratch.solve()
plain_model = make_solver(verbose=False).full_model  # the full model without the Hessian, for the measure

# ---------------------------------------------------------------- pure ALBCD
opt = new_solver()
x0, y0, data0 = opt.x.copy(), opt.y.copy(), dict(opt.data)
opt._t0 = time.perf_counter()
opt._albcd()
print(f"ALBCD: {len(opt.history['optimality'])} sweeps, {opt.history['time'][-1]:.1f} s, success {opt.success}", flush=True)
success["albcd"] = opt.success
runs["albcd"] = albcd_curve(opt, x0, y0, data0)
raw["albcd"] = dict(opt.history)

# ---------------------------------------------------------------- monolithic OpenSQP, identity and exact initial Hessian
for run, warm in (("monolithic", False), ("monolithic_hessian", True)):
    opt = new_solver(warm_hessian=warm)
    opt._monolithic()
    runs[run], raw[run], success[run] = sqp_curve(opt), sqp_raw(opt), opt.success
    print(f"{run}: {opt.sqp.results['niter']} iterations, {runs[run]['time'][-1]:.1f} s, success {opt.success}", flush=True)

# ---------------------------------------------------------------- hybrid, without and with the Hessian warm start
for run, warm, switch, label in (("hybrid", False, SWITCH, SWITCH_LABEL),
                                 ("hybrid_hessian", True, SWITCH_HESSIAN, SWITCH_HESSIAN_LABEL)):
    opt = new_solver(switch=switch, warm_hessian=warm)
    opt.solve()
    problem.report(opt)
    phase1, phase2 = albcd_curve(opt, x0, y0, data0), sqp_curve(opt, opt.t_albcd)
    runs[run] = {key: np.concatenate([phase1[key], phase2[key]]) for key in ("time", "optimality", "feasibility")}
    runs[run].update(x=phase1["x"], z=phase2["x"], objective=phase2["objective"],
                     **{key: phase1[key] for key in ("native_optimality", "native_feasibility", "y", "mu", "outer")},
                     switch_time=opt.t_albcd, switch_sweep=len(opt.history["optimality"]), switch_label=label)
    raw[run] = dict(albcd=dict(opt.history), **sqp_raw(opt))
    success[run] = opt.success

np.savez(os.path.join(HERE, f"convergence_{tag}.npz"),
         albcd_opt_tol=opt.opt_tol, albcd_feas_tol=opt.feas_tol,
         sqp_opt_tol=opt.sqp_options["opt_tol"], sqp_feas_tol=opt.sqp_options["feas_tol"],
         **{f"{run}_success": value for run, value in success.items()},
         **{f"{run}_{key}": value for run, curve in runs.items() for key, value in curve.items()})
with open(os.path.join(HERE, f"convergence_{tag}.pkl"), "wb") as f:
    pickle.dump(raw, f)

for run, curve in runs.items():
    print(f"{run:19s} {curve['time'][-1]:8.1f} s   optimality {curve['optimality'][-1]:.2e}   "
          f"feasibility {curve['feasibility'][-1]:.2e}   success {success[run]}")
