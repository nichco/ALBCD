"""ALBCD, ALC and ADMM (a small and a large penalty) on the Cessna problem of examples/jax/cessna, with
tight tolerances and the same budget of BCD sweeps for every method, so that it is clear which
methods converge and which stall.

    python cessna_comparison_tight.py [RUN ...] [--sweeps N] [--out FILE]

RUN is albcd, alc, admm_small, admm_large or admm=MU (any fixed penalty); no RUN runs the first four.

Compared with cessna_comparison.py (the example's settings):

- the blocks' SLSQP tolerance is 1e-13 instead of 1e-8, so a block solve does not stop before the
  block has moved (at 1e-8 the blocks freeze at small penalties, whatever the method)
- every method runs exactly N BCD sweeps (300 by default; one sweep = one solve of each block:
  aero, then structures). Their stopping tests are switched off, and each run records the sweep at
  which its own test (feasibility 1e-5, optimality 3e-5 for ALBCD and ADMM; ALC's Eqs. 26-27 with
  epsilon = 1e-5) would have stopped it. Past that sweep every method keeps running its own
  iteration unchanged: ALBCD more outer iterations of its final phase, ALC more outer iterations
  with its Eq. 25 weight update, ADMM more sweeps with its fixed penalty
- ALBCD starts from mu0 = 0.1, ALC from the initial weight of its Eq. 29, and ADMM uses a fixed
  mu = 0.1 (admm_small) or 10 (admm_large)

Past its stopping point ALC's weights grow by beta^2 = 4.84 at every outer iteration (Eq. 25 has no
feasibility floor, and constraints at round-off cannot shrink by gamma), until the block solves fail
and the iterates diverge (around sweep 150, with weights near 1e15). Each of those block solves runs
to SLSQP's 1000-iteration limit, so the ALC run takes hours; the others take minutes.
- the error is measured against a monolithic solution solved here with SLSQP ftol 1e-14 (the
  example's monolithic_solution.npz is solved with ftol 1e-8 and is about 7e-5 from it)

Every history has one entry per sweep. The methods differ in how many sweeps they run between
multiplier updates: ALBCD and ALC run an inner loop of one or more sweeps, ADMM updates after every
sweep. Each run records the sweeps after which its multipliers were updated, and the cumulative
SLSQP iterations after every sweep (a cost that does not depend on how work is split into sweeps).

Writes cessna_comparison_tight.npz (keys <run>_<field>, plus the reference ref_x and ref_CD), which
fig_cessna_comparison_tight.py plots. A run of some methods keeps the others' results in the file,
so the runs can go in separate processes.
"""

import os
import sys
import time
import argparse

# single-threaded BLAS and XLA, so that the runs are reproducible
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "jax", "cessna"))

import jax.numpy as jnp  # noqa: E402
import modopt as mo  # noqa: E402
import cessna as C  # noqa: E402  (the example; its __main__ block does not run on import)
from albcd import ALBCD, ALC, ADMM, initial_penalty  # noqa: E402

FTOL = 1e-13      # blocks' SLSQP tolerance
FEAS_TOL = 1e-5   # feasibility tolerance (ALC's epsilon, ADMM's primal tolerance)
OPT_TOL = 3e-5    # optimality tolerance: ALBCD's final phase, ADMM's dual tolerance
MU_ALBCD = 0.1    # ALBCD's initial penalty
MU_ADMM = {"admm_small": 0.1, "admm_large": 10.0}  # ADMM's fixed penalties

parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="*", default=["albcd", "alc", "admm_small", "admm_large"])
parser.add_argument("--sweeps", type=int, default=300, help="BCD sweeps per run")
parser.add_argument("--out", default=os.path.join(HERE, "cessna_comparison_tight.npz"))
args = parser.parse_args()
N_SWEEPS, OUT = args.sweeps, args.out


# --- results file, shared by runs in separate processes ---

def update_results(new):
    """Merge new entries into OUT under a lock file, keeping what other runs wrote."""
    lock = OUT + ".lock"
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL)
            break
        except FileExistsError:
            time.sleep(0.5)
    try:
        saved = dict(np.load(OUT)) if os.path.exists(OUT) else {}
        saved.update(new)
        np.savez(OUT, **saved)
        return saved
    finally:
        os.close(fd)
        os.remove(lock)


# --- tight monolithic reference: the problem of examples/jax/cessna/monolithic.py with ftol 1e-14 ---

def reference():
    nt, nh = C.num_cp_twist, C.num_cp_thickness

    def objective(z):
        return C.aero_model(z[:nt])[0]

    def constraints(z):
        _, loads, lift = C.aero_model(z[:nt])
        sigma, t_min, weight = C.structures_model(loads, z[nt:])
        return jnp.stack([sigma - C.sigma_yield_mpa, lift - weight, C.min_gauge * 1e3 - t_min])

    x_scaler = np.concatenate([10 * np.ones(nt), 100 * np.ones(nh)])
    prob = mo.JaxProblem(x0=np.concatenate([C.twist_cp0, C.thickness_cp0]), jax_obj=objective,
                         jax_con=constraints, cl=np.array([-np.inf, 0.0, -np.inf]), cu=np.zeros(3),
                         xl=np.concatenate([np.full(nt, -np.inf), np.zeros(nh)]),
                         xu=np.concatenate([np.full(nt, np.inf), np.full(nh, C.thickness_max)]),
                         x_scaler=x_scaler, c_scaler=np.array([1e-2, 1e-2, 1e-1]), o_scaler=1e2)
    opt = mo.SLSQP(prob, solver_options={"maxiter": 3000, "ftol": 1e-14}, turn_off_outputs=True)
    opt.solve()
    z = opt.results["x"] / x_scaler
    print(f"reference: success {bool(opt.results['success'])}, {int(opt.results['nit'])} SLSQP iterations, "
          f"CD {float(objective(z)):.10f}")
    return z, float(objective(z))


saved = dict(np.load(OUT)) if os.path.exists(OUT) else {}
if "ref_x" in saved:
    ref = saved["ref_x"]
else:
    ref, ref_CD = reference()
    update_results({"ref_x": ref, "ref_CD": ref_CD})


def design_error(x_history):
    """Relative error in [twist_cp, thickness_cp] against the tight reference."""
    h = np.array([np.concatenate([x[C.TWIST_SLICE], x[C.THICKNESS_SLICE]]) for x in x_history])
    return np.linalg.norm(h - ref, axis=1) / np.linalg.norm(ref)


# --- sweep budget and cost: cumulative SLSQP iterations after every sweep ---

SLSQP_ITERS = [0]
_slsqp_solve = mo.SLSQP.solve


def counted_solve(self):
    result = _slsqp_solve(self)
    SLSQP_ITERS[0] += int(result.get("nit", 0) or 0)
    return result


mo.SLSQP.solve = counted_solve
nit_history = []  # SLSQP_ITERS after every sweep


class BudgetSpent(Exception):
    """Raised at the start of sweep N_SWEEPS + 1, which ends the run with exactly N_SWEEPS sweeps."""


def subproblems(budget=True):
    subs = [C.AeroSubproblem(C.AERO_INDEX), C.StructSubproblem(C.STRUCT_INDEX)]
    for sub in subs:
        sub.FTOL = FTOL
    first, last = subs[0].solve, subs[-1].solve  # aero starts every sweep, struct ends it

    def solve_first(*a, **k):
        if budget and len(nit_history) >= N_SWEEPS:
            raise BudgetSpent
        first(*a, **k)

    def solve_last(*a, **k):
        last(*a, **k)
        nit_history.append(SLSQP_ITERS[0])
    subs[0].solve, subs[-1].solve = solve_first, solve_last
    return subs


class RecordingALBCD(ALBCD):
    """ALBCD that records its multiplier updates and where its stopping test would stop it.

    It runs with the phase tolerances [1e-1, OPT_TOL, OPT_TOL, ...]: a phase ends when an outer
    iteration ends feasible, which in the final phase is where ALBCD stops. With repeated final
    phases it carries on with identical iterations instead; the stop would be the second feasible
    outer iteration (the first ends phase 1)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.updates, self.feasible_outer = [], []

    def _update_mu(self, c_new, c_old):  # called right after every multiplier update
        self.updates.append(len(self.feas_history))
        if np.max(np.abs(c_new)) <= self.feas_tol:
            self.feasible_outer.append(len(self.feas_history))
        return super()._update_mu(c_new, c_old)


class RecordingALC(ALC):
    """ALC that records where its Eqs. 26-27 would stop it; run with feas_tol = 0 so that it never stops."""

    def __init__(self, *a, eps, **k):
        super().__init__(*a, **k)
        self.eps, self.stops = eps, []

    def _update_mu(self, q_new, q_old):  # called after every outer iteration, before the weights grow
        if np.max(np.abs(q_new)) < self.eps and np.max(np.abs(q_new - q_old)) < self.eps:
            self.stops.append(len(self.feas_history))
        return super()._update_mu(q_new, q_old)


data0 = {"weight": float(C.weight_init), "aero_loads": np.array(C.aero_loads_init), "CD": float(C.CD_init)}
phi0 = np.array(C.coupling(C.aero_loads_init, C.x0[C.LOADS_SLICE], C.weight_init, C.x0[C.WEIGHT_SLICE][0]))


def objective(x, data):
    """f in ALC's merit function: the aero block's 1e2 * CD."""
    return 1e2 * data["CD"]


for run in args.runs:
    print(f"\n=== {run} ({N_SWEEPS} sweeps) ===", flush=True)
    nit_history.clear()
    SLSQP_ITERS[0] = 0
    t0 = time.perf_counter()

    if run == "albcd":
        mu0 = MU_ALBCD
        opt = RecordingALBCD(subproblems=subproblems(), x0=C.x0, mu0=np.full(C.N_CON, mu0), data0=data0,
                             max_mu=1e6, rho=1.2, tau=0.5, feas_tol=FEAS_TOL, opt_tol=[1e-1] + [OPT_TOL] * N_SWEEPS,
                             max_outer_iter=10 * N_SWEEPS, max_inner_iter=200, verbose=False)
    elif run == "alc":
        # Eq. 29: one inner loop with a small penalty estimates the coupling constraints, then the
        # penalty term is set to 10% of the initial objective. Its cost is not counted.
        mu0 = initial_penalty(subproblems(budget=False), C.x0, C.N_CON, objective, phi0=phi0, data0=data0,
                              mu_small=1e-3, inner_tol=FEAS_TOL / 100)
        nit_history.clear()
        SLSQP_ITERS[0] = 0
        opt = RecordingALC(subproblems(), C.x0, np.full(C.N_CON, mu0), objective, phi0=phi0, data0=data0,
                           eps=FEAS_TOL, feas_tol=0.0, inner_tol=FEAS_TOL / 100, max_outer_iter=10 * N_SWEEPS,
                           max_inner_iter=200, verbose=False)
    elif run.startswith("admm"):
        mu0 = float(run.split("=")[1]) if "=" in run else MU_ADMM[run]
        opt = ADMM(subproblems(), C.x0, mu0, n_con=C.N_CON, data0=data0, feas_tol=0.0, opt_tol=0.0,
                   max_iter=N_SWEEPS, verbose=False)
    else:
        raise SystemExit(f"unknown run {run!r}: use albcd, alc, admm_small, admm_large or admm=MU")

    try:
        opt.solve()
    except BudgetSpent:
        pass

    feas, res = np.array(opt.feas_history), np.array(opt.opt_history)
    if run == "albcd":
        updates = opt.updates
        stop = opt.feasible_outer[1] if len(opt.feasible_outer) > 1 else 0
    elif run == "alc":
        updates = list(np.flatnonzero(np.diff(opt.outer_history)) + 1)
        stop = opt.stops[0] if opt.stops else 0
    else:
        updates = list(range(1, len(feas)))  # after every sweep but the last
        met = np.flatnonzero((feas <= FEAS_TOL) & (res <= OPT_TOL))
        stop = int(met[0]) + 1 if met.size else 0

    r = dict(opt_history=res, feas_history=feas,
             error=design_error(opt.x_history[2::2]),  # x after sweep k is x_history[2k]
             updates=np.array(updates, dtype=int), slsqp_iters=np.array(nit_history),
             stop=stop,  # sweep at which the method's own stopping test is met (0: never)
             mu0=mu0, max_mu=float(np.max(opt.mu)), x=opt.x, time=time.perf_counter() - t0, sweeps=len(feas))
    at = (f"its stopping test is met at sweep {stop} (optimality residual {res[stop - 1]:.2e}, "
          f"error {r['error'][stop - 1]:.2e})" if stop else "its stopping test is never met")
    print(f"{run}: {len(feas)} sweeps, {len(updates)} multiplier updates, {nit_history[-1]} SLSQP iterations, "
          f"final optimality residual {res[-1]:.2e}, feasibility {feas[-1]:.2e}, relative error "
          f"{r['error'][-1]:.2e}; {at}; mu0 {mu0:.3g}, max mu {r['max_mu']:.3g}, {r['time']:.0f} s", flush=True)
    update_results({f"{run}_{key}": value for key, value in r.items()})
