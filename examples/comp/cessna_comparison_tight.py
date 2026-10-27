"""ALBCD, ALC and ADMM on the Cessna problem of examples/jax/cessna, from small penalties and with
tight tolerances, so that it is clear which methods converge and which stall.

    python cessna_comparison_tight.py [albcd] [alc] [admm]      (no argument: all three)

Compared with cessna_comparison.py (the example's settings):

- the blocks' SLSQP tolerance is 1e-13 instead of 1e-8, so a block solve does not stop before the
  block has moved (at 1e-8 the blocks freeze at small penalties, whatever the method)
- the stopping tolerances are feasibility 1e-5 and optimality 3e-5
- all three methods start from small penalties: ALBCD mu0 = 0.1, ADMM a fixed mu = 0.1, ALC the
  initial weight of its Eq. 29
- the error is measured against a monolithic solution solved here with SLSQP ftol 1e-14 (the
  example's monolithic_solution.npz is solved with ftol 1e-8 and is about 7e-5 from it, more than
  ALBCD's final error)

Every history has one entry per BCD sweep (one solve of each block: aero, then structures). The
methods differ in how many sweeps they run between multiplier updates: ALBCD and ALC run an inner
loop of one or more sweeps, ADMM updates after every sweep. Each run records the sweeps after which
its multipliers were updated, and the cumulative SLSQP iterations after every sweep (a cost that
does not depend on how the work is split into sweeps).

Writes cessna_comparison_tight.npz (keys <method>_<field>, plus the reference ref_x and ref_CD),
which fig_cessna_comparison_tight.py plots. A run of some methods keeps the others' results in the
file. ADMM runs to its 2000-sweep limit, which takes much longer than the other two.
"""

import os
import sys
import time

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
FEAS_TOL = 1e-5   # feasibility tolerance of all three methods (ALC's epsilon, ADMM's primal tolerance)
OPT_TOL = 3e-5    # optimality tolerance: ALBCD's final phase, ADMM's dual tolerance
MU = 0.1          # ALBCD's initial penalty and ADMM's fixed penalty
OUT = os.path.join(HERE, "cessna_comparison_tight.npz")

methods = sys.argv[1:] or ["albcd", "alc", "admm"]
saved = dict(np.load(OUT)) if os.path.exists(OUT) else {}


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


if "ref_x" not in saved:
    saved["ref_x"], saved["ref_CD"] = reference()
ref = saved["ref_x"]


def design_error(x_history):
    """Relative error in [twist_cp, thickness_cp] against the tight reference."""
    h = np.array([np.concatenate([x[C.TWIST_SLICE], x[C.THICKNESS_SLICE]]) for x in x_history])
    return np.linalg.norm(h - ref, axis=1) / np.linalg.norm(ref)


# --- cost: cumulative SLSQP iterations after every sweep ---

SLSQP_ITERS = [0]
_slsqp_solve = mo.SLSQP.solve


def counted_solve(self):
    result = _slsqp_solve(self)
    SLSQP_ITERS[0] += int(result.get("nit", 0) or 0)
    return result


mo.SLSQP.solve = counted_solve
nit_history = []  # SLSQP_ITERS after every sweep


def subproblems():
    subs = [C.AeroSubproblem(C.AERO_INDEX), C.StructSubproblem(C.STRUCT_INDEX)]
    for sub in subs:
        sub.FTOL = FTOL
    last = subs[-1].solve  # the struct block ends every sweep

    def solve(*args, **kwargs):
        last(*args, **kwargs)
        nit_history.append(SLSQP_ITERS[0])
    subs[-1].solve = solve
    return subs


class RecordingALBCD(ALBCD):
    """ALBCD that records the sweeps after which it updates its multipliers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.updates = []

    def _update_mu(self, c_new, c_old):  # called right after every multiplier update
        self.updates.append(len(self.feas_history))
        return super()._update_mu(c_new, c_old)


data0 = {"weight": float(C.weight_init), "aero_loads": np.array(C.aero_loads_init), "CD": float(C.CD_init)}
phi0 = np.array(C.coupling(C.aero_loads_init, C.x0[C.LOADS_SLICE], C.weight_init, C.x0[C.WEIGHT_SLICE][0]))


def objective(x, data):
    """f in ALC's merit function: the aero block's 1e2 * CD."""
    return 1e2 * data["CD"]


for name in methods:
    print(f"\n=== {name.upper()} ===")
    nit_history.clear()
    SLSQP_ITERS[0] = 0
    t0 = time.perf_counter()

    if name == "albcd":
        mu0 = MU
        opt = RecordingALBCD(subproblems=subproblems(), x0=C.x0, mu0=np.full(C.N_CON, mu0), data0=data0,
                             max_mu=1e6, rho=1.2, tau=0.5, feas_tol=FEAS_TOL, opt_tol=[1e-1, OPT_TOL],
                             max_outer_iter=80, max_inner_iter=200, verbose=False)
        opt.solve()
        updates = opt.updates

    elif name == "alc":
        # Eq. 29: one inner loop with a small penalty estimates the coupling constraints, then the
        # penalty term is set to 10% of the initial objective. Its cost is not counted.
        mu0 = initial_penalty(subproblems(), C.x0, C.N_CON, objective, phi0=phi0, data0=data0,
                              mu_small=1e-3, inner_tol=FEAS_TOL / 100)
        nit_history.clear()
        SLSQP_ITERS[0] = 0
        opt = ALC(subproblems(), C.x0, np.full(C.N_CON, mu0), objective, phi0=phi0, data0=data0,
                  feas_tol=FEAS_TOL, max_outer_iter=100, max_inner_iter=200, verbose=False)
        opt.solve()
        # outer_history holds the outer iteration of every sweep: an update follows each change
        updates = list(np.flatnonzero(np.diff(opt.outer_history)) + 1)

    elif name == "admm":
        mu0 = MU
        opt = ADMM(subproblems(), C.x0, mu0, n_con=C.N_CON, data0=data0, feas_tol=FEAS_TOL, opt_tol=OPT_TOL,
                   max_iter=2000, verbose=False)
        opt.solve()
        updates = list(range(1, len(opt.feas_history)))  # after every sweep but the last

    else:
        raise SystemExit(f"unknown method {name!r}: use albcd, alc or admm")

    r = dict(opt_history=np.array(opt.opt_history), feas_history=np.array(opt.feas_history),
             error=design_error(opt.x_history[2::2]),  # x after sweep k is x_history[2k]
             updates=np.array(updates, dtype=int), slsqp_iters=np.array(nit_history),
             success=opt.success, mu0=mu0, max_mu=float(np.max(opt.mu)), x=opt.x,
             time=time.perf_counter() - t0, sweeps=len(opt.feas_history))
    print(f"{name}: {r['sweeps']} sweeps, {len(updates)} multiplier updates, {nit_history[-1]} SLSQP iterations, "
          f"reports {'success' if r['success'] else 'failure'}, optimality residual {r['opt_history'][-1]:.2e}, "
          f"feasibility {r['feas_history'][-1]:.2e}, relative error {r['error'][-1]:.2e}, mu0 {mu0:.3g}, "
          f"max mu {r['max_mu']:.3g}, {r['time']:.0f} s")

    # reload before writing, so that runs of other methods in the meantime are kept
    if os.path.exists(OUT):
        saved = {**dict(np.load(OUT)), **{k: v for k, v in saved.items() if k.startswith("ref_")}}
    saved.update({f"{name}_{key}": value for key, value in r.items()})
    np.savez(OUT, **saved)
