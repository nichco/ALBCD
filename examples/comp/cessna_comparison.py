"""Solve the Cessna aerostructural problem of examples/jax/cessna with ALBCD, ALC and ADMM.

All three methods use the same subproblems (AeroSubproblem and StructSubproblem from
examples/jax/cessna/cessna.py), the same initial point and the same SLSQP settings:

- ALBCD with the example's settings (make_albcd in cessna.py)
- ALC (Tosserams, Etman and Rooda, 2008) with its published settings: inner loop stopped on the
  relative merit change eps / 100, beta = 2.2, gamma = 0.4, and the initial weight of its Eq. 29
- canonical ADMM (Boyd et al., 2011): one sweep per multiplier update with a fixed penalty
  mu = 10 (ALBCD's initial penalty), stopped when the primal residual (feasibility) and the dual
  residual (the blocks' KKT residual) reach ALBCD's final tolerances

Writes cessna_comparison.npz, which fig_cessna_comparison.py plots.
"""

import os
import sys

# single-threaded BLAS and XLA: the runs are then reproducible (ALBCD's sweep count on this problem
# changes with the thread count)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "jax", "cessna"))

import cessna as C  # noqa: E402  (the example; its __main__ block does not run on import)
from albcd import ALC, ADMM, initial_penalty  # noqa: E402

FEAS_TOL = 3e-4  # the example's feasibility tolerance, also ALC's epsilon and ADMM's primal tolerance
OPT_TOL = 1e-3   # ALBCD's final optimality tolerance, also ADMM's dual tolerance
MU_ADMM = 10.0   # ADMM's fixed penalty: ALBCD's initial penalty

data0 = {"weight": float(C.weight_init), "aero_loads": np.array(C.aero_loads_init), "CD": float(C.CD_init)}
phi0 = np.array(C.coupling(C.aero_loads_init, C.x0[C.LOADS_SLICE], C.weight_init, C.x0[C.WEIGHT_SLICE][0]))


def objective(x, data):
    """f in the merit function: the aero block's 1e2 * CD (the struct block leaves it out as a constant)."""
    return 1e2 * data["CD"]


def subproblems():
    return [C.AeroSubproblem(C.AERO_INDEX), C.StructSubproblem(C.STRUCT_INDEX)]


def results(opt, mu0):
    """Histories with one entry per sweep. Every sweep solves both blocks, so x after sweep k is
    x_history[2k]."""
    return dict(opt_history=np.array(opt.opt_history), feas_history=np.array(opt.feas_history),
                error=C.design_error(opt.x_history[2::2]), success=opt.success, mu0=mu0, x=opt.x,
                time=opt.tf, sweeps=len(opt.feas_history))


runs = {}

print("\n=== ALBCD ===")
albcd = C.make_albcd()
albcd.solve()
runs["albcd"] = results(albcd, 10.0)

# ALC's initial weight (Eq. 29): one inner loop with a small penalty and zero multipliers estimates
# the coupling constraints, then the penalty term is set to 10% of the initial objective
mu0 = initial_penalty(subproblems(), C.x0, C.N_CON, objective, phi0=phi0, data0=data0,
                      mu_small=1e-3, inner_tol=FEAS_TOL / 100)
print(f"\nALC initial penalty (Eq. 29): mu0 = {mu0:.4g}")

print("\n=== ALC ===")
alc = ALC(subproblems(), C.x0, np.full(C.N_CON, mu0), objective, phi0=phi0, data0=data0, feas_tol=FEAS_TOL)
alc.solve()
runs["alc"] = results(alc, mu0)

print("\n=== ADMM ===")
admm = ADMM(subproblems(), C.x0, MU_ADMM, n_con=C.N_CON, data0=data0, feas_tol=FEAS_TOL, opt_tol=OPT_TOL)
admm.solve()
runs["admm"] = results(admm, MU_ADMM)

print()
for name, r in runs.items():
    print(f"{name:6s}: {r['sweeps']:3d} sweeps, reports {'success' if r['success'] else 'failure'}, "
          f"optimality residual {r['opt_history'][-1]:.2e}, feasibility {r['feas_history'][-1]:.2e}, "
          f"relative error {r['error'][-1]:.2e}")

np.savez(os.path.join(HERE, "cessna_comparison.npz"),
         **{f"{name}_{key}": value for name, r in runs.items() for key, value in r.items()})
