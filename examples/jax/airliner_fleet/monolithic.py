"""Monolithic reference solution of the airliner fleet problem in airliner_fleet.py.

The same problem (the same fleet and budget, declared below as in airliner_fleet.py), solved
over every flight's variables at once: minimize the fleet's total fuel subject to every
flight's path constraints and the block-time budget. Two solvers:

- slsqp: modopt's SLSQP, with dense matrices.
- ipopt: IPOPT through CasADi, with a limited-memory Hessian and the constraint Jacobian's
  sparsity (one dense block per flight plus the budget row) declared, so its linear algebra
  exploits the independence of the flights.

The constraint Jacobian is assembled from per-flight Jacobians, so JAX's cost grows linearly
with N and the solve time is set by the optimizer. The solution is saved to
monolithic_{solver}_N{N}.npz, which airliner_fleet.py compares the ALBCD solution against.
Pass the number of flights and the solver as arguments, e.g. `python monolithic.py 8 ipopt`.
"""

import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
import modopt as mo
import casadi as ca
from scipy.stats import qmc
import warnings
warnings.filterwarnings("ignore")

from models import nvar, cl, cu, xl, xu, x_scaler, setup_flight, flight_outputs, initial_guess, last_call_cache

HERE = os.path.dirname(os.path.abspath(__file__))

# the fleet and budget of airliner_fleet.py
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4  # number of flights
solver = sys.argv[2] if len(sys.argv) > 2 else "slsqp"
ranges, masses = qmc.scale(qmc.LatinHypercube(d=2, seed=0).random(N), [2000e3, 22000.0], [5000e3, 28000.0]).T
flights = [setup_flight(rf, m0) for rf, m0 in zip(ranges, masses)]
T_budget = 0.97 * np.sum(0.298 + 5.488e-6 * ranges)

# the flights' arrays stacked along a leading axis, for vmap
fleet = jax.tree.map(lambda *a: jnp.stack(a), *flights)

# global z = [v_0, v_1, ..., v_{N-1}], where v_i = [h coefficients, v coefficients, tau] of flight i
blocks = lambda z: z.reshape(N, nvar)
nc = len(cl)
tile = lambda a: np.tile(a, N)

# every flight's [local constraints, fuel] (N, nc + 1) and its Jacobian (N, nc + 1, nvar), each
# computed once per point and shared by the objective, the constraints and their derivatives
outputs_all = jax.jit(jax.vmap(flight_outputs))
jac_all = jax.jit(jax.vmap(jax.jacfwd(flight_outputs)))
outputs = last_call_cache(lambda z: np.asarray(outputs_all(jnp.asarray(blocks(z)), fleet)))
outputs_jac = last_call_cache(lambda z: np.asarray(jac_all(jnp.asarray(blocks(z)), fleet)))

fleet_fuel = lambda z: float(outputs(z)[:, -1].sum())
fuel_grad = lambda z: outputs_jac(z)[:, -1].ravel()


def constraints(z):
    """Every flight's local constraints, then the budget sum(tau) - T_budget."""
    return np.append(outputs(z)[:, :-1].ravel(), z[nvar - 1::nvar].sum() - T_budget)


z0 = np.concatenate([initial_guess(fl) for fl in flights])
constraints(z0), outputs_jac(z0)  # compile before timing the solve


def solve_slsqp():
    def jacobian(z):
        J = np.zeros((N * nc + 1, N * nvar))
        for i, Ji in enumerate(outputs_jac(z)[:, :-1]):
            J[i * nc:(i + 1) * nc, i * nvar:(i + 1) * nvar] = Ji
        J[-1, nvar - 1::nvar] = 1.0  # d(sum tau)/dz
        return J

    problem = mo.ProblemLite(x0=z0, obj=fleet_fuel, grad=fuel_grad, con=constraints, jac=jacobian,
                             xl=tile(xl), xu=tile(xu), cl=np.append(tile(cl), 0.0), cu=np.append(tile(cu), 0.0),
                             x_scaler=tile(x_scaler))
    optimizer = mo.SLSQP(problem, solver_options={"maxiter": 2000, "ftol": 1e-9}, turn_off_outputs=True)
    optimizer.solve()
    # the budget is the last of the equality rows, which come first in SLSQP's multipliers; SLSQP's
    # Lagrangian is f - lambda c, so the budget's multiplier in f + y c (the cost index) is -lambda
    return (optimizer.results["x"] / tile(x_scaler), -float(optimizer.results["multipliers"][N]),
            optimizer.total_time, optimizer.results["nit"], optimizer.results["success"])


def solve_ipopt():
    s = tile(x_scaler)  # IPOPT works on the scaled variables z * s

    # Jacobian sparsity in CasADi's column-major order: flight i's block is its nvar columns of nc
    # rows, and its last column (tau) also has the budget row, which comes after every flight's rows
    rows = [np.append(np.arange(i * nc, (i + 1) * nc), N * nc) if j == nvar - 1 else np.arange(i * nc, (i + 1) * nc)
            for i in range(N) for j in range(nvar)]
    pattern = ca.Sparsity(N * nc + 1, N * nvar, np.cumsum([0] + [len(r) for r in rows]).tolist(),
                          np.concatenate(rows).tolist())

    def jac_values(z):
        A = outputs_jac(z)[:, :-1].transpose(0, 2, 1) / x_scaler[None, :, None]  # (N, nvar, nc) per column
        return np.concatenate((A.reshape(N, -1), np.full((N, 1), 1.0 / x_scaler[-1])), axis=1).ravel()

    class Fn(ca.Callback):
        def __init__(self, name, f, sparsity):
            ca.Callback.__init__(self)
            self.f, self.sparsity = f, sparsity
            self.construct(name, {})
        def get_n_in(self): return 1
        def get_n_out(self): return 1
        def get_sparsity_in(self, i): return ca.Sparsity.dense(N * nvar, 1)
        def get_sparsity_out(self, i): return self.sparsity
        def eval(self, arg): return [self.f(np.asarray(arg[0]).ravel() / s)]

    f = Fn("f", fleet_fuel, ca.Sparsity.dense(1, 1))
    g = Fn("g", constraints, ca.Sparsity.dense(N * nc + 1, 1))
    df = Fn("df", lambda z: (fuel_grad(z) / s)[None, :], ca.Sparsity.dense(1, N * nvar))
    dg = Fn("dg", lambda z: ca.DM(pattern, jac_values(z)), pattern)
    x, p = ca.MX.sym("x", N * nvar), ca.MX.sym("p", 0, 1)
    options = {"grad_f": ca.Function("G", [x, p], [0, df(x)]), "jac_g": ca.Function("J", [x, p], [0, dg(x)]),
               "no_nlp_grad": True, "print_time": False,
               "ipopt": {"hessian_approximation": "limited-memory", "tol": 1e-8, "max_iter": 3000, "print_level": 0}}
    nlp = ca.nlpsol("nlp", "ipopt", {"x": x, "f": f(x), "g": g(x)}, options)
    t0 = time.perf_counter()
    sol = nlp(x0=z0 * s, lbx=tile(xl) * s, ubx=tile(xu) * s, lbg=np.append(tile(cl), 0.0), ubg=np.append(tile(cu), 0.0))
    elapsed = time.perf_counter() - t0
    stats = nlp.stats()
    # CasADi's Lagrangian is f + lam_g g, so the budget's multiplier is the cost index directly
    return (sol["x"].full().ravel() / s, float(sol["lam_g"].full()[-1, 0]), elapsed, stats["iter_count"], stats["success"])


z, cost_index, elapsed, nit, success = {"slsqp": solve_slsqp, "ipopt": solve_ipopt}[solver]()
fuel_kg = 1e3 * outputs(z)[:, -1]
tau = blocks(z)[:, -1]

print(f"{solver}: {'converged' if success else 'did not converge'} in {nit} iterations, {elapsed:.1f} s | "
      f"fleet fuel {fuel_kg.sum():.1f} kg, cost index {cost_index:.4f} kg/s, budget violation {tau.sum() - T_budget:.2e} ks")
for i in range(N):
    print(f"Flight {i} ({ranges[i] / 1e3:.0f} km, {masses[i]:.0f} kg): fuel {fuel_kg[i]:.1f} kg, block time {tau[i]:.3f} ks")

np.savez(os.path.join(HERE, f"monolithic_{solver}_N{N}.npz"), z=z, fuel=fuel_kg, tau=tau, cost_index=cost_index,
         time=elapsed, nit=nit, success=success)
