"""Data for fig_negotiation.py: the ALBCD solve of airliner_fleet.py recorded flight solve by flight solve, and
each flight's trade-off between fuel and block time.

Runs the same solve as airliner_fleet.py (same flights, budget and ALBCD settings) and records, after every
flight solve, the block times and fuel of all flights, the multiplier y and penalty mu on the budget, and the
budget violation phi before and after the solve. It then traces each flight's trade-off curve: the minimum
fuel of that flight alone with its block time fixed, over a grid of block times, with d(fuel)/d(block time)
from the multiplier of its flight-time constraint, and its own minimum-fuel point (block time free).

Saves negotiation_N{N}.npz. Pass the number of flights as an argument, e.g. `python negotiation_data.py 4`.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # small SLSQP matrices; must precede numpy
import sys
import numpy as np
import jax
import jax.numpy as jnp
import modopt as mo
from scipy.stats import qmc
from albcd import ALBCD, Subproblem
import warnings
warnings.filterwarnings("ignore")

from models import nvar, cl, cu, xl, xu, x_scaler, setup_flight, flight_outputs, initial_guess, last_call_cache

HERE = os.path.dirname(os.path.abspath(__file__))

# flights and block-time budget (ks), as in airliner_fleet.py
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ranges, masses = qmc.scale(qmc.LatinHypercube(d=2, seed=0).random(N), [2000e3, 22000.0], [5000e3, 28000.0]).T
flights = [setup_flight(rf, m0) for rf, m0 in zip(ranges, masses)]
T_budget = 0.9 * np.sum(0.276 + 5.468e-6 * ranges)
taus = lambda x: x[nvar - 1::nvar]

outputs_fn = jax.jit(flight_outputs)
jac_fn = jax.jit(jax.jacfwd(flight_outputs))
outputs = [last_call_cache(lambda v, fl=fl: np.asarray(outputs_fn(jnp.asarray(v), fl))) for fl in flights]
outputs_jac = [last_call_cache(lambda v, fl=fl: np.asarray(jac_fn(jnp.asarray(v), fl))) for fl in flights]


def kkt_rows(J):
    """Rows of the local constraint Jacobian in the order and sign of SLSQP's multipliers (as in airliner_fleet.py)."""
    eq = cl == cu
    return np.concatenate((J[eq], J[(cl != -np.inf) & ~eq], -J[(cu != np.inf) & ~eq]))


# one row per flight solve
history = dict(flight=[], tau=[], fuel=[], y=[], mu=[], phi_before=[], phi_after=[])
fuel_now = np.array([float(outputs[i](initial_guess(fl))[-1]) for i, fl in enumerate(flights)])


class FlightSubproblem(Subproblem):
    """Flight i of airliner_fleet.py, recording each solve."""

    def __init__(self, i):
        self.i = i
        super().__init__(slice(i * nvar, (i + 1) * nvar))

    def budget(self, v, x):
        return v[-1] + taus(x).sum() - x[self.index][-1] - T_budget

    def objective(self, v, x, y, mu):
        phi = self.budget(v, x)
        return outputs[self.i](v)[-1] + y[0] * phi + 0.5 * mu[0] * phi**2

    def gradient(self, v, x, y, mu):
        grad = outputs_jac[self.i](v)[-1].copy()
        grad[-1] += y[0] + mu[0] * self.budget(v, x)
        return grad

    def solve(self, x, y, mu, data, outputs_) -> None:
        prob = mo.ProblemLite(x0=np.array(self.decompose(x)),
                              obj=lambda v: self.objective(v, x, y, mu),
                              grad=lambda v: self.gradient(v, x, y, mu),
                              con=lambda v: outputs[self.i](v)[:-1],
                              jac=lambda v: outputs_jac[self.i](v)[:-1],
                              xl=xl, xu=xu, cl=cl, cu=cu, x_scaler=x_scaler)
        optimizer = mo.SLSQP(prob, solver_options={"maxiter": 500, "ftol": 1e-9}, turn_off_outputs=True)
        optimizer.solve()
        v_new = optimizer.results["x"] / x_scaler
        self._multipliers = np.asarray(optimizer.results["multipliers"])
        self._rows = kkt_rows(outputs_jac[self.i](v_new)[:-1])

        x_new = self.recompose(x, v_new)
        fuel_now[self.i] = float(outputs[self.i](v_new)[-1])
        for key, value in (("flight", self.i), ("tau", taus(x_new)), ("fuel", fuel_now.copy()), ("y", float(y[0])),
                           ("mu", float(mu[0])), ("phi_before", taus(x).sum() - T_budget),
                           ("phi_after", taus(x_new).sum() - T_budget)):
            history[key].append(value)
        outputs_["x"] = x_new
        outputs_["phi"] = np.array([taus(x_new).sum() - T_budget])

    def residual(self, x, y, mu, data) -> float:
        v = self.decompose(x)
        grad_L = self.gradient(v, x, y, mu) - self._rows.T @ self._multipliers
        return float(np.max(np.abs(v - np.clip(v - grad_L, xl, xu))))


x0 = np.concatenate([initial_guess(fl) for fl in flights])
opt = ALBCD(subproblems=[FlightSubproblem(i) for i in range(N)], x0=x0, mu0=np.array([0.01]), max_mu=1e3,
            rho=2.0, tau=0.5, feas_tol=1e-4, opt_tol=[1e-2, 1e-4], max_y=1e6, max_outer_iter=100,
            max_inner_iter=20, verbose=False)
opt.solve()
print(f"ALBCD: {len(opt.feas_history)} sweeps, cost index {opt.y[0]:.4f} kg/s")
blocks = opt.x.reshape(N, nvar)


def fixed_time(i, hv0, tau):
    """Minimum fuel of flight i alone with its block time fixed at tau: (h and v coefficients, fuel (t),
    d(fuel)/d(tau) (t/ks)). The derivative is the multiplier of the flight-time constraint, its only equality."""
    full = lambda hv: np.append(hv, tau)
    prob = mo.ProblemLite(x0=hv0, obj=lambda hv: outputs[i](full(hv))[-1],
                          grad=lambda hv: outputs_jac[i](full(hv))[-1, :-1],
                          con=lambda hv: outputs[i](full(hv))[:-1], jac=lambda hv: outputs_jac[i](full(hv))[:-1, :-1],
                          xl=xl[:-1], xu=xu[:-1], cl=cl, cu=cu, x_scaler=x_scaler[:-1])
    optimizer = mo.SLSQP(prob, solver_options={"maxiter": 500, "ftol": 1e-10}, turn_off_outputs=True)
    optimizer.solve()
    hv = optimizer.results["x"] / x_scaler[:-1]
    return hv, float(outputs[i](full(hv))[-1]), float(optimizer.results["multipliers"][0])


def minimum_fuel(i, v0):
    """Flight i's own minimum-fuel point, with its block time free: (block time (ks), fuel (t))."""
    prob = mo.ProblemLite(x0=v0, obj=lambda v: outputs[i](v)[-1], grad=lambda v: outputs_jac[i](v)[-1],
                          con=lambda v: outputs[i](v)[:-1], jac=lambda v: outputs_jac[i](v)[:-1],
                          xl=xl, xu=xu, cl=cl, cu=cu, x_scaler=x_scaler)
    optimizer = mo.SLSQP(prob, solver_options={"maxiter": 1000, "ftol": 1e-10}, turn_off_outputs=True)
    optimizer.solve()
    v = optimizer.results["x"] / x_scaler
    return v[-1], float(outputs[i](v)[-1])


# trade-off curves: from the ALBCD solution outward in both directions, each solve warm-started from the last
tau_history = np.array(history["tau"])
n_grid = 17
curve_tau, curve_fuel, curve_slope, min_fuel = [], [], [], []
for i in range(N):
    t_min_fuel, f_min_fuel = minimum_fuel(i, blocks[i])
    min_fuel.append((t_min_fuel, f_min_fuel))
    lo = min(tau_history[:, i].min(), blocks[i, -1]) - 0.3
    hi = max(tau_history[:, i].max(), x0[i * nvar + nvar - 1], t_min_fuel) + 0.3
    grid = np.linspace(lo, hi, n_grid)
    k = int(np.argmin(np.abs(grid - blocks[i, -1])))
    rows = {}
    for order in (range(k, n_grid), range(k - 1, -1, -1)):
        hv = blocks[i, :-1]
        for j in order:
            hv, fuel, slope = fixed_time(i, hv, grid[j])
            rows[j] = (fuel, slope)
    curve_tau.append(grid)
    curve_fuel.append([rows[j][0] for j in range(n_grid)])
    curve_slope.append([rows[j][1] for j in range(n_grid)])
    print(f"Flight {i}: minimum-fuel block time {t_min_fuel:.3f} ks, solution {blocks[i, -1]:.3f} ks")

solution = np.load(os.path.join(HERE, f"monolithic_solution_N{N}.npz"))
np.savez(os.path.join(HERE, f"negotiation_N{N}.npz"),
         **{k: np.array(v) for k, v in history.items()},
         opt_history=opt.opt_history, feas_history=opt.feas_history, tau0=taus(x0), T_budget=T_budget,
         ranges=ranges, masses=masses, tau_solution=blocks[:, -1], cost_index=opt.y[0],
         tau_monolithic=solution["tau"], cost_index_monolithic=float(solution["cost_index"]),
         curve_tau=np.array(curve_tau), curve_fuel=np.array(curve_fuel), curve_slope=np.array(curve_slope),
         min_fuel=np.array(min_fuel))
