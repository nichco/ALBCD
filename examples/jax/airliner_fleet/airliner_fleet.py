"""Airliner fleet trajectory optimization with ALBCD.

N flights of the same regional airliner, with ranges and takeoff masses sampled with a
Latin hypercube, minimize the fleet's total fuel subject to a total block-time budget. The
budget is 10 % tighter than the sum of the flights' minimum-fuel times, so the flights have
to trade fuel for time. Each flight (models.py) is one block:

    FlightSubproblem i  owns [h coefficients_i, v coefficients_i, tau_i]
                        local constraints: throttle and CL limits along the flight,
                                           flight time = tau_i

where tau_i is the flight's block-time allocation. The single coupling constraint is the
budget, phi = sum(tau_i) - T_budget, which is linear in the allocations: every flight
couples to the others through one scalar, and once the multiplier on the budget is fixed,
the flights are independent. That multiplier is the fleet's cost index (kg of fuel per
second of block time): each block minimizes its fuel plus y times its block time, plus the
penalty (mu/2) phi^2.

Run this file to solve the problem and compare the result with the monolithic solution in
monolithic_solution_N{N}.npz (generate it with monolithic.py). The solution and convergence
history are saved to convergence_N{N}.npz, which the fig_*.py scripts plot. Pass the number
of flights as an argument, e.g. `python airliner_fleet.py 8`.
"""

import os
# SLSQP's matrices are small (about 120 x 900 per block), and OpenBLAS's multithreading
# costs more than it saves on them. Must be set before numpy/scipy load OpenBLAS.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import numpy as np
import jax
import jax.numpy as jnp
import modopt as mo
from scipy.stats import qmc
from albcd import ALBCD, Subproblem
import warnings
warnings.filterwarnings("ignore")

from models import nvar, cl, cu, xl, xu, x_scaler, setup_flight, flight_outputs, initial_guess, last_call_cache, peak_memory

HERE = os.path.dirname(os.path.abspath(__file__))

# fleet: range (m) and takeoff mass (kg) of each flight
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4  # number of flights
ranges, masses = qmc.scale(qmc.LatinHypercube(d=2, seed=0).random(N), [2000e3, 22000.0], [5000e3, 28000.0]).T
flights = [setup_flight(rf, m0) for rf, m0 in zip(ranges, masses)]

# block-time budget (ks): 10 % below the sum of the flights' minimum-fuel times, from a fit of
# single-flight minimum-fuel solves at 2,000-5,000 km and 22-28 t (within 1 min; the takeoff
# mass changes the minimum-fuel time by about 0.1 ks, so only the range enters)
T_budget = 0.9 * np.sum(0.276 + 5.468e-6 * ranges)

# global x = [v_0, v_1, ..., v_{N-1}], where v_i = [h coefficients, v coefficients, tau] of flight i
taus = lambda x: x[nvar - 1::nvar]

y_history = []  # the multiplier on the budget (the cost index) at each sweep

# compiled once and shared by every block (the flight's data is an argument)
outputs_fn = jax.jit(flight_outputs)
jac_fn = jax.jit(jax.jacfwd(flight_outputs))


def kkt_rows(J):
    """The rows of the local constraint Jacobian J in the order and sign of SLSQP's multipliers:
    equalities, then lower bounds, then upper bounds (negated), so that grad f = rows^T multipliers."""
    eq = cl == cu
    return np.concatenate((J[eq], J[(cl != -np.inf) & ~eq], -J[(cu != np.inf) & ~eq]))


class FlightSubproblem(Subproblem):
    """Flight i: owns its trajectory and its block-time allocation tau_i."""

    def __init__(self, i):
        fl = flights[i]
        # this flight's [local constraints, fuel] and their Jacobian, each computed once per point
        self.outputs = last_call_cache(lambda v: np.asarray(outputs_fn(jnp.asarray(v), fl)))
        self.outputs_jac = last_call_cache(lambda v: np.asarray(jac_fn(jnp.asarray(v), fl)))
        super().__init__(slice(i * nvar, (i + 1) * nvar))

    def budget(self, v, x):
        """The budget constraint phi, with this flight's allocation taken from v and the others' from x."""
        return v[-1] + taus(x).sum() - x[self.index][-1] - T_budget

    def objective(self, v, x, y, mu):
        """Augmented Lagrangian: this flight's fuel plus the budget's multiplier and penalty terms."""
        phi = self.budget(v, x)
        return self.outputs(v)[-1] + y[0] * phi + 0.5 * mu[0] * phi**2

    def gradient(self, v, x, y, mu):
        grad = self.outputs_jac(v)[-1].copy()  # d(fuel)/dv
        grad[-1] += y[0] + mu[0] * self.budget(v, x)
        return grad

    def solve(self, x, y, mu, data, outputs) -> None:
        if self.index.start == 0:  # the first block of a sweep
            y_history.append(float(y[0]))
        prob = mo.ProblemLite(x0=np.array(self.decompose(x)),
                              obj=lambda v: self.objective(v, x, y, mu),
                              grad=lambda v: self.gradient(v, x, y, mu),
                              con=lambda v: self.outputs(v)[:-1],
                              jac=lambda v: self.outputs_jac(v)[:-1],
                              xl=xl, xu=xu, cl=cl, cu=cu, x_scaler=x_scaler)
        optimizer = mo.SLSQP(prob, solver_options={"maxiter": 500, "ftol": 1e-9}, turn_off_outputs=True)
        optimizer.solve()
        v_new = optimizer.results["x"] / x_scaler

        # cache SLSQP's multipliers and the matching rows of the local constraints' Jacobian for residual()
        self._multipliers = np.asarray(optimizer.results["multipliers"])
        self._rows = kkt_rows(self.outputs_jac(v_new)[:-1])

        outputs["x"] = self.recompose(x, v_new)
        outputs["phi"] = np.array([taus(outputs["x"]).sum() - T_budget])

    def residual(self, x, y, mu, data) -> float:
        v = self.decompose(x)

        # Lagrangian gradient (with SLSQP's multipliers for the local constraints) projected onto
        # the box bounds: zero iff v is a KKT point of this block's subproblem
        grad_L = self.gradient(v, x, y, mu) - self._rows.T @ self._multipliers
        return float(np.max(np.abs(v - np.clip(v - grad_L, xl, xu))))


x0 = np.concatenate([initial_guess(fl) for fl in flights])

# compile before the timed solve
outputs_fn(jnp.asarray(x0[:nvar]), flights[0]), jac_fn(jnp.asarray(x0[:nvar]), flights[0])

# The penalty starts small. The flights interact only through the penalty term (mu/2) phi^2,
# which couples every pair of allocations with strength mu, while one flight's fuel changes
# with its own allocation with a curvature of only 0.1-0.2 t/ks^2. Starting at mu >> that
# (e.g. mu0 = 1 at N = 4) makes the block sweeps crawl, and their inexact results then keep
# growing mu. Starting small lets the usual growth rule raise mu only as feasibility requires.
opt = ALBCD(subproblems=[FlightSubproblem(i) for i in range(N)],
            x0=x0,
            mu0=np.array([0.01]),
            max_mu=1e3,
            rho=2.0,
            tau=0.5,
            feas_tol=1e-4,
            opt_tol=[1e-2, 1e-4],
            max_y=1e6,
            max_outer_iter=100,
            max_inner_iter=20)

memory_before = peak_memory()  # MB: imports, setup and JAX compilation
opt.solve()
memory = peak_memory()  # MB, including the solve


blocks = opt.x.reshape(N, nvar)
fuel_kg = np.array([1e3 * float(outputs_fn(jnp.asarray(v), fl)[-1]) for v, fl in zip(blocks, flights)])
print(f"Fleet fuel: {fuel_kg.sum():.1f} kg, cost index (y): {opt.y[0]:.4f} kg/s, "
      f"budget violation: {opt.phi[0]:.2e} ks")
for i in range(N):
    print(f"Flight {i} ({ranges[i] / 1e3:.0f} km, {masses[i]:.0f} kg): fuel {fuel_kg[i]:.1f} kg, block time {blocks[i, -1]:.3f} ks")

# compare with the monolithic solution (the same problem solved with one SLSQP)
reference = os.path.join(HERE, f"monolithic_solution_N{N}.npz")
if os.path.exists(reference):
    solution = np.load(reference)
    scale = np.tile(x_scaler, N)
    error = np.linalg.norm((np.array(opt.x_history) - solution["z"]) * scale, axis=1) / np.linalg.norm(solution["z"] * scale)
    print(f"Monolithic: fleet fuel {solution['fuel'].sum():.1f} kg, cost index {float(solution['cost_index']):.4f} kg/s, "
          f"solve time {float(solution['time']):.1f} s")
    print(f"Fleet fuel difference: {fuel_kg.sum() - solution['fuel'].sum():+.2f} kg, relative error in x: {error[-1]:.2e}")
else:
    error = None
    print(f"No monolithic solution for N = {N}; run monolithic.py to create it.")

# solution and convergence data, for the figures. opt_history, feas_history and y_history have one
# entry per sweep; error has one per subproblem solve, and a leading entry for x0

np.savez(os.path.join(HERE, f"fleet_albcd_N{N}.npz"),
         opt_history=opt.opt_history, feas_history=opt.feas_history, y_history=y_history,
         error=np.array([]) if error is None else error, x=opt.x, fuel=fuel_kg, cost_index=opt.y[0],
         ranges=ranges, masses=masses, feas_tol=opt.feas_tol, opt_tol=opt.opt_tol,
         success=opt.success, time=opt.tf, solves=len(opt.x_history) - 1,
         peak_memory=memory, peak_memory_before_solve=memory_before)


# gifs of the trajectories after each sweep
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from models import simulate

sweeps = np.array(opt.x_history)[::N]  # x0, then x after each sweep
tau_history = sweeps[:, nvar - 1::nvar]  # block-time allocations (ks)
sims = [[simulate(jnp.asarray(v), fl) for v, fl in zip(x.reshape(N, nvar), flights)] for x in sweeps]
colors = plt.cm.plasma(np.linspace(0, 0.9, N))
labels = [f"{r / 1e3:,.0f} km" for r in ranges]

# # altitude vs range
# fig, ax = plt.subplots(figsize=(6.5, 3))
# lines = [ax.plot([], [], color=c, lw=1.5, label=label)[0] for c, label in zip(colors, labels)]
# ax.set(xlabel="Range (km)", ylabel="Altitude (km)", title=" ", xlim=(0, ranges.max() / 1e3), ylim=(0, 15))
# ax.legend(loc="lower center", ncol=4, fontsize=8)
# fig.tight_layout()

# def update(k):
#     for line, fl, s in zip(lines, flights, sims[k]):
#         line.set_data(np.asarray(fl["r_nodes"]) / 1e3, np.asarray(s["h"]) / 1e3)
#     ax.set_title(f"Sweep {k}")
#     return lines

# FuncAnimation(fig, update, frames=len(sweeps)).save(os.path.join(HERE, f"trajectories_N{N}.gif"), writer=PillowWriter(fps=4))

# altitude vs time, with the block-time allocations stacked against the budget below
fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.5, 4), gridspec_kw={"height_ratios": [3, 1]})
lines = [ax.plot([], [], color=c, lw=1.5, label=label)[0] for c, label in zip(colors, labels)]
t_max = max(float(s["t"][-1]) for sim in sims for s in sim) / 1e3
ax.set(xlabel="Time (ks)", ylabel="Altitude (km)", title=" ", xlim=(0, t_max), ylim=(0, 15))
ax.legend(loc="lower center", ncol=4, fontsize=8)
bars = bx.barh(0, np.ones(N), color=colors, edgecolor="w")
bx.axvline(T_budget, color="k", ls="--")
bx.text(T_budget, 1.02, "Time limit", transform=bx.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
bx.set(xlabel="Total block time (ks)", yticks=[], xlim=(0, 1.05 * tau_history.sum(1).max()))
fig.tight_layout()

def update_time(k):
    left = np.cumsum(tau_history[k]) - tau_history[k]
    for line, bar, s, l, w in zip(lines, bars, sims[k], left, tau_history[k]):
        line.set_data(np.asarray(s["t"]) / 1e3, np.asarray(s["h"]) / 1e3)
        bar.set_x(l), bar.set_width(w)
    ax.set_title(f"Sweep {k}: budget violation {tau_history[k].sum() - T_budget:+.3f} ks")
    return lines + list(bars)

FuncAnimation(fig, update_time, frames=len(sweeps)).save(os.path.join(HERE, f"trajectories_time_N{N}.gif"), writer=PillowWriter(fps=4))
