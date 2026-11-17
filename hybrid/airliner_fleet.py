"""The airliner fleet problem (examples/jax/airliner_fleet) with the hybrid ALBCD interface.

One airliner flies N flights and minimizes its total fuel subject to a total block-time budget. The
blocks are those of examples/jax/airliner_fleet/airliner_fleet.py: flight i owns its trajectory and its
block-time allocation tau_i, and the budget sum(tau_i) = T_budget is the one coupling constraint. The full
model is that of examples/jax/airliner_fleet/monolithic_opensqp.py: the same variables,
z = x = [flight 0 variables, ..., flight N-1 variables], with every flight's local constraints followed
by the budget.

Run this file to solve the problem with the hybrid solver; pass the number of flights as an argument
(default 4), e.g. ``python airliner_fleet.py 8``. make_solver() builds the solver for switching_study.py.
"""

import os

# single-threaded BLAS and XLA, as in examples/jax/airliner_fleet/albcd_timing.py, so the timings are
# reproducible and both phases run with the same resources. Must be set before numpy loads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import sys
import numpy as np
import jax
import jax.numpy as jnp
import modopt as mo
from scipy.stats import qmc
import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
FLEET = os.path.join(HERE, "..", "examples", "jax", "airliner_fleet")
sys.path.insert(0, FLEET)  # models.py and the modules it imports

from albcd import Subproblem
from hybrid_albcd import HybridALBCD, init_multipliers, merit_below, positive_definite
from models import nvar, cl, cu, xl, xu, x_scaler, setup_flight, flight_outputs, initial_guess, last_call_cache

taus = lambda x: x[nvar - 1::nvar]
nc = len(cl)  # local constraints of one flight

# compiled once and shared by every block and by the full model (the flight's data is an argument)
outputs_fn = jax.jit(flight_outputs)
jac_fn = jax.jit(jax.jacfwd(flight_outputs))
outputs_all = jax.jit(jax.vmap(flight_outputs))
jac_all = jax.jit(jax.vmap(jax.jacfwd(flight_outputs)))

# Hessian of one flight's Lagrangian, fuel - lam^T (local constraints), for the full model's Hessian warm start
hess_all = jax.jit(jax.vmap(jax.hessian(lambda v, fl, lam: jnp.append(-lam, 1.0) @ flight_outputs(v, fl))))

# the local constraints with a lower side, an upper side, or both sides equal: the order of SLSQP's multipliers
EQ = cl == cu
LOWER = (cl != -np.inf) & ~EQ
UPPER = (cu != np.inf) & ~EQ


def make_fleet(N):
    """Ranges (m), takeoff masses (kg), flight data and block-time budget (ks), as in airliner_fleet.py."""
    ranges, masses = qmc.scale(qmc.LatinHypercube(d=2, seed=0).random(N), [2000e3, 22000.0], [5000e3, 28000.0]).T
    flights = [setup_flight(rf, m0) for rf, m0 in zip(ranges, masses)]
    return ranges, masses, flights, 0.9 * np.sum(0.276 + 5.468e-6 * ranges)


# ---------------------------------------------------------------------------------------------
# subproblems: as in examples/jax/airliner_fleet/airliner_fleet.py
# ---------------------------------------------------------------------------------------------

class FlightSubproblem(Subproblem):
    """Flight i: owns its trajectory and its block-time allocation tau_i."""

    def __init__(self, i, flights, T_budget):
        fl = flights[i]
        self.i, self.T_budget = i, T_budget
        # this flight's [local constraints, fuel] and their Jacobian, each computed once per point
        self.outputs = last_call_cache(lambda v: np.asarray(outputs_fn(jnp.asarray(v), fl)))
        self.outputs_jac = last_call_cache(lambda v: np.asarray(jac_fn(jnp.asarray(v), fl)))
        super().__init__(slice(i * nvar, (i + 1) * nvar))

    def budget(self, v, x):
        """The budget constraint phi, with this flight's allocation taken from v and the others' from x."""
        return v[-1] + taus(x).sum() - x[self.index][-1] - self.T_budget

    def objective(self, v, x, y, mu):
        """Augmented Lagrangian: this flight's fuel plus the budget's multiplier and penalty terms."""
        phi = self.budget(v, x)
        return self.outputs(v)[-1] + y[0] * phi + 0.5 * mu[0] * phi**2

    def objective_gradient(self, v, x, y, mu):
        grad = self.outputs_jac(v)[-1].copy()  # d(fuel)/dv
        grad[-1] += y[0] + mu[0] * self.budget(v, x)
        return grad

    def solve(self, x, y, mu, data, outputs) -> None:
        prob = mo.ProblemLite(x0=np.array(self.decompose(x)),
                              obj=lambda v: self.objective(v, x, y, mu),
                              grad=lambda v: self.objective_gradient(v, x, y, mu),
                              con=lambda v: self.outputs(v)[:-1],
                              jac=lambda v: self.outputs_jac(v)[:-1],
                              xl=xl, xu=xu, cl=cl, cu=cu, x_scaler=x_scaler)
        optimizer = mo.SLSQP(prob, solver_options={"maxiter": 500, "ftol": 1e-9}, turn_off_outputs=True)
        optimizer.solve()
        v_new = optimizer.results["x"] / x_scaler

        # cache SLSQP's multipliers and the matching rows of the local constraints' Jacobian for gradient()
        self._multipliers = np.asarray(optimizer.results["multipliers"])
        J = self.outputs_jac(v_new)[:-1]
        self._rows = np.concatenate((J[EQ], J[LOWER], -J[UPPER]))

        outputs["x"] = self.recompose(x, v_new)
        outputs["phi"] = np.array([taus(outputs["x"]).sum() - self.T_budget])
        outputs[f"multipliers_{self.i}"] = self._multipliers.copy()  # shared so the full model can warm start

    def gradient(self, x, y, mu, data) -> np.ndarray:
        v = self.decompose(x)
        # Lagrangian gradient (with SLSQP's multipliers for the local constraints) projected onto
        # the box bounds: zero iff v is a KKT point of this block's subproblem
        grad_L = self.objective_gradient(v, x, y, mu) - self._rows.T @ self._multipliers
        return v - np.clip(v - grad_L, xl, xu)


# ---------------------------------------------------------------------------------------------
# full model: as in examples/jax/airliner_fleet/monolithic_opensqp.py
# ---------------------------------------------------------------------------------------------

def make_full_model(flights, T_budget, warm_multipliers=True, warm_hessian=False):
    """The full model in the form HybridALBCD takes: (x, y, data) -> (problem, init_multipliers).

    The objective is the fleet's fuel, and the constraints are every flight's local constraints followed
    by the budget. All flights are simulated in one vectorized call, and the Jacobian is assembled from
    the flights' own Jacobians. The multipliers are warm started from ALBCD: the budget's from y, and the
    local constraints' from the multipliers of each flight's last block solve, shared through ``data``.

    With ``warm_hessian``, OpenSQP's BFGS matrix starts from the exact Hessian of the Lagrangian at the ALBCD
    iterate, with those multipliers, instead of the identity. The budget is linear, so the Hessian is block
    diagonal, with one block per flight, each made positive definite on its own.
    """
    N = len(flights)
    fleet = jax.tree.map(lambda *arrays: jnp.stack(arrays), *flights)
    out = last_call_cache(lambda z: np.asarray(outputs_all(jnp.asarray(z).reshape(N, nvar), fleet)))      # (N, nc + 1)
    out_jac = last_call_cache(lambda z: np.asarray(jac_all(jnp.asarray(z).reshape(N, nvar), fleet)))     # (N, nc + 1, nvar)

    def jac(z):
        J = np.zeros((N * nc + 1, N * nvar))
        for i, Ji in enumerate(out_jac(z)):
            J[i * nc:(i + 1) * nc, i * nvar:(i + 1) * nvar] = Ji[:-1]
        J[-1, nvar - 1::nvar] = 1.0  # the budget
        return J

    def hessian(z, lam):
        """Exact Hessian of the Lagrangian, fuel - lam^T c, in the scaled variables, made positive definite. The
        budget is linear, so it is block diagonal, one block per flight, each made positive definite on its own."""
        blocks = np.asarray(hess_all(jnp.asarray(z).reshape(N, nvar), fleet, jnp.asarray(lam[:-1].reshape(N, nc))))
        H = np.zeros((N * nvar, N * nvar))
        for i, Hi in enumerate(blocks):  # in OpenSQP's scaled variables, z * x_scaler
            H[i * nvar:(i + 1) * nvar, i * nvar:(i + 1) * nvar] = positive_definite(Hi / np.outer(x_scaler, x_scaler))
        return H

    def full_model(x, y, data):
        problem = mo.ProblemLite(x0=np.array(x),
                                 obj=lambda z: np.sum(out(z)[:, -1]),
                                 grad=lambda z: out_jac(z)[:, -1].reshape(-1),
                                 con=lambda z: np.append(out(z)[:, :-1].reshape(-1), taus(z).sum() - T_budget),
                                 jac=jac,
                                 xl=np.tile(xl, N), xu=np.tile(xu, N), cl=np.append(np.tile(cl, N), 0.0),
                                 cu=np.append(np.tile(cu, N), 0.0), x_scaler=np.tile(x_scaler, N))
        started = "multipliers_0" in data  # False at the initial point, before any block solve
        if not warm_multipliers or not (started or warm_hessian):
            return problem
        # SLSQP's multipliers (equalities, lower sides, upper sides, with grad f = J_eq^T m_eq + J_l^T m_l
        # - J_u^T m_u) as one signed multiplier per local constraint; the budget's is -y, since ALBCD's
        # Lagrangian is f + y phi
        lam = np.zeros(N * nc + 1)
        for i in range(N if started else 0):
            m = data[f"multipliers_{i}"]
            local = np.zeros(nc)
            local[EQ] += m[:EQ.sum()]
            local[LOWER] += m[EQ.sum():EQ.sum() + LOWER.sum()]
            local[UPPER] -= m[EQ.sum() + LOWER.sum():]
            lam[i * nc:(i + 1) * nc] = local
        lam[-1] = -y[0]
        # at the initial point lam = 0, and the Hessian below is that of the objective
        if not warm_hessian:
            return problem, init_multipliers(problem, lam)
        return problem, init_multipliers(problem, lam), hessian(x, lam)
    full_model.hessian = hessian  # for a mid-run Hessian injection (RecordingOpenSQP.hessian)
    return full_model


# the tolerances of the full problem's solve (OpenSQP's scaled KKT measures)
SQP_OPTIONS = dict(maxiter=1000, opt_tol=1e-6, feas_tol=1e-6)


def make_solver(N, switch=None, warm_multipliers=True, warm_hessian=False, verbose=True, **options):
    """The hybrid solver with the ALBCD settings of examples/jax/airliner_fleet/airliner_fleet.py."""
    ranges, masses, flights, T_budget = make_fleet(N)
    settings = dict(max_mu=1e3, rho=2.0, tau=0.5, feas_tol=1e-4, opt_tol=1e-4, max_y=1e6, max_outer_iter=100,
                    max_inner_iter=20, sqp_options=SQP_OPTIONS, verbose=verbose)
    settings.update(options)
    opt = HybridALBCD(subproblems=[FlightSubproblem(i, flights, T_budget) for i in range(N)],
                      full_model=make_full_model(flights, T_budget, warm_multipliers, warm_hessian),
                      x0=np.concatenate([initial_guess(fl) for fl in flights]),
                      mu0=np.array([0.01]),
                      switch=switch, **settings)
    opt.fleet = (ranges, masses, flights, T_budget)
    return opt


def compile_models(opt):
    """Run every jitted function once, so that compilation is not timed."""
    N = len(opt.subproblems)
    flights = opt.fleet[2]
    fleet = jax.tree.map(lambda *arrays: jnp.stack(arrays), *flights)
    z = jnp.asarray(opt.x)
    outputs_fn(z[:nvar], flights[0]), jac_fn(z[:nvar], flights[0])
    outputs_all(z.reshape(N, nvar), fleet), jac_all(z.reshape(N, nvar), fleet)
    hess_all(z.reshape(N, nvar), fleet, jnp.zeros((N, nc)))
    # the first evaluation of this solver's full model (and of its warm start) has a one-time cost as well
    opt.full_model(opt.x, opt.y, opt.data)


def report(opt):
    """Print the fleet's fuel, the cost index and each flight at the full problem's solution opt.z."""
    ranges, masses, flights, T_budget = opt.fleet
    N = len(flights)
    blocks = opt.z.reshape(N, nvar)
    fuel_kg = np.array([1e3 * float(outputs_fn(jnp.asarray(v), fl)[-1]) for v, fl in zip(blocks, flights)])
    # OpenSQP's multipliers start with the equalities: each flight's time = tau, then the budget
    cost_index = -opt.sqp.results["pi"][N]
    print(f"Fleet fuel: {fuel_kg.sum():.2f} kg, cost index: {cost_index:.4f} kg/s, "
          f"budget violation: {taus(opt.z).sum() - T_budget:.2e} ks")
    for i in range(N):
        print(f"Flight {i} ({ranges[i] / 1e3:.0f} km, {masses[i]:.0f} kg): fuel {fuel_kg[i]:.1f} kg, "
              f"block time {blocks[i, -1]:.3f} ks")
    reference = os.path.join(FLEET, f"monolithic_solution_N{N}.npz")
    if os.path.exists(reference):
        solution = np.load(reference)
        error = np.linalg.norm((opt.z - solution["z"]) * np.tile(x_scaler, N))
        print(f"Monolithic SLSQP reference: fleet fuel {solution['fuel'].sum():.2f} kg, cost index "
              f"{float(solution['cost_index']):.4f} kg/s, ||z - z*|| {error:.2e}")


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    # switch when ALBCD is within 10x of its own tolerances (the default from switching_study.py)
    opt = make_solver(N, switch=merit_below(10.0))
    compile_models(opt)
    opt.solve()
    report(opt)
