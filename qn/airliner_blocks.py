"""Airliner fleet blocks for QuasiNewtonALBCD.

Copied from examples/jax/airliner_fleet/airliner_fleet.py (read only): the fleet setup (lines 43-51), taus (54),
the compiled functions (59-60), kkt_rows (63-67) and FlightSubproblem (70-120). The copied code is unchanged
except:
  - FlightSubproblem subclasses QNSubproblem instead of Subproblem, and takes (i, flights, T_budget) in place of
    the example's globals (budget() reads self.T_budget),
  - additions: x_scale, grad_phi(), local_multipliers(), and the harvesting wrappers of the grad and jac
    callbacks passed to mo.ProblemLite.
The example's models.py is imported through sys.path, not copied.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path

from example_import import import_example_models

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "jax" / "airliner_fleet"


_models = import_example_models(EXAMPLE, "airliner_fleet_models")

import numpy as np
import jax
import jax.numpy as jnp
import modopt as mo
from scipy.stats import qmc
from albcd import ALBCD
import warnings
warnings.filterwarnings("ignore")

from qn_subproblem import QNSubproblem
nvar, cl, cu, xl, xu, x_scaler = _models.nvar, _models.cl, _models.cu, _models.xl, _models.xu, _models.x_scaler
setup_flight, flight_outputs = _models.setup_flight, _models.flight_outputs
initial_guess, last_call_cache = _models.initial_guess, _models.last_call_cache


def make_fleet(N):
    """Ranges (m), takeoff masses (kg), flight data and block-time budget (ks) of airliner_fleet.py."""
    ranges, masses = qmc.scale(qmc.LatinHypercube(d=2, seed=0).random(N), [2000e3, 22000.0], [5000e3, 28000.0]).T
    flights = [setup_flight(rf, m0) for rf, m0 in zip(ranges, masses)]
    T_budget = 0.9 * np.sum(0.276 + 5.468e-6 * ranges)
    return ranges, masses, flights, T_budget


# global x = [v_0, v_1, ..., v_{N-1}], where v_i = [h coefficients, v coefficients, tau] of flight i
taus = lambda x: x[nvar - 1::nvar]

y_history = []  # the multiplier on the budget (the cost index) at each sweep (as in the example)

# compiled once and shared by every block (the flight's data is an argument)
outputs_fn = jax.jit(flight_outputs)
jac_fn = jax.jit(jax.jacfwd(flight_outputs))


def kkt_rows(J):
    """The rows of the local constraint Jacobian J in the order and sign of SLSQP's multipliers:
    equalities, then lower bounds, then upper bounds (negated), so that grad f = rows^T multipliers."""
    eq = cl == cu
    return np.concatenate((J[eq], J[(cl != -np.inf) & ~eq], -J[(cu != np.inf) & ~eq]))


class FlightSubproblem(QNSubproblem):
    """Flight i: owns its trajectory and its block-time allocation tau_i."""

    x_scale = x_scaler

    def __init__(self, i, flights, T_budget):
        fl = flights[i]
        self.i = i
        self.T_budget = T_budget
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
        if self.index.start == 0:  # the first block of a sweep
            y_history.append(float(y[0]))
        self.curvature = []  # addition: harvested (v, grad_obj, Jh), when self.harvest
        prob = mo.ProblemLite(x0=np.array(self.decompose(x)),
                              obj=lambda v: self.objective(v, x, y, mu),
                              grad=lambda v: self._record(v, self.objective_gradient(v, x, y, mu)),
                              con=lambda v: self.outputs(v)[:-1],
                              jac=lambda v: self.outputs_jac(v)[:-1],
                              xl=xl, xu=xu, cl=cl, cu=cu, x_scaler=x_scaler)
        optimizer = mo.SLSQP(prob, solver_options={"maxiter": 500, "ftol": 1e-9}, turn_off_outputs=True)
        optimizer.solve()
        v_new = optimizer.results["x"] / x_scaler

        # cache SLSQP's multipliers and the matching rows of the local constraints' Jacobian for gradient()
        self._multipliers = np.asarray(optimizer.results["multipliers"])
        self._rows = kkt_rows(self.outputs_jac(v_new)[:-1])

        outputs["x"] = self.recompose(x, v_new)
        outputs["phi"] = np.array([taus(outputs["x"]).sum() - self.T_budget])

    def gradient(self, x, y, mu, data) -> np.ndarray:
        v = self.decompose(x)

        # Lagrangian gradient (with SLSQP's multipliers for the local constraints) projected onto
        # the box bounds: zero iff v is a KKT point of this block's subproblem
        grad_L = self.objective_gradient(v, x, y, mu) - self._rows.T @ self._multipliers
        return v - np.clip(v - grad_L, xl, xu)

    # ------------------------------------------------------------------------------------------------------
    # additions for QuasiNewtonALBCD
    # ------------------------------------------------------------------------------------------------------

    def _record(self, v, grad):
        """Harvesting wrapper of the grad callback: the AL gradient at v and the (cached) constraint Jacobian."""
        if self.harvest:
            self.curvature.append((np.array(v, copy=True), np.array(grad, copy=True),
                                   np.array(self.outputs_jac(v)[:-1], copy=True)))
        return grad

    def grad_phi(self, x, data):
        v = np.asarray(self.decompose(x), dtype=float)
        out = self.outputs(v)
        jac = self.outputs_jac(v)
        J = np.zeros((1, nvar))
        J[0, -1] = 1.0
        return dict(grad_f=jac[-1].copy(),
                    c=np.array([v[-1] - (self.T_budget if self.i == 0 else 0.0)]),
                    J=J,
                    h=out[:-1].copy(), Jh=jac[:-1].copy(),
                    cl=cl.copy(), cu=cu.copy(), xl=xl.copy(), xu=xu.copy(),
                    outputs={})

    def local_multipliers(self):
        """kappa from SLSQP's multipliers m (rows of kkt_rows): kappa_eq = -m_eq, kappa_lower = -m_lower,
        kappa_upper = +m_upper; inequality rows with |m| <= act_tol max(1, ||m||_inf) are inactive (0)."""
        m = self._multipliers
        eq = cl == cu
        lower = (cl != -np.inf) & ~eq
        upper = (cu != np.inf) & ~eq
        n_eq, n_lo = int(eq.sum()), int(lower.sum())
        m_eq, m_lo, m_up = m[:n_eq], m[n_eq:n_eq + n_lo], m[n_eq + n_lo:]
        thresh = self.act_tol * max(1.0, np.max(np.abs(m), initial=0.0))
        kappa = np.zeros(len(cl))
        kappa[eq] = -m_eq
        kappa[lower] += np.where(np.abs(m_lo) > thresh, -m_lo, 0.0)
        kappa[upper] += np.where(np.abs(m_up) > thresh, m_up, 0.0)
        return kappa


def make_solver(N, cls=ALBCD, **options):
    """The fleet of N flights and a solver with airliner_fleet.py's settings (options override them)."""
    ranges, masses, flights, T_budget = make_fleet(N)
    x0 = np.concatenate([initial_guess(fl) for fl in flights])
    settings = dict(mu0=np.array([0.01]), max_mu=1e3, rho=2.0, tau=0.5, feas_tol=1e-4, opt_tol=1e-4, max_y=1e6,
                    max_outer_iter=100, max_inner_iter=20)
    settings.update(options)
    settings["mu0"] = np.array(settings["mu0"], dtype=float)  # a fresh array: ALBCD keeps the caller's mu0
    opt = cls(subproblems=[FlightSubproblem(i, flights, T_budget) for i in range(N)], x0=x0, **settings)
    opt.fleet = dict(N=N, ranges=ranges, masses=masses, flights=flights, T_budget=T_budget)
    return opt


def compile_models(opt):
    """Compile the jitted flight functions before a timed solve."""
    x = opt.x[:nvar]
    fl = opt.fleet["flights"][0]
    outputs_fn(jnp.asarray(x), fl).block_until_ready()
    jac_fn(jnp.asarray(x), fl).block_until_ready()
