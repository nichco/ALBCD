"""Monolithic solution of the Cessna wing problem with modOpt's OpenSQP.

The same problem as monolithic.py (one optimizer over the twist and thickness control
points, without the coupling copies), with the same objective and constraint scaling and
the same bounds, solved with OpenSQP instead of SLSQP. The design and the wall time at
every major iteration are recorded, and each recorded design is scored with kkt_measure(),
a KKT residual of this monolithic problem that albcd_timing.py also applies to the ALBCD
iterates, so the two methods are measured the same way.

Writes monolithic_opensqp_history.npz, which compare_monolithic_and_albcd.py plots.
"""

import os

# single-threaded BLAS and XLA, as in examples/comp/cessna_comparison.py, so the timings are
# reproducible and both methods run with the same resources
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # modopt and the models work in float64
import jax.numpy as jnp
import modopt as mo
from scipy.optimize import lsq_linear
import warnings
warnings.filterwarnings("ignore")

from models import (aero_model, structures_model, twist_cp0, thickness_cp0, num_cp_twist, num_cp_thickness,
                    min_gauge, sigma_yield_mpa, thickness_max)

HERE = os.path.dirname(os.path.abspath(__file__))


def objective(x):
    CD, _, _ = aero_model(x[:num_cp_twist])
    return CD


def constraints(x):
    _, aero_loads, lift = aero_model(x[:num_cp_twist])
    max_sigma_mpa, min_thickness_mm, weight = structures_model(aero_loads, x[num_cp_twist:])
    return jnp.stack([max_sigma_mpa - sigma_yield_mpa,
                      lift - weight,
                      min_gauge * 1e3 - min_thickness_mm])


# the same problem data and scaling as monolithic.py
x0 = np.concatenate([twist_cp0, thickness_cp0])
cl = np.array([-np.inf, 0.0, -np.inf])  # stress <= yield, lift = weight, thickness >= min gauge
cu = np.array([0.0, 0.0, 0.0])
c_scaler = np.array([1e-2, 1e-2, 1e-1])
o_scaler = 1e2
x_scaler = np.concatenate([10 * np.ones(num_cp_twist), 100 * np.ones(num_cp_thickness)])
xl = np.concatenate([np.full(num_cp_twist, -np.inf), np.zeros(num_cp_thickness)])
xu = np.concatenate([np.full(num_cp_twist, np.inf), np.full(num_cp_thickness, thickness_max)])


# ---------------------------------------------------------------------------------------------
# KKT measure shared with albcd_timing.py
# ---------------------------------------------------------------------------------------------

_grad = jax.jit(jax.grad(objective))
_con = jax.jit(constraints)
_jac = jax.jit(jax.jacrev(constraints))

EQ = np.array([1])     # lift = weight
INEQ = np.array([0, 2])  # stress <= yield, min gauge <= thickness, both as g(x) <= 0


def kkt_measure(x):
    """KKT residual and feasibility of the monolithic problem at x = [twist_cp, thickness_cp].

    Everything is measured in the scaled problem that the optimizers see (x_scaler, c_scaler,
    o_scaler), so the twist, the thickness and the three constraints are on comparable scales.
    The constraints are the lift = weight equality h, the stress and minimum-gauge inequalities,
    and the thickness bounds, all written as g <= 0. The multipliers are not taken from either
    optimizer: they are the least-squares estimate that minimizes

        ||grad f + J_h^T lam_h + J_g^T lam_g||^2 + ||lam_g * g||^2   subject to lam_g >= 0,

    the stationarity and complementarity residuals together, so the same rule applies to the
    OpenSQP and the ALBCD iterates. Returns (optimality, feasibility), where optimality is the
    larger of the max-norm stationarity and complementarity residuals at those multipliers, and
    feasibility is the max-norm violation of the constraints (the bounds are always satisfied).
    """
    x = jnp.asarray(x)
    g_f = o_scaler * np.array(_grad(x)) / x_scaler                     # objective gradient
    c = c_scaler * np.array(_con(x))                                   # scaled constraints
    J = c_scaler[:, None] * np.array(_jac(x)) / x_scaler[None, :]      # their Jacobian

    # thickness bounds in scaled variables: xl - x <= 0 and x - xu <= 0
    xs = x_scaler * np.array(x)
    t = np.arange(num_cp_twist, num_cp_twist + num_cp_thickness)
    I = np.eye(len(xs))[t]
    g_bounds = np.concatenate([x_scaler[t] * xl[t] - xs[t], xs[t] - x_scaler[t] * xu[t]])
    J_bounds = np.vstack([-I, I])

    g = np.concatenate([c[INEQ], g_bounds])  # all inequalities, g <= 0
    A = np.vstack([J[EQ], J[INEQ], J_bounds])  # rows: equality, then the inequalities
    n_eq, n_in = len(EQ), len(g)

    # least-squares multipliers: stationarity rows, then complementarity rows lam_g * g
    M = np.vstack([A.T, np.hstack([np.zeros((n_in, n_eq)), np.diag(g)])])
    rhs = np.concatenate([-g_f, np.zeros(n_in)])
    lb = np.concatenate([np.full(n_eq, -np.inf), np.zeros(n_in)])
    lam = lsq_linear(M, rhs, bounds=(lb, np.full(n_eq + n_in, np.inf)), tol=1e-12).x

    stationarity = np.max(np.abs(g_f + A.T @ lam))
    complementarity = np.max(np.abs(lam[n_eq:] * g))
    feasibility = max(np.max(np.abs(c[EQ])), np.max(c[INEQ]), 0.0)
    return max(stationarity, complementarity), feasibility


def kkt_history(xs):
    """kkt_measure() at every design in xs, as two arrays (optimality, feasibility)."""
    out = np.array([kkt_measure(x) for x in xs])
    return out[:, 0], out[:, 1]


# ---------------------------------------------------------------------------------------------
# OpenSQP solve
# ---------------------------------------------------------------------------------------------

class RecordingOpenSQP(mo.OpenSQP):
    """OpenSQP that keeps the wall time, the design and OpenSQP's own optimality and feasibility
    at every major iteration (iteration 0 is the initial point), even with turn_off_outputs."""

    def solve(self):
        self.history = []
        self.t_start = time.perf_counter()
        return super().solve()

    def update_outputs(self, **kwargs):
        self.history.append((time.perf_counter() - self.t_start,
                             np.array(kwargs["x"]) / x_scaler, kwargs["opt"], kwargs["feas"]))
        super().update_outputs(**kwargs)


def make_problem():
    # JaxProblem jits the objective, the constraints and their derivatives (jax.grad, jax.jacrev)
    return mo.JaxProblem(x0=x0, jax_obj=objective, jax_con=constraints, xl=xl, xu=xu,
                         cl=cl, cu=cu, x_scaler=x_scaler, c_scaler=c_scaler, o_scaler=o_scaler)


SOLVER_OPTIONS = {'maxiter': 300, 'opt_tol': 1e-7, 'feas_tol': 1e-7}


if __name__ == "__main__":
    prob = make_problem()

    # warm-up: one major iteration compiles the jitted functions, so the timed solve
    # below measures the optimization and not JAX's compile time
    mo.OpenSQP(prob, maxiter=1, turn_off_outputs=True).solve()
    kkt_measure(x0)

    optimizer = RecordingOpenSQP(prob, turn_off_outputs=True, **SOLVER_OPTIONS)
    optimizer.solve()
    optimizer.print_results()

    times, xs, opt_native, feas_native = (np.array(a) for a in zip(*optimizer.history))
    optimality, feasibility = kkt_history(xs)

    x_star = xs[-1]
    CD_star = float(objective(jnp.asarray(x_star)))
    print(f'CD: {CD_star:.6e}   wall time {times[-1]:.2f} s   KKT {optimality[-1]:.2e}   '
          f'feasibility {feasibility[-1]:.2e}')

    # one entry per major iteration, starting with the initial point at time 0 (+ setup time)
    np.savez(os.path.join(HERE, 'monolithic_opensqp_history.npz'),
             time=times, optimality=optimality, feasibility=feasibility,
             opt_native=opt_native, feas_native=feas_native, x=xs, CD=CD_star,
             success=bool(optimizer.results['success']))
