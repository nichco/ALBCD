"""The ranger aerostructural problem (examples/jax/ranger) with the hybrid ALBCD interface.

The blocks are those of examples/jax/ranger/ranger.py: an aero block that owns [twist_cp, weight_copy]
and a structures block that owns [thickness_cp, aero_loads_copy], coupled by the consistency of the
copies. The full model is that of examples/jax/ranger/monolithic.py: minimize CD over z = [twist_cp,
thickness_cp] subject to stress <= yield and lift = weight, with the aero loads passed straight from the
aero model to the structures model, so it has no copies and no coupling constraints.

Run this file to solve the problem with the hybrid solver. make_solver() builds the solver for
switching_study.py.
"""

import os
import sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import modopt as mo
import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RANGER = os.path.join(HERE, "..", "examples", "jax", "ranger")
sys.path.insert(0, RANGER)  # models.py and ranger_geometry_parameterization.py

from albcd import Subproblem
from hybrid_albcd import HybridALBCD, init_multipliers, merit_below, positive_definite
from models import (aero_model, structures_model, twist_cp0, thickness_cp0, num_cp_twist, num_cp_thickness,
                    num_nodes, bspline_thickness, min_gauge, sigma_yield_mpa, thickness_max)

# x = [twist_cp, weight_copy | thickness_cp, aero_loads_copy], the aero block's variables then the struct block's
TWIST = slice(0, num_cp_twist)
WEIGHT = num_cp_twist
THICKNESS = slice(WEIGHT + 1, WEIGHT + 1 + num_cp_thickness)
LOADS = slice(THICKNESS.stop, THICKNESS.stop + 2 * num_nodes)


# ---------------------------------------------------------------------------------------------
# subproblems: as in examples/jax/ranger/ranger.py (without the proximal term, which is zero there)
# ---------------------------------------------------------------------------------------------

def coupling(aero_loads, aero_loads_copy, weight, weight_copy):
    """the scaled drag and lift residual at each node, then the weight residual"""
    return jnp.concatenate([1e-3 * (aero_loads - aero_loads_copy), 1e-4 * jnp.reshape(weight - weight_copy, 1)])


class SQPSubproblem(Subproblem):
    """A block solved with OpenSQP; its functions are jitted once, in setup()."""

    def setup(self):
        self.obj, self.grad = jax.jit(self.objective), jax.jit(jax.grad(self.objective))
        self.con, self.jac = jax.jit(self.local_constraints), jax.jit(jax.jacrev(self.local_constraints))

    def args(self, x, y, mu, data):
        return tuple(jnp.asarray(a) for a in (self.other(x), y, mu, data[self.DATA]))

    def sqp(self, x, y, mu, data, cl=0.0, cu=0.0, xl=None, xu=None):
        """This block's variables and constraint multipliers at the solution of its subproblem."""
        args = self.args(x, y, mu, data)
        prob = mo.ProblemLite(x0=self.decompose(x),
                              obj=lambda v: np.float64(self.obj(jnp.asarray(v), *args)),
                              grad=lambda v: np.array(self.grad(jnp.asarray(v), *args)),
                              con=lambda v: np.array(self.con(jnp.asarray(v))),
                              jac=lambda v: np.array(self.jac(jnp.asarray(v))),
                              xl=xl, xu=xu, x_scaler=self.X_SCALER, cl=cl, cu=cu, c_scaler=self.C_SCALER)
        optimizer = mo.OpenSQP(prob, maxiter=50, opt_tol=1e-8, feas_tol=1e-9, turn_off_outputs=True)
        optimizer.solve()
        # OpenSQP orders its multipliers as the equality constraints, the variable bounds, then the
        # inequality constraints; the bound multipliers are dropped
        pi = np.asarray(optimizer.results['pi'])
        num_eq = np.sum(np.asarray(cl) == np.asarray(cu))
        num_bounds = sum(0 if b is None else np.sum(np.isfinite(b)) for b in (xl, xu))
        multipliers = np.concatenate([pi[:num_eq], pi[num_eq + num_bounds:]])
        return optimizer.results['x'] / self.X_SCALER, multipliers * self.C_SCALER

    def gradient(self, x, y, mu, data):
        """KKT stationarity residual with OpenSQP's multipliers for the local constraint"""
        v = jnp.asarray(self.decompose(x))
        return np.array(self.grad(v, *self.args(x, y, mu, data)) - self.jac(v).T @ self.multipliers)


class AeroSubproblem(SQPSubproblem):
    """Owns [twist_cp, weight_copy]: minimizes drag subject to lift = weight_copy."""

    DATA = "weight"
    X_SCALER = np.concatenate([np.full(num_cp_twist, 10.0), [1e-3]])
    C_SCALER = 1e-2

    def objective(self, v, other, y, mu, weight):
        CD, aero_loads, _ = aero_model(v[:-1])
        c = coupling(aero_loads, other[num_cp_thickness:], weight, v[-1])
        return 1e2 * CD + jnp.sum(y * c) + 0.5 * jnp.sum(mu * c**2)

    def local_constraints(self, v):
        _, _, lift = aero_model(v[:-1])
        return jnp.reshape(lift - v[-1], 1)

    def solve(self, x, y, mu, data, outputs):
        v, self.multipliers = self.sqp(x, y, mu, data)
        CD, aero_loads, _ = aero_model(v[:-1])
        outputs["x"] = self.recompose(x, v)
        outputs["aero_loads"] = np.array(aero_loads)
        outputs["CD"] = float(CD)
        outputs["lift_multiplier"] = self.multipliers.copy()  # shared so the full model can warm start
        outputs["phi"] = np.array(coupling(aero_loads, x[LOADS], data["weight"], v[-1]))


class StructSubproblem(SQPSubproblem):
    """Owns [thickness_cp, aero_loads_copy]: sizes the spar for stress and minimum gauge."""

    DATA = "aero_loads"
    X_SCALER = np.concatenate([np.full(num_cp_thickness, 1e2), np.full(2 * num_nodes, 1e-2)])
    C_SCALER = np.array([1e-2])
    XL = np.concatenate([np.full(num_cp_thickness, min_gauge), np.full(2 * num_nodes, -np.inf)])
    XU = np.concatenate([np.full(num_cp_thickness, thickness_max), np.full(2 * num_nodes, np.inf)])

    def objective(self, v, other, y, mu, aero_loads):
        _, weight = structures_model(v[num_cp_thickness:], v[:num_cp_thickness])
        c = coupling(aero_loads, v[num_cp_thickness:], weight, other[num_cp_twist])
        return jnp.sum(y * c) + 0.5 * jnp.sum(mu * c**2)  # the drag term is constant in this block

    def local_constraints(self, v):
        max_sigma_mpa, _ = structures_model(v[num_cp_thickness:], v[:num_cp_thickness])
        return jnp.reshape(max_sigma_mpa - sigma_yield_mpa, 1)

    def solve(self, x, y, mu, data, outputs):
        v, multipliers = self.sqp(x, y, mu, data, cl=np.full(1, -np.inf), cu=np.zeros(1), xl=self.XL, xu=self.XU)
        # OpenSQP treats c(v) <= cu as cu - c(v) >= 0, so its multipliers have the opposite sign
        self.multipliers = -multipliers
        _, weight = structures_model(v[num_cp_thickness:], v[:num_cp_thickness])
        outputs["x"] = self.recompose(x, v)
        outputs["weight"] = float(weight)
        outputs["stress_multiplier"] = self.multipliers.copy()  # shared so the full model can warm start
        outputs["phi"] = np.array(coupling(data["aero_loads"], v[num_cp_thickness:], weight, x[WEIGHT]))

    def gradient(self, x, y, mu, data):
        # on a bound, only the part of the residual that pushes the variable out of bounds is a violation
        resid = super().gradient(x, y, mu, data)
        v = self.decompose(x)
        resid = np.where(v <= self.XL + 1e-8, np.minimum(resid, 0.0), resid)
        return np.where(v >= self.XU - 1e-8, np.maximum(resid, 0.0), resid)


# ---------------------------------------------------------------------------------------------
# full model: as in examples/jax/ranger/monolithic.py, z = [twist_cp, thickness_cp]
# ---------------------------------------------------------------------------------------------

Z_TWIST = slice(0, num_cp_twist)
Z_THICKNESS = slice(num_cp_twist, num_cp_twist + num_cp_thickness)
Z_SCALER = np.concatenate([np.full(num_cp_twist, 10.0), np.full(num_cp_thickness, 100.0)])
ZL = np.concatenate([np.full(num_cp_twist, -np.inf), np.full(num_cp_thickness, min_gauge)])
ZU = np.concatenate([np.full(num_cp_twist, np.inf), np.full(num_cp_thickness, thickness_max)])
O_SCALER, C_SCALER = 1e2, np.array([1e-2, 1e-2])


def full_objective(z):
    return aero_model(z[Z_TWIST])[0]


def full_constraints(z):
    """stress - yield (<= 0) and lift - weight (= 0)"""
    _, aero_loads, lift = aero_model(z[Z_TWIST])
    max_sigma_mpa, weight = structures_model(aero_loads, z[Z_THICKNESS])
    return jnp.stack([max_sigma_mpa - sigma_yield_mpa, lift - weight])


_obj, _grad = jax.jit(full_objective), jax.jit(jax.grad(full_objective))
_con, _jac = jax.jit(full_constraints), jax.jit(jax.jacrev(full_constraints))
# Hessian of the scaled problem's Lagrangian, O_SCALER f - lam^T (C_SCALER c), in the unscaled variables
_hess = jax.jit(jax.hessian(lambda z, lam: O_SCALER * full_objective(z) - lam @ (C_SCALER * full_constraints(z))))


def full_problem(z0):
    return mo.ProblemLite(x0=z0,
                          obj=lambda z: np.float64(_obj(jnp.asarray(z))),
                          grad=lambda z: np.array(_grad(jnp.asarray(z))),
                          con=lambda z: np.array(_con(jnp.asarray(z))),
                          jac=lambda z: np.array(_jac(jnp.asarray(z))),
                          xl=ZL, xu=ZU, cl=np.array([-np.inf, 0.0]), cu=np.zeros(2),
                          x_scaler=Z_SCALER, o_scaler=O_SCALER, c_scaler=C_SCALER)


def make_full_model(warm_multipliers=True, warm_hessian=False):
    """The full model in the form HybridALBCD takes: (x, y, data) -> (problem, init_multipliers).

    The copies are dropped: the full problem starts from the twist and thickness of the ALBCD iterate.
    Its two multipliers are warm started from the blocks' multipliers of the same constraints (the stress
    constraint of the structures block and lift = weight_copy of the aero block), which the blocks share
    through ``data``. Both blocks minimize 1e2 CD plus coupling terms, the scaled objective of the full
    problem, so only the constraint scaling has to be applied.

    With ``warm_hessian``, OpenSQP's BFGS matrix starts from the exact Hessian of the Lagrangian at the ALBCD
    iterate, with those multipliers, made positive definite, instead of the identity.
    """
    def hessian(z, lam):
        """Exact Hessian of the scaled problem's Lagrangian in the scaled variables, made positive definite."""
        return positive_definite(np.asarray(_hess(jnp.asarray(z), jnp.asarray(lam))) / np.outer(Z_SCALER, Z_SCALER))

    def full_model(x, y, data):
        problem = full_problem(np.concatenate([x[TWIST], x[THICKNESS]]))
        started = "stress_multiplier" in data  # False at the initial point, before any block solve
        if not warm_multipliers or not (started or warm_hessian):
            return problem
        # at the initial point lam = 0, and the Hessian below is that of the objective
        lam = np.array([data["stress_multiplier"][0], data["lift_multiplier"][0]]) / C_SCALER if started else np.zeros(2)
        if not warm_hessian:
            return problem, init_multipliers(problem, lam)
        return problem, init_multipliers(problem, lam), hessian(np.concatenate([x[TWIST], x[THICKNESS]]), lam)
    full_model.hessian = hessian  # for a mid-run Hessian injection (RecordingOpenSQP.hessian)
    return full_model


# the tolerances of the full problem's solve (OpenSQP's scaled KKT measures)
SQP_OPTIONS = dict(maxiter=300, opt_tol=1e-6, feas_tol=1e-7)


def make_solver(switch=None, warm_multipliers=True, warm_hessian=False, verbose=True, **options):
    """The hybrid solver with the ALBCD settings of examples/jax/ranger/ranger.py."""
    CD_init, aero_loads_init, _ = aero_model(twist_cp0)
    _, weight_init = structures_model(aero_loads_init, thickness_cp0)
    settings = dict(max_mu=1e6, rho=1.2, tau=0.5, feas_tol=3e-4, opt_tol=1e-4, max_outer_iter=40,
                    max_inner_iter=20, sqp_options=SQP_OPTIONS, verbose=verbose)
    settings.update(options)
    return HybridALBCD(
        subproblems=[AeroSubproblem(slice(0, THICKNESS.start)), StructSubproblem(slice(THICKNESS.start, None))],
        full_model=make_full_model(warm_multipliers, warm_hessian),
        x0=np.concatenate([twist_cp0, [float(weight_init)], thickness_cp0, np.array(aero_loads_init)]),
        mu0=np.full(2 * num_nodes + 1, 10.0),
        data0={"weight": float(weight_init), "aero_loads": np.array(aero_loads_init), "CD": float(CD_init)},
        switch=switch, **settings)


def compile_models(opt):
    """Run every jitted function once, so that compilation is not timed."""
    z0 = jnp.asarray(np.concatenate([opt.x[TWIST], opt.x[THICKNESS]]))
    _obj(z0), _grad(z0), _con(z0), _jac(z0), _hess(z0, jnp.zeros(2))
    # the blocks also call the models outside jit, which compiles each JAX primitive on first use
    structures_model(aero_model(opt.x[TWIST])[1], opt.x[THICKNESS])
    # the first evaluation of this solver's full model (and of its warm start) has a one-time cost as well
    opt.full_model(opt.x, opt.y, opt.data)
    for sp in opt.subproblems:
        v = jnp.asarray(sp.decompose(opt.x))
        args = sp.args(opt.x, opt.y, opt.mu, opt.data)
        sp.obj(v, *args), sp.grad(v, *args), sp.con(v), sp.jac(v)


def report(opt):
    """Print the design of the full problem's solution opt.z."""
    twist_cp, thickness_cp = opt.z[Z_TWIST], opt.z[Z_THICKNESS]
    CD, aero_loads, lift = aero_model(twist_cp)
    max_sigma_mpa, weight = structures_model(aero_loads, thickness_cp)
    print(f"CD: {float(CD):.7f}")
    print(f"Lift (N): {float(lift):.3f}   Weight (N): {float(weight):.3f}")
    print(f"Max von Mises stress, (KS) (MPa): {float(max_sigma_mpa):.3f}")
    print(f"Min wall thickness (mm): {float(np.min(bspline_thickness @ thickness_cp)) * 1e3:.4f}")
    reference = os.path.join(RANGER, "monolithic_solution.npz")
    if os.path.exists(reference):
        solution = np.load(reference)
        error = np.linalg.norm(opt.z - np.concatenate([solution["twist_cp"], solution["thickness_cp"]]))
        print(f"CD (monolithic SLSQP reference): {float(solution['CD']):.7f}   ||z - z*||: {error:.2e}")


if __name__ == "__main__":
    # switch when ALBCD is within 10x of its own tolerances (the default from switching_study.py)
    opt = make_solver(switch=merit_below(10.0))
    compile_models(opt)
    opt.solve()
    report(opt)
