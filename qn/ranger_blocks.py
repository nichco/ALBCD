"""Ranger blocks for QuasiNewtonALBCD.

Copied from examples/jax/ranger/ranger.py (read only): tau (line 23), the variable slices (27-30), coupling
(33-35), SQPSubproblem (38-85), AeroSubproblem (88-110), StructSubproblem (113-148) and the x0/data0 construction
(152-158). The copied code is unchanged except:
  - SQPSubproblem subclasses QNSubproblem instead of Subproblem,
  - additions: x_scale on both blocks, the new jitted functions in setup(), linearize(), local_multipliers(),
    tight_tol on the structures block, and the harvesting wrappers of the grad and jac callbacks in sqp()
    (gradients recorded without the proximal term).
The example's models.py is imported through sys.path, not copied. Importing it takes ~30-40 s (STEP geometry).
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "jax" / "ranger"


def import_example_models(example, alias):
    """Import ``example``/models.py through sys.path, registered as ``alias``: both examples name their model
    file models.py, so a plain ``import models`` would return whichever example was imported first."""
    import importlib
    if alias in sys.modules:
        return sys.modules[alias]
    if str(example) not in sys.path:
        sys.path.insert(0, str(example))
    saved = sys.modules.pop("models", None)
    try:
        module = importlib.import_module("models")
    finally:
        sys.modules.pop("models", None)
        if saved is not None:
            sys.modules["models"] = saved
    sys.modules[alias] = module
    return module


_models = import_example_models(EXAMPLE, "ranger_models")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import modopt as mo
from albcd import ALBCD
import warnings
warnings.filterwarnings("ignore")

from qn_subproblem import QNSubproblem
aero_model, structures_model = _models.aero_model, _models.structures_model
twist_cp0, thickness_cp0 = _models.twist_cp0, _models.thickness_cp0
num_cp_twist, num_cp_thickness, num_nodes = _models.num_cp_twist, _models.num_cp_thickness, _models.num_nodes
min_gauge, sigma_yield_mpa, thickness_max = _models.min_gauge, _models.sigma_yield_mpa, _models.thickness_max

# proximal coefficient
tau = 0.0

# x = [twist_cp, weight_copy | thickness_cp, aero_loads_copy], the aero block's variables then the struct block's.
# aero_loads_copy is [drag_0..drag_{num_nodes-1}, lift_0..lift_{num_nodes-1}]
TWIST = slice(0, num_cp_twist)
WEIGHT = num_cp_twist
THICKNESS = slice(WEIGHT + 1, WEIGHT + 1 + num_cp_thickness)
LOADS = slice(THICKNESS.stop, THICKNESS.stop + 2 * num_nodes)


def coupling(aero_loads, aero_loads_copy, weight, weight_copy):
    """the scaled drag and lift residual at each node, then the weight residual"""
    return jnp.concatenate([1e-3 * (aero_loads - aero_loads_copy), 1e-4 * jnp.reshape(weight - weight_copy, 1)])


class SQPSubproblem(QNSubproblem):
    """
    OpenSQP stops on the KKT residual of the subproblem rather than on the change in the objective, so a block
    solved from a nearby warm start still reaches its tolerance instead of stopping after the first iteration.

    The functions are jitted once, in setup(), with the inputs that change between solves as arguments,
    so every solve reuses the same compiled VLM and frame models.
    """

    def setup(self):
        self.obj, self.grad = jax.jit(self.objective), jax.jit(jax.grad(self.objective))
        self.con, self.jac = jax.jit(self.local_constraints), jax.jit(jax.jacrev(self.local_constraints))
        # addition: this block's first-order information for linearize(), in one jitted function
        self.lin_fn = jax.jit(self.linear_pieces)

    def args(self, x, y, mu, data):
        return tuple(jnp.asarray(a) for a in (self.other(x), y, mu, data[self.DATA]))

    def sqp(self, x, y, mu, data, cl=0.0, cu=0.0, xl=None, xu=None):
        """This block's variables and constraint multipliers at the solution of its subproblem.

        OpenSQP minimizes objective(v, *args) + tau/2 ||X_SCALER * (v - v0)||^2, where the proximal term penalizes
        the distance from the block's current values v0. It is measured in the scaled variables OpenSQP sees, so
        the copies of the loads and weight (O(1e2-1e3) N) do not dominate the twist and thickness.
        """
        args = self.args(x, y, mu, data)
        v0 = self.decompose(x)
        w = tau * self.X_SCALER**2  # proximal weight on each unscaled variable
        self.curvature, self._pending = [], {}  # addition: harvested (v, grad_obj, Jh), when self.harvest
        # jnp.asarray: modopt passes numpy arrays, which jax.jit would compile for separately
        prob = mo.ProblemLite(x0=v0,
                              obj=lambda v: np.float64(self.obj(jnp.asarray(v), *args)) + 0.5 * np.sum(w * (v - v0)**2),
                              grad=lambda v: self._record(v, "g", np.array(self.grad(jnp.asarray(v), *args))) + w * (v - v0),
                              con=lambda v: np.array(self.con(jnp.asarray(v))),
                              jac=lambda v: self._record(v, "J", np.array(self.jac(jnp.asarray(v)))),
                              xl=xl, xu=xu, x_scaler=self.X_SCALER, cl=cl, cu=cu, c_scaler=self.C_SCALER)
        # opt_tol 1e-8: tighter tolerances are out of reach for some solves, which then run to maxiter
        optimizer = mo.OpenSQP(prob, maxiter=50, opt_tol=1e-8, feas_tol=1e-9, turn_off_outputs=True)
        optimizer.solve()
        # OpenSQP works in scaled variables and constraints, and orders its multipliers as the equality
        # constraints, the variable bounds, then the inequality constraints; the bound multipliers are dropped
        pi = np.asarray(optimizer.results['pi'])
        num_eq = np.sum(np.asarray(cl) == np.asarray(cu))
        num_bounds = sum(0 if b is None else np.sum(np.isfinite(b)) for b in (xl, xu))
        multipliers = np.concatenate([pi[:num_eq], pi[num_eq + num_bounds:]])
        self._optimizer_results = optimizer.results  # addition: for the tests
        return optimizer.results['x'] / self.X_SCALER, multipliers * self.C_SCALER

    def gradient(self, x, y, mu, data):
        """KKT stationarity residual with OpenSQP's multipliers for the local constraint"""
        v = jnp.asarray(self.decompose(x))
        return np.array(self.grad(v, *self.args(x, y, mu, data)) - self.jac(v).T @ self.multipliers)

    # ------------------------------------------------------------------------------------------------------
    # additions for QuasiNewtonALBCD
    # ------------------------------------------------------------------------------------------------------

    def _record(self, v, kind, value):
        """Harvesting wrapper of the grad and jac callbacks: pairs the AL gradient (without the proximal term)
        and the constraint Jacobian evaluated at the same v, in the order the pairs complete."""
        if self.harvest:
            key = np.asarray(v, dtype=float).tobytes()
            entry = self._pending.setdefault(key, {"v": np.array(v, dtype=float, copy=True)})
            entry[kind] = np.array(value, copy=True)
            if "g" in entry and "J" in entry:
                self.curvature.append((entry["v"], entry["g"], entry["J"]))
                del self._pending[key]
        return value

    def linearize(self, x, data):
        v = jnp.asarray(self.decompose(x), dtype=float)
        pieces = {k: np.asarray(a) for k, a in self.lin_fn(v).items()}
        outputs = {k: pieces.pop(k) for k in self.OUTPUTS}
        for k in ("CD", "weight"):
            if k in outputs:
                outputs[k] = float(outputs[k])
        pieces.update(cl=np.atleast_1d(self.CL).astype(float), cu=np.atleast_1d(self.CU).astype(float),
                      xl=np.asarray(self.XL, dtype=float), xu=np.asarray(self.XU, dtype=float), outputs=outputs)
        return pieces

    def local_multipliers(self):
        return -np.asarray(self.multipliers, dtype=float)


class AeroSubproblem(SQPSubproblem):
    """Owns [twist_cp, weight_copy]: minimizes drag subject to lift = weight_copy."""

    DATA = "weight"
    X_SCALER = np.concatenate([np.full(num_cp_twist, 10.0), [1e-3]])
    C_SCALER = 1e-2
    x_scale = X_SCALER                                  # addition
    CL, CU = 0.0, 0.0                                   # addition: lift - weight_copy = 0
    XL, XU = np.full(num_cp_twist + 1, -np.inf), np.full(num_cp_twist + 1, np.inf)  # addition
    OUTPUTS = ("aero_loads", "CD")                      # addition

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
        outputs["phi"] = np.array(coupling(aero_loads, x[LOADS], data["weight"], v[-1]))

    # addition
    @staticmethod
    def c_aero(v):
        _, aero_loads, _ = aero_model(v[:-1])
        return jnp.concatenate([1e-3 * aero_loads, jnp.reshape(-1e-4 * v[-1], 1)])

    def linear_pieces(self, v):
        """grad of 1e2 CD, c_aero and its Jacobian, the local constraint and its Jacobian, and the outputs.

        One forward-mode Jacobian (9 inputs) of [1e2 CD, c_aero, lift - weight_copy] gives all three derivatives;
        reverse mode over c_aero's 83 outputs costs ~0.6 s per call through the VLM."""
        def stacked(u):
            CD, aero_loads, lift = aero_model(u[:-1])
            F = jnp.concatenate([jnp.reshape(1e2 * CD, 1), self.c_aero(u), jnp.reshape(lift - u[-1], 1)])
            return F, (F, aero_loads, CD)
        Jfull, (F, aero_loads, CD) = jax.jacfwd(stacked, has_aux=True)(v)
        m = 2 * num_nodes + 1
        return dict(grad_f=Jfull[0], c=F[1:1 + m], J=Jfull[1:1 + m], h=F[1 + m:], Jh=Jfull[1 + m:],
                    aero_loads=aero_loads, CD=CD)


class StructSubproblem(SQPSubproblem):
    """Owns [thickness_cp, aero_loads_copy]: sizes the spar for stress and minimum gauge."""

    DATA = "aero_loads"
    X_SCALER = np.concatenate([np.full(num_cp_thickness, 1e2), np.full(2 * num_nodes, 1e-2)])
    C_SCALER = np.array([1e-2])
    # thickness bounds (m): the minimum gauge, active over most of the span at the solution, and the outer
    # radius of the tube spar, which keeps OpenSQP's iterates where the tube model is physical. The thickness
    # is a B-spline, a convex combination of its control points, so the bounds hold for every element.
    XL = np.concatenate([np.full(num_cp_thickness, min_gauge), np.full(2 * num_nodes, -np.inf)])
    XU = np.concatenate([np.full(num_cp_thickness, thickness_max), np.full(2 * num_nodes, np.inf)])
    x_scale = X_SCALER                                  # addition
    CL, CU = -np.inf, 0.0                               # addition: max_sigma - yield <= 0
    OUTPUTS = ("weight",)                               # addition
    tight_tol = 1e-8                                    # addition: the stress row is active if |h| <= 1e-8 (§1.3)

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
        outputs["phi"] = np.array(coupling(data["aero_loads"], v[num_cp_thickness:], weight, x[WEIGHT]))
        self._last_v = np.array(v, dtype=float)  # addition: the solved point, for local_multipliers()

    def gradient(self, x, y, mu, data):
        # on a bound, only the part of the residual that pushes the variable out of bounds is a violation
        resid = super().gradient(x, y, mu, data)
        v = self.decompose(x)
        resid = np.where(v <= self.XL + 1e-8, np.minimum(resid, 0.0), resid)
        return np.where(v >= self.XU - 1e-8, np.maximum(resid, 0.0), resid)

    # additions
    @staticmethod
    def c_struct(v):
        _, weight = structures_model(v[num_cp_thickness:], v[:num_cp_thickness])
        return jnp.concatenate([-1e-3 * v[num_cp_thickness:], jnp.reshape(1e-4 * weight, 1)])

    def linear_pieces(self, v):
        max_sigma_mpa, weight = structures_model(v[num_cp_thickness:], v[:num_cp_thickness])
        return dict(grad_f=jnp.zeros_like(v), c=self.c_struct(v), J=jax.jacrev(self.c_struct)(v),
                    h=jnp.reshape(max_sigma_mpa - sigma_yield_mpa, 1), Jh=jax.jacrev(self.local_constraints)(v),
                    weight=weight)

    def local_multipliers(self):
        """kappa = -self.multipliers, active if |kappa| > act_tol max(1, |kappa|) or the stress is within 1e-8 of
        yield; 0 otherwise."""
        kappa = -np.asarray(self.multipliers, dtype=float)
        v = self._last_v
        h = np.asarray(self.con(jnp.asarray(v)))
        active = (np.abs(kappa) > self.act_tol * np.maximum(1.0, np.abs(kappa))) | (np.abs(h) <= self.tight_tol)
        return np.where(active, kappa, 0.0)


def initial_point():
    """x0 and data0 of ranger.py: both models evaluated once at the initial design."""
    CD_init, aero_loads_init, _ = aero_model(twist_cp0)
    _, weight_init = structures_model(aero_loads_init, thickness_cp0)
    x0 = np.concatenate([twist_cp0, [float(weight_init)], thickness_cp0, np.array(aero_loads_init)])
    data0 = {"weight": float(weight_init), "aero_loads": np.array(aero_loads_init), "CD": float(CD_init)}
    return x0, data0


def make_solver(cls=ALBCD, **options):
    """A solver for ranger.py's problem with its settings (options override them)."""
    x0, data0 = initial_point()
    settings = dict(mu0=np.full(2 * num_nodes + 1, 10.0), max_mu=1e6, rho=1.2, tau=0.5, feas_tol=3e-4, opt_tol=1e-4,
                    max_outer_iter=40, max_inner_iter=20)
    settings.update(options)
    settings["mu0"] = np.array(settings["mu0"], dtype=float)  # a fresh array: ALBCD keeps the caller's mu0
    return cls(subproblems=[AeroSubproblem(slice(0, THICKNESS.start)), StructSubproblem(slice(THICKNESS.start, None))],
               x0=x0, data0=data0, **settings)


def compile_models(opt):
    """Compile every jitted block function (solve callbacks and linearize) before a timed solve."""
    x, y, mu, data = opt.x, np.zeros_like(opt.mu), opt.mu, opt.data
    for sp in opt.subproblems:
        v = jnp.asarray(sp.decompose(x))
        args = sp.args(x, y, mu, data)
        jax.block_until_ready((sp.obj(v, *args), sp.grad(v, *args), sp.con(v), sp.jac(v), sp.lin_fn(v)))
