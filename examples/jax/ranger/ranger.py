import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import modopt as mo
import matplotlib.pyplot as plt
import pyvista as pv
from albcd import ALBCD, Subproblem
import warnings
warnings.filterwarnings("ignore")
from jeometry.plotting import set_view
from jvlm import panel_pressure
from jvlm.viz import to_pyvista_mesh
from models import (aero_model, structures_model, twist_cp0, thickness_cp0, num_cp_twist, num_cp_thickness,
                    num_nodes, bspline_twist, bspline_thickness, min_gauge, sigma_yield_mpa, y, y_elem, q, v_inf,
                    rho_atm, vlm_mesh, solve_aero, thickness_max)
from ranger_geometry_parameterization import twist_ffd

HERE = os.path.dirname(os.path.abspath(__file__))

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


class SQPSubproblem(Subproblem):
    """
    OpenSQP stops on the KKT residual of the subproblem rather than on the change in the objective, so a block
    solved from a nearby warm start still reaches its tolerance instead of stopping after the first iteration.

    The functions are jitted once, in setup(), with the inputs that change between solves as arguments,
    so every solve reuses the same compiled VLM and frame models.
    """

    def setup(self):
        self.obj, self.grad = jax.jit(self.objective), jax.jit(jax.grad(self.objective))
        self.con, self.jac = jax.jit(self.local_constraints), jax.jit(jax.jacrev(self.local_constraints))

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
        # jnp.asarray: modopt passes numpy arrays, which jax.jit would compile for separately
        prob = mo.ProblemLite(x0=v0,
                              obj=lambda v: np.float64(self.obj(jnp.asarray(v), *args)) + 0.5 * np.sum(w * (v - v0)**2),
                              grad=lambda v: np.array(self.grad(jnp.asarray(v), *args)) + w * (v - v0),
                              con=lambda v: np.array(self.con(jnp.asarray(v))),
                              jac=lambda v: np.array(self.jac(jnp.asarray(v))),
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
        outputs["phi"] = np.array(coupling(aero_loads, x[LOADS], data["weight"], v[-1]))


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

    def gradient(self, x, y, mu, data):
        # on a bound, only the part of the residual that pushes the variable out of bounds is a violation
        resid = super().gradient(x, y, mu, data)
        v = self.decompose(x)
        resid = np.where(v <= self.XL + 1e-8, np.minimum(resid, 0.0), resid)
        return np.where(v >= self.XU - 1e-8, np.maximum(resid, 0.0), resid)


# evaluate both models once at the initial design to initialize the copies and the shared data
CD_init, aero_loads_init, _ = aero_model(twist_cp0)
_, weight_init = structures_model(aero_loads_init, thickness_cp0)

opt = ALBCD(subproblems=[AeroSubproblem(slice(0, THICKNESS.start)), StructSubproblem(slice(THICKNESS.start, None))],
            x0=np.concatenate([twist_cp0, [float(weight_init)], thickness_cp0, np.array(aero_loads_init)]),
            mu0=np.full(2 * num_nodes + 1, 10.0),
            data0={"weight": float(weight_init), "aero_loads": np.array(aero_loads_init), "CD": float(CD_init)},
            max_mu=1e6, rho=1.2, tau=0.5, feas_tol=3e-4, opt_tol=1e-4, max_outer_iter=40, max_inner_iter=20)
opt.solve()

twist_cp, thickness_cp = opt.x[TWIST], opt.x[THICKNESS]
_, _, lift = aero_model(twist_cp)
max_sigma_mpa, weight = structures_model(opt.x[LOADS], thickness_cp)
print('Max von Mises stress, (KS) (MPa): ', float(max_sigma_mpa))
print('Min wall thickness (mm): ', float(np.min(bspline_thickness @ thickness_cp)) * 1e3)
print('Lift (N): ', float(lift), '  Weight (N): ', float(weight))

# compare with the monolithic solution (the same problem solved with one SLSQP): ||x - x*|| in
# [twist_cp, thickness_cp] after every block solve, and for x0
solution = np.load(os.path.join(HERE, 'monolithic_solution.npz'))
x_history = np.array(opt.history["x"])
error = np.linalg.norm(np.hstack([x_history[:, TWIST], x_history[:, THICKNESS]])
                       - np.concatenate([solution['twist_cp'], solution['thickness_cp']]), axis=1)
print('CD: ', opt.data['CD'])
print('CD (monolithic): ', float(solution['CD']))
print('||x - x*||: ', error[-1])

np.savez(os.path.join(HERE, 'ranger_albcd.npz'),
         opt_history=opt.history["optimality"], feas_history=opt.history["feasibility"], error=error)

# optimized twist and thickness
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
ax1.plot(y, np.degrees(bspline_twist @ twist_cp))
ax1.set_xlabel('Spanwise location (m)')
ax1.set_ylabel('Twist (deg)')
ax2.plot(y_elem, (bspline_thickness @ thickness_cp) * 1e3)
ax2.set_xlabel('Spanwise location (m)')
ax2.set_ylabel('Spar wall thickness (mm)')
plt.tight_layout()

# optimality and feasibility after every sweep, and the design error after every block solve
iterations = np.arange(1, len(opt.history["feasibility"]) + 1)
fig, axs = plt.subplots(1, 3, figsize=(8, 2.5))
axs[0].semilogy(iterations, opt.history["optimality"], color='tab:blue', linewidth=2)
axs[1].semilogy(iterations, opt.history["feasibility"], color='tab:orange', linewidth=2)
axs[2].plot(np.log10(error), color='tab:red', linewidth=2)
for ax, label in zip(axs, ['Optimality', 'Feasibility', r"$\log_{10}\Vert x - x^*\Vert$"]):
    ax.set_xlabel('Iteration')
    ax.set_ylabel(label)
    ax.grid(color='lavender', alpha=0.5, axis='y')
plt.tight_layout()
plt.show()

# the aircraft with the optimized (twisted) wing, and the wing's VLM mesh colored by the pressure coefficient
out = solve_aero(vlm_mesh(twist_cp), v_inf, rho_atm, symmetry=False, with_viscous=False)
cp = panel_pressure(out["mesh"], out["sec_forces"], out["normals"], orient="up") / q
plotter = pv.Plotter()
twist_ffd.twisted_aircraft(twist_cp).plot(color='silver', opacity=0.3, title='', plotter=plotter)
plotter.add_mesh(to_pyvista_mesh(out["mesh"], {"Cp": cp}), scalars="Cp", cmap="coolwarm", show_edges=True,
                 show_scalar_bar=False)
set_view(plotter)
plotter.show()
