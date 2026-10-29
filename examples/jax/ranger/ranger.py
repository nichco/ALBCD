"""Aerostructural design of the Vashon Ranger R7 wing with ALBCD.

Minimize the drag of the wing (jvlm-oas vortex-lattice aerodynamics) over its twist,
subject to lift = weight, while sizing the wall thickness of its square-tube wing box
(a jframe beam) so that the von Mises stress stays below yield and the walls stay
above a minimum gauge. The two disciplines are the two blocks, coupled through copies
of each other's outputs:

    AeroSubproblem    owns [twist_cp, weight_copy]           local constraint: lift = weight_copy
    StructSubproblem  owns [thickness_cp, aero_loads_copy]   local constraint: KS stress <= yield,
                                                             bounds: min gauge <= thickness_cp

The coupling constraints phi force the copies to equal the actual aero loads and
weight. Gradients come from JAX. Run this file to solve the problem, compare
the result with the monolithic solution in monolithic_solution.npz (regenerate it
with monolithic.py), and plot the convergence.
"""

import os
from collections import namedtuple
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # modopt and the models work in float64
import jax.numpy as jnp
import modopt as mo
import matplotlib.pyplot as plt
import pyvista as pv
from albcd import ALBCD, Subproblem
import warnings
warnings.filterwarnings("ignore")

from jvlm import panel_pressure
from jvlm import viz as vlm_viz
from models import (aero_model, structures_model, twist_cp0, thickness_cp0, num_cp_twist, num_cp_thickness,
                    num_nodes, bspline_twist, bspline_thickness, min_gauge, sigma_yield_mpa, y, y_elem, q, v_inf,
                    rho_atm, mesh0_jnp, vlm_geom, solve_aero, thickness_max)

HERE = os.path.dirname(os.path.abspath(__file__))

# global x is a single flat vector, ordered so that each subproblem's block is contiguous:
# [twist_cp, weight_copy, thickness_cp, aero_loads_copy], where
# aero_loads_copy is [drag_0..drag_{num_nodes-1}, lift_0..lift_{num_nodes-1}]
TWIST_SLICE     = slice(0, num_cp_twist)
WEIGHT_SLICE    = slice(TWIST_SLICE.stop, TWIST_SLICE.stop + 1)
THICKNESS_SLICE = slice(WEIGHT_SLICE.stop, WEIGHT_SLICE.stop + num_cp_thickness)
LOADS_SLICE     = slice(THICKNESS_SLICE.stop, THICKNESS_SLICE.stop + 2 * num_nodes)

AERO_INDEX   = slice(TWIST_SLICE.start, WEIGHT_SLICE.stop)     # owns [twist_cp, weight_copy]
STRUCT_INDEX = slice(THICKNESS_SLICE.start, LOADS_SLICE.stop)  # owns [thickness_cp, aero_loads_copy]

# coupling constraints: drag and lift residual at each node, and the weight residual
N_CON = 2 * num_nodes + 1
f_scale = 1e-3
w_scale = 1e-4


def coupling(aero_loads, aero_loads_copy, weight, weight_copy):
    return jnp.concatenate([f_scale * (aero_loads - aero_loads_copy),
                            w_scale * jnp.reshape(weight - weight_copy, 1)])


# a subproblem's objective, its gradient, its local constraints and their Jacobian, each jitted
Compiled = namedtuple("Compiled", "obj grad con jac")


def compile_subproblem(objective, local_constraints):
    """Jit objective(v, *args), its gradient in v, local_constraints(v) and their Jacobian.

    Each subproblem does this once, in setup(). The inputs that change between solves
    (other, y, mu, ...) are arguments of the compiled functions rather than constants in a
    closure, so every solve reuses the same compiled code. Wrapping a new closure in
    mo.JaxProblem for each solve would instead retrace and recompile the VLM and frame
    models every time.
    """
    return Compiled(jax.jit(objective), jax.jit(jax.grad(objective)),
                    jax.jit(local_constraints), jax.jit(jax.jacrev(local_constraints)))


def solve_slsqp(fns, args, v0, x_scaler, c_scaler, ftol, cl=0.0, cu=0.0, tau=0.0, xl=None, xu=None):
    """Minimize fns.obj(v, *args) + tau/2 ||x_scaler * (v - v0)||^2 s.t. cl <= fns.con(v) <= cu, xl <= v <= xu
    with modopt's SLSQP.

    The second term is the optional proximal term of the ALBCD paper (Eq. 25), which penalizes
    the distance from the block's current values v0. It is measured in the scaled variables that
    SLSQP sees, so that the copies of the loads and weight (O(1e2-1e3) N) do not dominate the twist
    and thickness. tau = 0 turns it off.

    Returns the solution, SLSQP's constraint multipliers and the constraint Jacobian at the solution.
    """
    w = tau * x_scaler**2  # proximal weight on each unscaled variable
    # modopt passes numpy arrays; convert them so the compiled functions always see JAX
    # arrays, as they do in residual(). jax.jit compiles numpy and JAX arguments separately.
    prob = mo.ProblemLite(x0=v0,
                          obj=lambda v: np.float64(fns.obj(jnp.asarray(v), *args)) + 0.5 * np.sum(w * (v - v0)**2),
                          grad=lambda v: np.array(fns.grad(jnp.asarray(v), *args)) + w * (v - v0),
                          con=lambda v: np.array(fns.con(jnp.asarray(v))),
                          jac=lambda v: np.array(fns.jac(jnp.asarray(v))),
                          xl=xl, xu=xu, x_scaler=x_scaler, cl=cl, cu=cu, c_scaler=c_scaler)
    optimizer = mo.SLSQP(prob, solver_options={'maxiter': 1000, 'ftol': ftol}, turn_off_outputs=True)
    optimizer.solve()
    v = optimizer.results['x'] / x_scaler
    # SLSQP solves in c_scaler-scaled constraint space, so its multipliers are
    # rescaled to the unscaled constraints used in the residual() methods below
    multipliers = np.asarray(optimizer.results['multipliers']) * c_scaler
    jac = np.atleast_2d(np.array(fns.jac(jnp.asarray(v))))
    return v, multipliers, jac


class AeroSubproblem(Subproblem):
    """Owns [twist_cp, weight_copy]: minimizes drag subject to lift = weight_copy."""

    X_SCALER = np.concatenate([np.full(num_cp_twist, 10.0), [1e-3]])  # twist_cp, weight_copy
    C_SCALER = 1e-2  # lift = weight_copy
    FTOL = 1e-8      # SLSQP tolerance
    tau = 0.0        # proximal coefficient, see solve_slsqp()

    def setup(self) -> None:
        self._fns = compile_subproblem(self.objective, self.local_constraints)

    def objective(self, v, other, y, mu, weight):
        twist_cp, weight_copy = v[:num_cp_twist], v[num_cp_twist]
        aero_loads_copy = other[num_cp_thickness:]

        CD, aero_loads, _ = aero_model(twist_cp)
        c = coupling(aero_loads, aero_loads_copy, weight, weight_copy)

        return 1e2 * CD + jnp.sum(y * c) + 0.5 * jnp.sum(mu * c**2)

    def local_constraints(self, v):
        _, _, lift = aero_model(v[:num_cp_twist])
        return jnp.reshape(lift - v[num_cp_twist], 1)

    def solve(self, x, y, mu, data, outputs) -> None:
        other, y, mu, weight = (jnp.asarray(a) for a in (self.other(x), y, mu, data["weight"]))

        v_new, self._multipliers, self._jac_con = solve_slsqp(
            self._fns, (other, y, mu, weight),
            np.array(self.decompose(x)), self.X_SCALER, self.C_SCALER, self.FTOL, tau=self.tau)

        x_new = self.recompose(x, v_new)
        CD, aero_loads, _ = aero_model(v_new[:num_cp_twist])

        outputs["x"] = x_new
        outputs["aero_loads"] = np.array(aero_loads)
        outputs["CD"] = float(CD)
        # StructSubproblem's variables, and so its weight, are unchanged by this solve
        outputs["phi"] = np.array(coupling(aero_loads, x_new[LOADS_SLICE], weight, x_new[WEIGHT_SLICE][0]))

    def residual(self, x, y, mu, data) -> float:
        v, other, y, mu, weight = (jnp.asarray(a) for a in (self.decompose(x), self.other(x),
                                                               y, mu, data["weight"]))
        grad_f = np.array(self._fns.grad(v, other, y, mu, weight))

        # KKT stationarity with SLSQP's multipliers for the local constraint
        resid = grad_f - self._jac_con.T @ self._multipliers
        return float(np.max(np.abs(resid)))


class StructSubproblem(Subproblem):
    """Owns [thickness_cp, aero_loads_copy]: sizes the wing box for stress and minimum gauge."""

    X_SCALER = np.concatenate([np.full(num_cp_thickness, 1e2), np.full(2 * num_nodes, 1e-2)])  # thickness, loads
    C_SCALER = np.array([1e-2])  # KS stress
    FTOL = 1e-8                  # SLSQP tolerance
    tau = 0.0                    # proximal coefficient, see solve_slsqp()
    # bounds on the thickness control points (m). The lower bound is the minimum gauge, active over most of
    # the span at the solution. The upper bound, half the box side where the box is solid, keeps SLSQP's
    # iterates where the box model is physical, and is inactive at the solution. The thickness is a
    # B-spline, a convex combination of its control points, so its bounds hold for every element.
    XL = np.concatenate([np.full(num_cp_thickness, min_gauge), np.full(2 * num_nodes, -np.inf)])
    XU = np.concatenate([np.full(num_cp_thickness, thickness_max), np.full(2 * num_nodes, np.inf)])
    BOUND_TOL = 1e-8  # a thickness control point within this of a bound (m) is taken to be on it

    def setup(self) -> None:
        self._fns = compile_subproblem(self.objective, self.local_constraints)

    @staticmethod
    def structures(v):
        """structures_model() of this block's variables v = [thickness_cp, aero_loads_copy]"""
        return structures_model(v[num_cp_thickness:], v[:num_cp_thickness])

    def objective(self, v, other, y, mu, aero_loads):
        aero_loads_copy = v[num_cp_thickness:]
        weight_copy = other[num_cp_twist]

        _, weight = self.structures(v)
        c = coupling(aero_loads, aero_loads_copy, weight, weight_copy)

        # the drag term of the augmented Lagrangian is constant in this block, so it is left out
        return jnp.sum(y * c) + 0.5 * jnp.sum(mu * c**2)

    def local_constraints(self, v):
        max_sigma_mpa, _ = self.structures(v)
        return jnp.reshape(max_sigma_mpa - sigma_yield_mpa, 1)

    def solve(self, x, y, mu, data, outputs) -> None:
        other, y, mu, aero_loads = (jnp.asarray(a) for a in (self.other(x), y, mu, data["aero_loads"]))

        v_new, multipliers, self._jac_con = solve_slsqp(
            self._fns, (other, y, mu, aero_loads),
            np.array(self.decompose(x)), self.X_SCALER, self.C_SCALER, self.FTOL,
            cl=np.full(1, -np.inf), cu=np.zeros(1), tau=self.tau, xl=self.XL, xu=self.XU)
        # modopt hands SLSQP each upper-bounded inequality as cu - c(v) >= 0, with
        # Jacobian -J, so its multipliers enter the stationarity condition with the
        # opposite sign to the aero block's equality constraint
        self._multipliers = -multipliers

        x_new = self.recompose(x, v_new)
        _, weight = self.structures(v_new)

        outputs["x"] = x_new
        outputs["weight"] = float(weight)
        # AeroSubproblem's variables, and so its aero loads, are unchanged by this solve
        outputs["phi"] = np.array(coupling(aero_loads, v_new[num_cp_thickness:], weight, x_new[WEIGHT_SLICE][0]))

    def residual(self, x, y, mu, data) -> float:
        v, other, y, mu, aero_loads = (jnp.asarray(a) for a in (self.decompose(x), self.other(x),
                                                                   y, mu, data["aero_loads"]))
        grad_f = np.array(self._fns.grad(v, other, y, mu, aero_loads))

        # KKT stationarity with SLSQP's multipliers for the stress constraint
        resid = grad_f - self._jac_con.T @ self._multipliers
        # A variable on a bound is stationary when what is left of its gradient pushes it against that bound,
        # where the bound's (nonnegative) multiplier balances it: resid >= 0 on a lower bound and resid <= 0
        # on an upper bound. Only the part of resid with the wrong sign there is a stationarity violation.
        v = np.asarray(v)
        resid = np.where(v <= self.XL + self.BOUND_TOL, np.minimum(resid, 0.0), resid)
        resid = np.where(v >= self.XU - self.BOUND_TOL, np.maximum(resid, 0.0), resid)
        return float(np.max(np.abs(resid)))


# evaluate both models once at the initial design to initialize the copies and the shared data
CD_init, aero_loads_init, _ = aero_model(twist_cp0)
_, weight_init = structures_model(aero_loads_init, thickness_cp0)

x0 = np.concatenate([twist_cp0, [float(weight_init)], thickness_cp0, np.array(aero_loads_init)])


def make_albcd(rho=1.2, mu0=10.0, max_inner_iter=12, max_outer_iter=40, tau=0.0):
    """The ALBCD optimizer for this problem, with penalty growth factor rho, initial penalty mu0,
    at most max_inner_iter sweeps per outer iteration, and proximal coefficient tau in both subproblems."""
    subproblems = [AeroSubproblem(AERO_INDEX), StructSubproblem(STRUCT_INDEX)]
    for sub in subproblems:
        sub.tau = tau
    return ALBCD(subproblems=subproblems,
                 x0=x0,
                 mu0=np.full(N_CON, mu0),
                 data0={"weight": float(weight_init), "aero_loads": np.array(aero_loads_init), "CD": float(CD_init)},
                 max_mu=1e6,
                 rho=rho,
                 tau=0.5,
                 feas_tol=3e-4,
                 opt_tol=[1e-1, 1e-3],
                 max_outer_iter=max_outer_iter,
                 max_inner_iter=max_inner_iter)


def design_error(x_history):
    """Relative error in [twist_cp, thickness_cp] against the monolithic solution, for each x in x_history."""
    solution = np.load(os.path.join(HERE, 'monolithic_solution.npz'))
    x_star = np.concatenate([solution['twist_cp'], solution['thickness_cp']])
    history = np.array([np.concatenate([h[TWIST_SLICE], h[THICKNESS_SLICE]]) for h in x_history])
    return np.linalg.norm(history - x_star, axis=1) / np.linalg.norm(x_star)


if __name__ == "__main__":
    opt = make_albcd()
    opt.solve()



    twist_cp = opt.x[TWIST_SLICE]
    thickness_cp = opt.x[THICKNESS_SLICE]
    aero_loads_copy = opt.x[LOADS_SLICE]

    _, _, lift = aero_model(twist_cp)
    max_sigma_mpa, weight = structures_model(aero_loads_copy, thickness_cp)
    print('Max von Mises stress, (KS) (MPa): ', float(max_sigma_mpa))
    print('Min wall thickness (mm): ', float(np.min(bspline_thickness @ thickness_cp)) * 1e3)
    print('Lift (N): ', float(lift), '  Weight (N): ', float(weight))

    # compare with the monolithic solution (the same problem solved with one SLSQP)
    solution = np.load(os.path.join(HERE, 'monolithic_solution.npz'))
    error = design_error(opt.x_history)

    print('CD: ', opt.data['CD'])
    print('CD (monolithic): ', float(solution['CD']))
    print('Relative error: ', error[-1])

    # convergence data. opt_history and feas_history have one entry per sweep; error has one per
    # subproblem solve, and a leading entry for x0
    np.savez(os.path.join(HERE, 'ranger_albcd.npz'),
             opt_history=opt.opt_history, feas_history=opt.feas_history, error=error)


    twist = np.array(bspline_twist @ twist_cp)
    thickness = np.array(bspline_thickness @ thickness_cp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.plot(y, np.degrees(twist))
    ax1.set_xlabel('Spanwise location (m)')
    ax1.set_ylabel('Twist (deg)')
    ax2.plot(y_elem, thickness * 1e3)
    ax2.set_xlabel('Spanwise location (m)')
    ax2.set_ylabel('Wing box wall thickness (mm)')
    plt.tight_layout()


    # optimality and feasibility after every sweep, and the error in the design after every block solve
    iterations = np.arange(1, len(opt.feas_history) + 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 2.5))
    ax1.semilogy(iterations, opt.opt_history, color='tab:blue', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Optimality')
    ax1.grid(color='lavender', alpha=0.5, axis='y')

    ax2.semilogy(iterations, opt.feas_history, color='tab:orange', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Feasibility')
    ax2.grid(color='lavender', alpha=0.5, axis='y')

    ax3.semilogy(error, linewidth=2, color='tab:red')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Relative error')
    ax3.grid(color='lavender', alpha=0.5, axis='y')

    plt.tight_layout()
    plt.show()


    # 3D view of the optimized wing, colored by pressure coefficient, on the Ranger R7 airframe
    def_mesh = vlm_geom.apply_twist_rotation(mesh0_jnp, jnp.rad2deg(bspline_twist @ twist_cp),
                                             ref_axis_pos=0.25, symmetry=False, rotate_x=True)
    out = solve_aero(def_mesh, v_inf, rho_atm, symmetry=False, with_viscous=False)
    cp = np.asarray(panel_pressure(out["mesh"], out["sec_forces"], out["normals"], orient="up")) / q
    vlm_grid = vlm_viz.to_pyvista_mesh(out["mesh"], cell_data={"Cp": cp})

    # The STL is in millimeters, with the span along x, the nose toward +y and z up. Scale it to meters,
    # cut away its wing panels outboard of the cabin (the VLM mesh replaces them), turn it so the span
    # is along y and the nose toward -x, and move its wing's leading edge and mid-depth to the VLM mesh's
    airframe = pv.read(os.path.join(HERE, 'Vashon_Ranger_R7.stl')).scale(1e-3, inplace=False)
    for side in (1, -1):
        airframe = airframe.clip_box(sorted([side * 0.62, side * 4.6]) + [-3.15, -0.8, 0.4, 1.15], invert=True)
    airframe = airframe.rotate_z(90, inplace=False).translate((-1.43, 0.0, -0.68), inplace=False)

    plotter = pv.Plotter(window_size=(2400, 1350))
    plotter.enable_anti_aliasing("ssaa")
    plotter.add_mesh(vlm_grid, scalars="Cp", cmap="coolwarm", show_edges=True)
    plotter.add_mesh(airframe, color="silver", show_edges=False)

    # camera at azimuth 150 deg, elevation 30 deg, 1.5 bounding-box diagonals from the center
    plotter.camera_position = "iso"  # sets the bounds and focal point before they are overridden
    center = np.array(plotter.center)
    bounds = plotter.bounds
    distance = 1.5 * np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
    az, el = np.radians(150), np.radians(30)
    direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    plotter.camera_position = [tuple(center + distance * direction), tuple(center), (0.0, 0.0, 1.0)]

    plotter.show()
