"""Aerostructural wing design with ALBCD.

Minimize the drag of a tapered wing (vortex-lattice aerodynamics) over its twist,
subject to lift = weight, while sizing the wall thickness of its tubular spar
(beam finite elements) so that both wing tips deflect by a target amount. The
two disciplines are the two blocks, coupled through copies of each other's outputs:

    AeroSubproblem    owns [twist_cp, weight_copy]         local constraint: lift = weight_copy
    StructSubproblem  owns [thickness_cp, aero_loads_copy]  local constraints: tip displacements = target

The coupling constraints phi force the copies to equal the actual aero loads and
weight. Gradients come from PyTorch. Run this file to solve the problem, compare
the result with the monolithic solution in monolithic_solution.npz (regenerate it
with monolithic.py), and plot the convergence.
"""

import os
import numpy as np
import torch
torch.set_default_dtype(torch.float64)  # modopt and the models work in float64
import modopt as mo
import matplotlib.pyplot as plt
from albcd import ALBCD, Subproblem
import warnings
warnings.filterwarnings("ignore")

from models import (aero_model, structures_model, twist_cp0, thickness_cp0, num_cp_twist, num_cp_thickness,
                    num_nodes, bspline_twist, bspline_thickness, min_gauge, r, tip_disp_target, y)

HERE = os.path.dirname(os.path.abspath(__file__))

# global x is a single flat vector, ordered so that each subproblem's block is contiguous:
# [twist_cp, weight_copy, thickness_cp, aero_loads_copy], where aero_loads_copy is
# [drag_0..drag_{num_nodes-1}, lift_0..lift_{num_nodes-1}]
TWIST_SLICE     = slice(0, num_cp_twist)
WEIGHT_SLICE    = slice(num_cp_twist, num_cp_twist + 1)
THICKNESS_SLICE = slice(num_cp_twist + 1, num_cp_twist + 1 + num_cp_thickness)
LOADS_SLICE     = slice(num_cp_twist + 1 + num_cp_thickness,
                        num_cp_twist + 1 + num_cp_thickness + 2 * num_nodes)

AERO_INDEX   = slice(TWIST_SLICE.start, WEIGHT_SLICE.stop)     # owns [twist_cp, weight_copy]
STRUCT_INDEX = slice(THICKNESS_SLICE.start, LOADS_SLICE.stop)  # owns [thickness_cp, aero_loads_copy]

# coupling constraints: drag and lift residual at each node, and the weight residual
N_CON = 2 * num_nodes + 1
f_scale = 1e-3
w_scale = 1e-4


def coupling(aero_loads, aero_loads_copy, weight, weight_copy):
    return torch.cat([f_scale * (aero_loads - aero_loads_copy),
                      w_scale * (weight - weight_copy).reshape(1)])


def solve_slsqp(obj, con, v0, x_scaler, c_scaler, ftol, xl=None, xu=None):
    """Minimize obj(v) s.t. con(v) = 0 and xl <= v <= xu with modopt's SLSQP, using PyTorch derivatives.

    Returns the solution, SLSQP's constraint multipliers and the constraint Jacobian at the solution.
    """
    prob = mo.ProblemLite(x0=v0,
                          obj=lambda v: np.float64(obj(torch.as_tensor(v))),
                          grad=lambda v: np.array(torch.func.grad(obj)(torch.as_tensor(v))),
                          con=lambda v: np.array(con(torch.as_tensor(v))),
                          jac=lambda v: np.array(torch.func.jacrev(con)(torch.as_tensor(v))),
                          xl=xl, xu=xu, x_scaler=x_scaler, cl=0, cu=0, c_scaler=c_scaler)
    optimizer = mo.SLSQP(prob, solver_options={'maxiter': 1000, 'ftol': ftol}, turn_off_outputs=True)
    optimizer.solve()
    v = optimizer.results['x'] / x_scaler
    # SLSQP solves in c_scaler-scaled constraint space, so its multipliers are
    # rescaled to the unscaled constraints used in the residual() methods below
    multipliers = np.asarray(optimizer.results['multipliers']) * c_scaler
    jac = np.atleast_2d(np.array(torch.func.jacrev(con)(torch.as_tensor(v))))
    return v, multipliers, jac


class AeroSubproblem(Subproblem):
    """Owns [twist_cp, weight_copy]: minimizes drag subject to lift = weight_copy."""

    X_SCALER = np.concatenate([np.full(num_cp_twist, 10.0), [1e-3]])  # twist_cp, weight_copy
    C_SCALER = 1e-2  # lift = weight_copy
    FTOL = 1e-8      # SLSQP tolerance

    def setup(self) -> None:
        self.add_input("x")
        self.add_input("y")  # Lagrange multipliers
        self.add_input("mu") # penalty parameters
        self.add_input("weight")  # from StructSubproblem
        self.add_output("x")
        self.add_output("phi") # coupling constraints at the new x
        self.add_output("aero_loads")  # to StructSubproblem
        self.add_output("CD")

    def objective(self, v, other, y, mu, weight):
        twist_cp, weight_copy = v[:num_cp_twist], v[num_cp_twist]
        aero_loads_copy = other[num_cp_thickness:]

        CD, aero_loads, _ = aero_model(twist_cp)
        c = coupling(aero_loads, aero_loads_copy, weight, weight_copy)

        return 1e2 * CD + torch.sum(y * c) + 0.5 * torch.sum(mu * c**2)

    def local_constraints(self, v):
        _, _, lift = aero_model(v[:num_cp_twist])
        return (lift - v[num_cp_twist]).reshape(1)

    def solve(self, inputs, outputs) -> None:
        x = inputs["x"]
        other, y, mu, weight = (torch.as_tensor(a) for a in (self.other(x), inputs["y"], inputs["mu"], inputs["weight"]))

        v_new, self._multipliers, self._jac_con = solve_slsqp(
            lambda v: self.objective(v, other, y, mu, weight), self.local_constraints,
            np.array(self.decompose(x)), self.X_SCALER, self.C_SCALER, self.FTOL)

        x_new = self.recompose(x, v_new)
        CD, aero_loads, _ = aero_model(torch.as_tensor(v_new[:num_cp_twist]))

        outputs["x"] = x_new
        outputs["aero_loads"] = np.array(aero_loads)
        outputs["CD"] = float(CD)
        # StructSubproblem's variables, and so its weight, are unchanged by this solve
        outputs["phi"] = np.array(coupling(aero_loads, torch.as_tensor(x_new[LOADS_SLICE]),
                                           weight, torch.as_tensor(x_new[WEIGHT_SLICE][0])))

    def residual(self, inputs) -> float:
        x = inputs["x"]
        v, other, y, mu, weight = (torch.as_tensor(a) for a in (self.decompose(x), self.other(x),
                                                                  inputs["y"], inputs["mu"], inputs["weight"]))
        grad_f = np.array(torch.func.grad(self.objective)(v, other, y, mu, weight))

        # KKT stationarity with SLSQP's multipliers for the local constraint
        resid = grad_f - self._jac_con.T @ self._multipliers
        return float(np.max(np.abs(resid)))


class StructSubproblem(Subproblem):
    """Owns [thickness_cp, aero_loads_copy]: sizes the spar so both tip displacements equal the target."""

    # thickness_cp, drag copies, lift copies. The drag loads are ~100x smaller than the lift
    # loads, so they get their own scaler; a larger thickness scaler would magnify SLSQP's
    # termination error in residual(), which is measured in unscaled variables.
    X_SCALER = np.concatenate([np.full(num_cp_thickness, 1e1), np.full(num_nodes, 1.0), np.full(num_nodes, 1e-2)])
    C_SCALER = 1e1  # tip displacements
    # SLSQP tolerance. Tighter than AeroSubproblem's: with 1e-8, this block's residual
    # stalls above the final opt_tol of 1e-3.
    FTOL = 1e-9

    # thickness_cp in [min gauge, smallest tube radius], aero_loads_copy unbounded. The
    # B-spline basis is nonnegative and sums to one, so every element's thickness is a
    # convex combination of the control points and inherits the same bounds.
    XL = np.concatenate([np.full(num_cp_thickness, min_gauge),      np.full(2 * num_nodes, -np.inf)])
    XU = np.concatenate([np.full(num_cp_thickness, float(r.min())), np.full(2 * num_nodes,  np.inf)])

    def setup(self) -> None:
        self.add_input("x")
        self.add_input("y")  # Lagrange multipliers
        self.add_input("mu") # penalty parameters
        self.add_input("aero_loads")  # from AeroSubproblem
        self.add_output("x")
        self.add_output("phi") # coupling constraints at the new x
        self.add_output("weight")  # to AeroSubproblem

    def objective(self, v, other, y, mu, aero_loads):
        thickness_cp, aero_loads_copy = v[:num_cp_thickness], v[num_cp_thickness:]
        weight_copy = other[num_cp_twist]

        _, _, weight = structures_model(aero_loads_copy, thickness_cp)
        c = coupling(aero_loads, aero_loads_copy, weight, weight_copy)

        # the drag term of the augmented Lagrangian is constant in this block, so it is left out
        return torch.sum(y * c) + 0.5 * torch.sum(mu * c**2)

    def local_constraints(self, v):
        u_r, u_l, _ = structures_model(v[num_cp_thickness:], v[:num_cp_thickness])
        return torch.stack([u_r - tip_disp_target,
                            u_l - tip_disp_target])

    def solve(self, inputs, outputs) -> None:
        x = inputs["x"]
        other, y, mu, aero_loads = (torch.as_tensor(a) for a in (self.other(x), inputs["y"], inputs["mu"], inputs["aero_loads"]))

        v_new, self._multipliers, self._jac_con = solve_slsqp(
            lambda v: self.objective(v, other, y, mu, aero_loads), self.local_constraints,
            np.array(self.decompose(x)), self.X_SCALER, self.C_SCALER, self.FTOL, xl=self.XL, xu=self.XU)

        x_new = self.recompose(x, v_new)
        aero_loads_copy = torch.as_tensor(v_new[num_cp_thickness:])
        _, _, weight = structures_model(aero_loads_copy, torch.as_tensor(v_new[:num_cp_thickness]))

        outputs["x"] = x_new
        outputs["weight"] = float(weight)
        # AeroSubproblem's variables, and so its aero loads, are unchanged by this solve
        outputs["phi"] = np.array(coupling(aero_loads, aero_loads_copy,
                                           weight, torch.as_tensor(x_new[WEIGHT_SLICE][0])))

    def residual(self, inputs) -> float:
        x = inputs["x"]
        v, other, y, mu, aero_loads = (torch.as_tensor(a) for a in (self.decompose(x), self.other(x),
                                                                      inputs["y"], inputs["mu"], inputs["aero_loads"]))
        grad_f = np.array(torch.func.grad(self.objective)(v, other, y, mu, aero_loads))

        # KKT stationarity with SLSQP's multipliers for the (equality) tip-displacement constraints
        resid = grad_f - self._jac_con.T @ self._multipliers

        # SLSQP's multipliers don't cover the box bounds, so at an active bound only the
        # component of the residual pointing into the infeasible region counts
        v = np.array(v)
        violation = np.abs(resid)
        violation = np.where(v <= self.XL + 1e-6, np.maximum(0.0, -resid), violation)
        violation = np.where(v >= self.XU - 1e-6, np.maximum(0.0, resid), violation)
        return float(np.max(violation))


# evaluate both models once at the initial design to initialize the copies and the shared weight
_, aero_loads_init, _ = aero_model(torch.as_tensor(twist_cp0))
_, _, weight_init = structures_model(aero_loads_init, torch.as_tensor(thickness_cp0))

x0 = np.concatenate([twist_cp0, [float(weight_init)], thickness_cp0, np.array(aero_loads_init)])

opt = ALBCD(subproblems=[AeroSubproblem(AERO_INDEX), StructSubproblem(STRUCT_INDEX)],
            x0=x0,
            mu0=np.ones(N_CON),
            data0={"weight": float(weight_init)},
            max_mu=1e6,
            rho=1.2,
            tau=0.5,
            feas_tol=3e-5,
            opt_tol=[1e-1, 1e-3],
            max_outer_iter=60,
            max_inner_iter=10)

opt.solve()


twist_cp = opt.x[TWIST_SLICE]
thickness_cp = opt.x[THICKNESS_SLICE]
aero_loads_copy = opt.x[LOADS_SLICE]

_, _, lift = aero_model(torch.as_tensor(twist_cp))
u_r, u_l, weight = structures_model(torch.as_tensor(aero_loads_copy), torch.as_tensor(thickness_cp))
print('Tip displacements (m): ', float(u_r), float(u_l))
print('Lift (N): ', float(lift), '  Weight (N): ', float(weight))

# compare with the monolithic solution (the same problem solved with one SLSQP)
solution = np.load(os.path.join(HERE, 'monolithic_solution.npz'))
x_star = np.concatenate([solution['twist_cp'], solution['thickness_cp']])
history = np.array([np.concatenate([h[TWIST_SLICE], h[THICKNESS_SLICE]]) for h in opt.history])
error = np.linalg.norm(history - x_star, axis=1) / np.linalg.norm(x_star)

print('CD: ', opt.data['CD'])
print('CD (monolithic): ', float(solution['CD']))
print('Relative error: ', error[-1])


twist = np.array(bspline_twist @ torch.as_tensor(twist_cp))
thickness = np.array(bspline_thickness @ torch.as_tensor(thickness_cp))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
ax1.plot(y, np.degrees(twist))
ax1.set_xlabel('Spanwise location (m)')
ax1.set_ylabel('Twist (deg)')
ax2.plot(0.5 * (y[:-1] + y[1:]), thickness * 1e3)
ax2.set_xlabel('Spanwise location (m)')
ax2.set_ylabel('Thickness (mm)')
plt.tight_layout()


# optimality and feasibility after every subproblem solve, with their tolerances.
# Optimality oscillates because solving one block leaves the other non-stationary.
iterations = np.arange(1, len(opt.feas_history) + 1)  # opt_history[0] is nan; see ALBCD.opt_history

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 2.5))
ax1.semilogy(iterations, opt.opt_history, color='tab:blue', linewidth=2)
# ax1.axhline(opt.opt_tol[-1], color='tab:gray', linewidth=1, linestyle='--', alpha=0.5)
# ax1.annotate('Tolerance', xy=(0.3, opt.opt_tol[-1]), xycoords=('axes fraction', 'data'), xytext=(0, 3), textcoords='offset points', ha='center', color='tab:gray', fontsize=8)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Optimality')
ax1.grid(color='lavender', alpha=0.5, axis='y')

ax2.semilogy(iterations, opt.feas_history, color='tab:orange', linewidth=2)
# ax2.axhline(opt.feas_tol, color='tab:gray', linewidth=1, linestyle='--', alpha=0.5)
# ax2.annotate('Tolerance', xy=(0.3, opt.feas_tol), xycoords=('axes fraction', 'data'), xytext=(0, 3), textcoords='offset points', ha='center', color='tab:gray', fontsize=8)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Feasibility')
ax2.grid(color='lavender', alpha=0.5, axis='y')

ax3.semilogy(error, linewidth=2, color='tab:red')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Relative error')
ax3.grid(color='lavender', alpha=0.5, axis='y')

plt.tight_layout()

# plt.savefig('aerostruct.pdf')

plt.show()
