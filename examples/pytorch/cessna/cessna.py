"""Aerostructural design of a Cessna 182-like wing with ALBCD.

Minimize the drag of the wing (vortex-lattice aerodynamics) over its twist,
subject to lift = weight, while sizing the wall thickness of its tubular spar
(beam finite elements) so that the von Mises stress stays below yield and the
wall stays above a minimum gauge. The two disciplines are the two blocks,
coupled through copies of each other's outputs:

    AeroSubproblem    owns [twist_cp, weight_copy]         local constraint: lift = weight_copy
    StructSubproblem  owns [thickness_cp, aero_loads_copy]  local constraints: KS stress <= yield,
                                                                               KS thickness >= min gauge

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
import pyvista as pv
from albcd import ALBCD, Subproblem
import warnings
warnings.filterwarnings("ignore")

from models import (aero_model, structures_model, twist_cp0, thickness_cp0, num_cp_twist, num_cp_thickness,
                    num_nodes, bspline_twist, bspline_thickness, min_gauge, sigma_yield_mpa, y, q, v_inf, rho_atm,
                    mesh0_torch, vlm_geom, solve_aero)
from vlm_torch import viz as vlm_viz
from vlm_torch.analysis import panel_pressure

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


def solve_slsqp(obj, con, v0, x_scaler, c_scaler, ftol, cl=0.0, cu=0.0):
    """Minimize obj(v) s.t. cl <= con(v) <= cu with modopt's SLSQP, using PyTorch derivatives.

    Returns the solution, SLSQP's constraint multipliers and the constraint Jacobian at the solution.
    """
    prob = mo.ProblemLite(x0=v0,
                          obj=lambda v: np.float64(obj(torch.as_tensor(v))),
                          grad=lambda v: np.array(torch.func.grad(obj)(torch.as_tensor(v))),
                          con=lambda v: np.array(con(torch.as_tensor(v))),
                          jac=lambda v: np.array(torch.func.jacrev(con)(torch.as_tensor(v))),
                          x_scaler=x_scaler, cl=cl, cu=cu, c_scaler=c_scaler)
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

    def objective(self, v, other, y, mu, weight):
        twist_cp, weight_copy = v[:num_cp_twist], v[num_cp_twist]
        aero_loads_copy = other[num_cp_thickness:]

        CD, aero_loads, _ = aero_model(twist_cp)
        c = coupling(aero_loads, aero_loads_copy, weight, weight_copy)

        return 1e2 * CD + torch.sum(y * c) + 0.5 * torch.sum(mu * c**2)

    def local_constraints(self, v):
        _, _, lift = aero_model(v[:num_cp_twist])
        return (lift - v[num_cp_twist]).reshape(1)

    def solve(self, x, y, mu, data, outputs) -> None:
        other, y, mu, weight = (torch.as_tensor(a) for a in (self.other(x), y, mu, data["weight"]))

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

    def residual(self, x, y, mu, data) -> float:
        v, other, y, mu, weight = (torch.as_tensor(a) for a in (self.decompose(x), self.other(x),
                                                                  y, mu, data["weight"]))
        grad_f = np.array(torch.func.grad(self.objective)(v, other, y, mu, weight))

        # KKT stationarity with SLSQP's multipliers for the local constraint
        resid = grad_f - self._jac_con.T @ self._multipliers
        return float(np.max(np.abs(resid)))


class StructSubproblem(Subproblem):
    """Owns [thickness_cp, aero_loads_copy]: sizes the spar for stress and minimum gauge."""

    X_SCALER = np.concatenate([np.full(num_cp_thickness, 1e2), np.full(2 * num_nodes, 1e-2)])  # thickness_cp, loads
    C_SCALER = np.array([1e-2, 1e-1])  # KS stress, KS min thickness
    FTOL = 1e-8                        # SLSQP tolerance

    def objective(self, v, other, y, mu, aero_loads):
        thickness_cp, aero_loads_copy = v[:num_cp_thickness], v[num_cp_thickness:]
        weight_copy = other[num_cp_twist]

        _, _, weight = structures_model(aero_loads_copy, thickness_cp)
        c = coupling(aero_loads, aero_loads_copy, weight, weight_copy)

        # the drag term of the augmented Lagrangian is constant in this block, so it is left out
        return torch.sum(y * c) + 0.5 * torch.sum(mu * c**2)

    def local_constraints(self, v):
        max_sigma_mpa, min_thickness_mm, _ = structures_model(v[num_cp_thickness:], v[:num_cp_thickness])
        return torch.stack([max_sigma_mpa - sigma_yield_mpa,
                            min_gauge * 1e3 - min_thickness_mm])

    def solve(self, x, y, mu, data, outputs) -> None:
        other, y, mu, aero_loads = (torch.as_tensor(a) for a in (self.other(x), y, mu, data["aero_loads"]))

        v_new, multipliers, self._jac_con = solve_slsqp(
            lambda v: self.objective(v, other, y, mu, aero_loads), self.local_constraints,
            np.array(self.decompose(x)), self.X_SCALER, self.C_SCALER, self.FTOL,
            cl=np.full(2, -np.inf), cu=np.zeros(2))
        # modopt hands SLSQP each upper-bounded inequality as cu - c(v) >= 0, with
        # Jacobian -J, so its multipliers enter the stationarity condition with the
        # opposite sign to the aero block's equality constraint
        self._multipliers = -multipliers

        x_new = self.recompose(x, v_new)
        aero_loads_copy = torch.as_tensor(v_new[num_cp_thickness:])
        _, _, weight = structures_model(aero_loads_copy, torch.as_tensor(v_new[:num_cp_thickness]))

        outputs["x"] = x_new
        outputs["weight"] = float(weight)
        # AeroSubproblem's variables, and so its aero loads, are unchanged by this solve
        outputs["phi"] = np.array(coupling(aero_loads, aero_loads_copy,
                                           weight, torch.as_tensor(x_new[WEIGHT_SLICE][0])))

    def residual(self, x, y, mu, data) -> float:
        v, other, y, mu, aero_loads = (torch.as_tensor(a) for a in (self.decompose(x), self.other(x),
                                                                      y, mu, data["aero_loads"]))
        grad_f = np.array(torch.func.grad(self.objective)(v, other, y, mu, aero_loads))

        # KKT stationarity with SLSQP's multipliers for the stress and thickness constraints
        resid = grad_f - self._jac_con.T @ self._multipliers
        return float(np.max(np.abs(resid)))


# evaluate both models once at the initial design to initialize the copies and the shared data
CD_init, aero_loads_init, _ = aero_model(torch.as_tensor(twist_cp0))
_, _, weight_init = structures_model(aero_loads_init, torch.as_tensor(thickness_cp0))

x0 = np.concatenate([twist_cp0, [float(weight_init)], thickness_cp0, np.array(aero_loads_init)])

opt = ALBCD(subproblems=[AeroSubproblem(AERO_INDEX), StructSubproblem(STRUCT_INDEX)],
            x0=x0,
            mu0=np.full(N_CON, 10.0),
            data0={"weight": float(weight_init), "aero_loads": np.array(aero_loads_init), "CD": float(CD_init)},
            max_mu=1e6,
            rho=1.2,
            tau=0.5,
            feas_tol=3e-4,
            opt_tol=[1e-1, 1e-3],
            max_outer_iter=40,
            max_inner_iter=12)

opt.solve()


twist_cp = opt.x[TWIST_SLICE]
thickness_cp = opt.x[THICKNESS_SLICE]
aero_loads_copy = opt.x[LOADS_SLICE]

_, _, lift = aero_model(torch.as_tensor(twist_cp))
max_sigma_mpa, min_thickness_mm, weight = structures_model(torch.as_tensor(aero_loads_copy),
                                                           torch.as_tensor(thickness_cp))
print('Max von Mises stress, (KS) (MPa): ', float(max_sigma_mpa))
print('Min thickness, (KS) (mm): ', float(min_thickness_mm))
print('Lift (N): ', float(lift), '  Weight (N): ', float(weight))

# compare with the monolithic solution (the same problem solved with one SLSQP)
solution = np.load(os.path.join(HERE, 'monolithic_solution.npz'))
x_star = np.concatenate([solution['twist_cp'], solution['thickness_cp']])
history = np.array([np.concatenate([h[TWIST_SLICE], h[THICKNESS_SLICE]]) for h in opt.x_history])
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


# 3D view of the optimized wing, colored by pressure coefficient, on the Cessna 182 fuselage
def_mesh = vlm_geom.apply_twist_rotation(mesh0_torch, torch.rad2deg(bspline_twist @ torch.as_tensor(twist_cp)),
                                         ref_axis_pos=0.25, symmetry=False, rotate_x=True)
out = solve_aero(def_mesh, v_inf, rho_atm, symmetry=False, with_viscous=False)
cp = np.asarray(panel_pressure(out["mesh"], out["sec_forces"], out["normals"], orient="up")) / q
vlm_grid = vlm_viz.to_pyvista_mesh(out["mesh"], cell_data={"Cp": cp})

# the STL is in feet; scale to meters and place it under the wing
cessna_mesh = pv.read(os.path.join(HERE, 'cessna182_no_wing.stl'))
cessna_mesh = cessna_mesh.scale(0.3048, inplace=False).translate((-1.4, 0.0, -0.9), inplace=False)

plotter = pv.Plotter(window_size=(2400, 1350))
plotter.enable_anti_aliasing("ssaa")
plotter.add_mesh(vlm_grid, scalars="Cp", cmap="coolwarm", show_edges=True)
plotter.add_mesh(cessna_mesh, color="silver", show_edges=False)

# camera at azimuth 150 deg, elevation 30 deg, 1.5 bounding-box diagonals from the center
plotter.camera_position = "iso"  # sets the bounds and focal point before they are overridden
center = np.array(plotter.center)
bounds = plotter.bounds
distance = 1.5 * np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
az, el = np.radians(150), np.radians(30)
direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
plotter.camera_position = [tuple(center + distance * direction), tuple(center), (0.0, 0.0, 1.0)]

plotter.show()
