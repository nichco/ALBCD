"""Monolithic reference solution of the aerostructural problem in aerostruct.py.

The same problem, solved with a single SLSQP over the twist and thickness
control points, without the coupling copies (aero_loads_copy -> aero_loads,
weight_copy -> weight). Its optimum is saved to monolithic_solution.npz, which
aerostruct.py measures the ALBCD solution's error against.
"""

import os
import numpy as np
import torch
import modopt as mo
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from beam_torch import Beam, CSTube
from models import (aero_model, structures_model, twist_cp0, thickness_cp0, num_cp_twist, num_cp_thickness,
                    bspline_twist, bspline_thickness, min_gauge, r, tip_disp_target, y, b, v_inf, rho_atm,
                    mesh0_torch, vlm_geom, vlm_aero, solve_aero, struct_mesh, num_nodes, fixed_nodes,
                    E, G, rho_mat, load_factor, safety_factor)

HERE = os.path.dirname(os.path.abspath(__file__))


def objective(x):
    CD, _, _ = aero_model(x[:num_cp_twist])
    return CD


def constraints(x):
    _, aero_loads, lift = aero_model(x[:num_cp_twist])
    u_r, u_l, weight = structures_model(aero_loads, x[num_cp_twist:])
    return torch.stack([lift - weight,
                        u_r - tip_disp_target,
                        u_l - tip_disp_target])


x0 = np.concatenate([twist_cp0, thickness_cp0])

# same thickness_cp box as aerostruct.py's StructSubproblem, twist_cp free
xl = np.concatenate([np.full(num_cp_twist, -np.inf), np.full(num_cp_thickness, min_gauge)])
xu = np.concatenate([np.full(num_cp_twist,  np.inf), np.full(num_cp_thickness, float(r.min()))])
c_scaler = np.array([1e-3, 1e1, 1e1])
x_scaler = np.concatenate([10 * np.ones(num_cp_twist), 100 * np.ones(num_cp_thickness)])

prob = mo.ProblemLite(x0=x0,
                      obj=lambda x: np.float64(objective(torch.as_tensor(x))),
                      grad=lambda x: np.array(torch.func.grad(objective)(torch.as_tensor(x))),
                      con=lambda x: np.array(constraints(torch.as_tensor(x))),
                      jac=lambda x: np.array(torch.func.jacrev(constraints)(torch.as_tensor(x))),
                      xl=xl, xu=xu, cl=0, cu=0, x_scaler=x_scaler, c_scaler=c_scaler, o_scaler=1e3)

optimizer = mo.SLSQP(prob, solver_options={'maxiter': 500, 'ftol': 1e-10}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()

x = optimizer.results['x'] / x_scaler
twist_cp_star = x[:num_cp_twist]
thickness_cp_star = x[num_cp_twist:]

CD_star = float(objective(torch.as_tensor(x)))

np.savez(os.path.join(HERE, 'monolithic_solution.npz'),
         twist_cp=twist_cp_star,
         thickness_cp=thickness_cp_star,
         CD=CD_star,
         )


twist = bspline_twist @ torch.as_tensor(twist_cp_star)
thickness = bspline_thickness @ torch.as_tensor(thickness_cp_star)

def_mesh = vlm_geom.apply_twist_rotation(
    mesh0_torch, torch.rad2deg(twist), ref_axis_pos=0.25, symmetry=False, rotate_x=True)
out = solve_aero(def_mesh, v_inf, rho_atm, symmetry=False, with_viscous=False)
print('CD: ', float(out["CD"]))
print('CL: ', float(out["CL"]))

_, aero_loads, lift = aero_model(torch.as_tensor(twist_cp_star))
u_r, u_l, weight = structures_model(aero_loads, torch.as_tensor(thickness_cp_star))
print('Lift (N): ', float(lift), '  Weight (N): ', float(weight))
print('Tip displacements (m): ', float(u_r), float(u_l))
print('Min thickness (mm): ', float(torch.min(thickness)) * 1e3)

F = torch.zeros((num_nodes, 6))
F[:, 0] = aero_loads[:num_nodes] * load_factor * safety_factor
F[:, 2] = aero_loads[num_nodes:] * load_factor * safety_factor
beam = Beam(mesh=struct_mesh, E=E, G=G, rho=rho_mat,
            cs=CSTube(radius=r, thickness=thickness), F=F, fixed_nodes=fixed_nodes)
u = beam.solve()
print('Mass (kg): ', float(beam.mass))


fig, (ax_twist, ax_thick, ax_lift, ax_disp) = plt.subplots(4, 1, figsize=(4, 3.75))
facecolor = 'whitesmoke'

nspan_twist = y / (b / 2)
nspan_thick = 0.5 * (y[:-1] + y[1:]) / (b / 2)

twist_initial = np.degrees(np.array(bspline_twist @ torch.as_tensor(twist_cp0)))
ax_twist.plot(nspan_twist, twist_initial, linewidth=1.5, label='Initial')
ax_twist.plot(nspan_twist, np.degrees(np.array(twist)), linewidth=1.5, label='Optimal')
ax_twist.legend()
ax_twist.set_xlim([-1, 1])
ax_twist.set_ylabel('Twist (deg)')
ax_twist.set_xticks([])
ax_twist.set_facecolor(facecolor)

thickness_initial = np.array(bspline_thickness @ torch.as_tensor(thickness_cp0))
ax_thick.plot(nspan_thick, thickness_initial * 1e3, linewidth=1.5, label='Initial')
ax_thick.plot(nspan_thick, np.array(thickness) * 1e3, linewidth=1.5, label='Optimal')
ax_thick.legend()
ax_thick.set_xlim([-1, 1])
ax_thick.set_ylabel('Thickness (mm)')
ax_thick.set_xticks([])
ax_thick.set_facecolor(facecolor)

strip_forces = vlm_aero.sectional_forces(out["sec_forces"])   # (num_elems, 3), physical panel forces
lift_per_span = np.array(strip_forces[:, 2]) / np.array(out["widths"])   # N/m

L_total = float(out["L"])
elliptical = (4 * L_total / (np.pi * b)) * np.sqrt(np.clip(1 - nspan_thick**2, 0.0, None))

ax_lift.plot(nspan_thick, lift_per_span, linewidth=1.5, label='Solution')
ax_lift.plot(nspan_thick, elliptical, linewidth=1.5, linestyle='--', label='Elliptical', color='tab:green')
ax_lift.legend()
ax_lift.set_xlim([-1, 1])
ax_lift.set_ylabel('Lift (N/m)')
ax_lift.set_xticks([])
ax_lift.set_facecolor(facecolor)

ax_disp.plot(nspan_twist, np.array(torch.linalg.norm(u[:, :3], dim=1)), linewidth=1.5, label='Displacement')
ax_disp.axhline(tip_disp_target, color='black', linestyle='--', linewidth=1, label='Tip target')
ax_disp.legend()
ax_disp.set_xlim([-1, 1])
ax_disp.set_xlabel('Normalized spanwise location')
ax_disp.set_ylabel('Disp. (m)')
ax_disp.set_xticks([-1, -0.5, 0, 0.5, 1])
ax_disp.set_facecolor(facecolor)

plt.tight_layout(h_pad=0.2)
plt.show()
