"""Monolithic reference solution of the Cessna wing problem in cessna.py.

The same problem, solved with a single SLSQP over the twist and thickness
control points, without the coupling copies (aero_loads_copy -> aero_loads,
weight_copy -> weight). Its optimum is saved to monolithic_solution.npz, which
cessna.py measures the ALBCD solution's error against.
"""

import os
import numpy as np
import torch
torch.set_default_dtype(torch.float64)  # modopt and the models work in float64
import modopt as mo
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from models import (aero_model, structures_model, twist_cp0, thickness_cp0, num_cp_twist, num_cp_thickness,
                    bspline_twist, bspline_thickness, min_gauge, sigma_yield_mpa, y, b, v_inf, rho_atm,
                    mesh0_torch, vlm_geom, vlm_aero, solve_aero)

HERE = os.path.dirname(os.path.abspath(__file__))


def objective(x):
    CD, _, _ = aero_model(x[:num_cp_twist])
    return CD


def constraints(x):
    _, aero_loads, lift = aero_model(x[:num_cp_twist])
    max_sigma_mpa, min_thickness_mm, weight = structures_model(aero_loads, x[num_cp_twist:])
    return torch.stack([max_sigma_mpa - sigma_yield_mpa,
                        lift - weight,
                        min_gauge * 1e3 - min_thickness_mm])


x0 = np.concatenate([twist_cp0, thickness_cp0])

cl = np.array([-np.inf, 0.0, -np.inf])  # stress <= yield, lift = weight, thickness >= min gauge
cu = np.array([0.0, 0.0, 0.0])
c_scaler = np.array([1e-2, 1e-2, 1e-1])
x_scaler = np.concatenate([10 * np.ones(num_cp_twist), 100 * np.ones(num_cp_thickness)])

prob = mo.ProblemLite(x0=x0,
                      obj=lambda x: np.float64(objective(torch.as_tensor(x))),
                      grad=lambda x: np.array(torch.func.grad(objective)(torch.as_tensor(x))),
                      con=lambda x: np.array(constraints(torch.as_tensor(x))),
                      jac=lambda x: np.array(torch.func.jacrev(constraints)(torch.as_tensor(x))),
                      cl=cl, cu=cu, x_scaler=x_scaler, c_scaler=c_scaler, o_scaler=1e2)

optimizer = mo.SLSQP(prob, solver_options={'maxiter': 300, 'ftol': 1e-8}, turn_off_outputs=True)
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
max_sigma_mpa, min_thickness_mm, weight = structures_model(aero_loads, torch.as_tensor(thickness_cp_star))
print('Lift (N): ', float(lift), '  Weight (N): ', float(weight))
print('Max von Mises stress, (KS) (MPa): ', float(max_sigma_mpa))
print('Min thickness, (KS) (mm): ', float(min_thickness_mm))
print('Min thickness, (true) (mm): ', float(torch.min(thickness)) * 1e3)


fig, (ax_twist, ax_thick, ax_lift) = plt.subplots(3, 1, figsize=(4, 3.75))
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
ax_lift.set_xlabel('Normalized spanwise location')
ax_lift.set_ylabel('Lift (N/m)')
ax_lift.set_xticks([-1, -0.5, 0, 0.5, 1])
ax_lift.set_facecolor(facecolor)

plt.tight_layout(h_pad=0.2)
plt.show()
