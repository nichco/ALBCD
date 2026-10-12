"""Monolithic reference solution of the strut-braced Cessna wing problem in cessna.py.

The same problem, solved with a single SLSQP over the twist and spar thickness
control points and the strut thicknesses, without the coupling copies
(aero_loads_copy -> aero_loads, weight_copy -> weight). Its optimum is saved to
monolithic_solution.npz, which cessna.py measures the ALBCD solution's error against.
"""

import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # modopt and the models work in float64
import jax.numpy as jnp
import modopt as mo
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from models import (aero_model, structures_model, build_frame, twist_cp0, thickness_cp0, strut_thickness0,
                    num_cp_twist, num_cp_thickness, num_struts, num_left_elems, bspline_twist, bspline_thickness,
                    min_gauge, sigma_yield_mpa, y, y_spar_elem, b, v_inf, rho_atm, mesh0_jnp, vlm_geom, vlm_aero,
                    solve_aero)

HERE = os.path.dirname(os.path.abspath(__file__))

TWIST_SLICE = slice(0, num_cp_twist)
THICKNESS_SLICE = slice(num_cp_twist, num_cp_twist + num_cp_thickness)
STRUT_SLICE = slice(num_cp_twist + num_cp_thickness, num_cp_twist + num_cp_thickness + num_struts)


def objective(x):
    CD, _, _ = aero_model(x[TWIST_SLICE])
    return CD


def constraints(x):
    _, aero_loads, lift = aero_model(x[TWIST_SLICE])
    max_sigma_mpa, min_thickness_mm, weight = structures_model(aero_loads, x[THICKNESS_SLICE], x[STRUT_SLICE])
    return jnp.stack([max_sigma_mpa - sigma_yield_mpa,
                      lift - weight,
                      min_gauge * 1e3 - min_thickness_mm])


x0 = np.concatenate([twist_cp0, thickness_cp0, strut_thickness0])

cl = np.array([-np.inf, 0.0, -np.inf])  # stress <= yield, lift = weight, thickness >= min gauge
cu = np.array([0.0, 0.0, 0.0])
c_scaler = np.array([1e-2, 1e-2, 1e-1])
x_scaler = np.concatenate([10 * np.ones(num_cp_twist), 100 * np.ones(num_cp_thickness + num_struts)])

# JaxProblem jits the objective, the constraints and their derivatives (jax.grad, jax.jacrev)
prob = mo.JaxProblem(x0=x0, jax_obj=objective, jax_con=constraints,
                     cl=cl, cu=cu, x_scaler=x_scaler, c_scaler=c_scaler, o_scaler=1e2)

optimizer = mo.SLSQP(prob, solver_options={'maxiter': 300, 'ftol': 1e-8}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()

x = optimizer.results['x'] / x_scaler
twist_cp_star = x[TWIST_SLICE]
thickness_cp_star = x[THICKNESS_SLICE]
strut_thickness_star = x[STRUT_SLICE]

CD_star = float(objective(x))

np.savez(os.path.join(HERE, 'monolithic_solution.npz'),
         twist_cp=twist_cp_star,
         thickness_cp=thickness_cp_star,
         strut_thickness=strut_thickness_star,
         CD=CD_star,
         )


twist = bspline_twist @ twist_cp_star
thickness = bspline_thickness @ thickness_cp_star

def_mesh = vlm_geom.apply_twist_rotation(
    mesh0_jnp, jnp.rad2deg(twist), ref_axis_pos=0.25, symmetry=False, rotate_x=True)
out = solve_aero(def_mesh, v_inf, rho_atm, symmetry=False, with_viscous=False)
print('CD: ', float(out["CD"]))
print('CL: ', float(out["CL"]))

_, aero_loads, lift = aero_model(twist_cp_star)
max_sigma_mpa, min_thickness_mm, weight = structures_model(aero_loads, thickness_cp_star, strut_thickness_star)
frame = build_frame(aero_loads, thickness_cp_star, strut_thickness_star)
spars, struts = frame.beams[:2], frame.beams[2:]
print('Lift (N): ', float(lift), '  Weight (N): ', float(weight))
print('Spar mass (kg): ', float(sum(beam.mass for beam in spars)),
      '  Strut mass (kg): ', float(sum(beam.mass for beam in struts)))
print('Max von Mises stress, (KS) (MPa): ', float(max_sigma_mpa))
print('Min thickness, (KS) (mm): ', float(min_thickness_mm))
print('Min spar thickness, (true) (mm): ', float(jnp.min(thickness)) * 1e3)
print('Strut thickness (mm): ', strut_thickness_star * 1e3)


fig, (ax_twist, ax_thick, ax_lift) = plt.subplots(3, 1, figsize=(4, 3.75))
facecolor = 'whitesmoke'

nspan_twist = y / (b / 2)
nspan_thick = y_spar_elem / (b / 2)
nspan_strip = 0.5 * (y[:-1] + y[1:]) / (b / 2)


def spar_line(values):
    """break the line at the fuselage, between the left and right spars"""
    return np.insert(nspan_thick, num_left_elems, np.nan), np.insert(np.asarray(values), num_left_elems, np.nan)


twist_initial = np.degrees(np.array(bspline_twist @ twist_cp0))
ax_twist.plot(nspan_twist, twist_initial, linewidth=1.5, label='Initial')
ax_twist.plot(nspan_twist, np.degrees(np.array(twist)), linewidth=1.5, label='Optimal')
ax_twist.legend()
ax_twist.set_xlim([-1, 1])
ax_twist.set_ylabel('Twist (deg)')
ax_twist.set_xticks([])
ax_twist.set_facecolor(facecolor)

thickness_initial = np.array(bspline_thickness @ thickness_cp0)
ax_thick.plot(*spar_line(thickness_initial * 1e3), linewidth=1.5, label='Initial')
ax_thick.plot(*spar_line(np.array(thickness) * 1e3), linewidth=1.5, label='Optimal')
ax_thick.legend()
ax_thick.set_xlim([-1, 1])
ax_thick.set_ylabel('Thickness (mm)')
ax_thick.set_xticks([])
ax_thick.set_facecolor(facecolor)

strip_forces = vlm_aero.sectional_forces(out["sec_forces"])   # (num_elems, 3), physical panel forces
lift_per_span = np.array(strip_forces[:, 2]) / np.array(out["widths"])   # N/m

L_total = float(out["L"])
elliptical = (4 * L_total / (np.pi * b)) * np.sqrt(np.clip(1 - nspan_strip**2, 0.0, None))

ax_lift.plot(nspan_strip, lift_per_span, linewidth=1.5, label='Solution')
ax_lift.plot(nspan_strip, elliptical, linewidth=1.5, linestyle='--', label='Elliptical', color='tab:green')
ax_lift.legend()
ax_lift.set_xlim([-1, 1])
ax_lift.set_xlabel('Normalized spanwise location')
ax_lift.set_ylabel('Lift (N/m)')
ax_lift.set_xticks([-1, -0.5, 0, 0.5, 1])
ax_lift.set_facecolor(facecolor)

plt.tight_layout(h_pad=0.2)
plt.show()
