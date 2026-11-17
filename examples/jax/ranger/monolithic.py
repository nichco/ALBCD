import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import modopt as mo
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pyvista as pv
from jeometry.plotting import set_view, sampled_mesh, EDGE_COLOR
from jvlm import panel_pressure
from jvlm.viz import to_pyvista_mesh
import warnings
warnings.filterwarnings("ignore")

from models import (aero_model, structures_model, build_frame, twist_cp0, thickness_cp0, num_cp_twist,
                    num_cp_thickness, bspline_twist, bspline_thickness, min_gauge, sigma_yield_mpa, y, y_elem,
                    b, v_inf, rho_atm, q, vlm_mesh, vlm_aero, solve_aero, thickness_max, struct_mesh, d_tube)
from ranger_geometry_parameterization import ranger, twist_ffd

HERE = os.path.dirname(os.path.abspath(__file__))

TWIST_SLICE = slice(0, num_cp_twist)
THICKNESS_SLICE = slice(num_cp_twist, num_cp_twist + num_cp_thickness)


def objective(x):
    CD, _, _ = aero_model(x[TWIST_SLICE])
    return CD


def constraints(x):
    _, aero_loads, lift = aero_model(x[TWIST_SLICE])
    max_sigma_mpa, weight = structures_model(aero_loads, x[THICKNESS_SLICE])
    return jnp.stack([max_sigma_mpa - sigma_yield_mpa,
                      lift - weight])


x0 = np.concatenate([twist_cp0, thickness_cp0])

cl = np.array([-np.inf, 0.0])  # stress <= yield, lift = weight
cu = np.array([0.0, 0.0])
c_scaler = np.array([1e-2, 1e-2])
x_scaler = np.concatenate([10 * np.ones(num_cp_twist), 100 * np.ones(num_cp_thickness)])
# the same thickness bounds as the structures block in ranger.py: min gauge <= thickness <= d_tube / 2
xl = np.concatenate([np.full(num_cp_twist, -np.inf), np.full(num_cp_thickness, min_gauge)])
xu = np.concatenate([np.full(num_cp_twist, np.inf), np.full(num_cp_thickness, thickness_max)])

# JaxProblem jits the objective, the constraints and their derivatives (jax.grad, jax.jacrev)
prob = mo.JaxProblem(x0=x0, jax_obj=objective, jax_con=constraints, xl=xl, xu=xu,
                     cl=cl, cu=cu, x_scaler=x_scaler, c_scaler=c_scaler, o_scaler=1e2)

optimizer = mo.SLSQP(prob, solver_options={'maxiter': 300, 'ftol': 1e-8}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()

x = optimizer.results['x'] / x_scaler
twist_cp_star = x[TWIST_SLICE]
thickness_cp_star = x[THICKNESS_SLICE]

CD_star = float(objective(x))

np.savez(os.path.join(HERE, 'monolithic_solution.npz'),
         twist_cp=twist_cp_star,
         thickness_cp=thickness_cp_star,
         CD=CD_star,
         )


twist = bspline_twist @ twist_cp_star
thickness = bspline_thickness @ thickness_cp_star

def_mesh = vlm_mesh(twist_cp_star)
out = solve_aero(def_mesh, v_inf, rho_atm, symmetry=False, with_viscous=False)
print('CD: ', float(out["CD"]))
print('CL: ', float(out["CL"]))

_, aero_loads, lift = aero_model(twist_cp_star)
max_sigma_mpa, weight = structures_model(aero_loads, thickness_cp_star)
frame = build_frame(aero_loads, thickness_cp_star)
u = frame.solve()
print('Lift (N): ', float(lift), '  Weight (N): ', float(weight))
print('Wing box mass (kg): ', float(frame.mass))
print('Max von Mises stress, (KS) (MPa): ', float(max_sigma_mpa))
print('Min wall thickness (mm): ', float(jnp.min(thickness)) * 1e3)
print('Tip deflection at the design load (m): ', float(u[0][-1, 2]))


fig, (ax_twist, ax_thick, ax_lift) = plt.subplots(3, 1, figsize=(4, 3.75))
facecolor = 'whitesmoke'

nspan_twist = y / (b / 2)
nspan_thick = y_elem / (b / 2)

twist_initial = np.degrees(np.array(bspline_twist @ twist_cp0))
ax_twist.plot(nspan_twist, twist_initial, linewidth=1.5, label='Initial')
ax_twist.plot(nspan_twist, np.degrees(np.array(twist)), linewidth=1.5, label='Optimal')
ax_twist.legend()
ax_twist.set_xlim([-1, 1])
ax_twist.set_ylabel('Twist (deg)')
ax_twist.set_xticks([])
ax_twist.set_facecolor(facecolor)

thickness_initial = np.array(bspline_thickness @ thickness_cp0)
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


# the aircraft with the optimal (twisted) wing, and the wing's VLM mesh colored by the pressure coefficient
cp = panel_pressure(out["mesh"], out["sec_forces"], out["normals"], orient="up") / q
plotter = pv.Plotter()
twist_ffd.twisted_aircraft(twist_cp_star).plot(color='silver', opacity=0.3, title='', plotter=plotter)
plotter.add_mesh(to_pyvista_mesh(out["mesh"], {"Cp": cp}), scalars="Cp", cmap="coolwarm", show_edges=True,
                 show_scalar_bar=False)
set_view(plotter)
plotter.show()


# Problem figure (fig_problem_render.pdf): the optimal design on the airframe, split at the root. The left
# semispan shows the vortex lattice colored by the pressure coefficient, and the right semispan the tube spar,
# deflected under the design load (true scale) over an outline of the undeflected spar, colored by its von Mises
# stress relative to yield. The airframe and the wing are translucent.
mesh = np.asarray(out["mesh"])
cp = np.asarray(cp)
cp_norm = Normalize(0.0, np.ceil(cp.max() * 10) / 10)

# the vortex lattice, split at the root (the middle spanwise station): the left half colored by Cp, the right
# half a translucent skin over the spar. StructuredGrid orders its cells chordwise first.
root = mesh.shape[1] // 2
aero = pv.StructuredGrid(*mesh[:, :root + 1].transpose(2, 0, 1))
aero.cell_data["Cp"] = cp[:, :root].ravel(order="F")
skin = pv.StructuredGrid(*mesh[:, root:].transpose(2, 0, 1))

# the right half of the spar, deflected, one segment per element colored by its stress, and its undeflected
# outline, slightly thinner so that it stays inside the deflected spar near the root, where the two coincide
nodes = struct_mesh[root:]
spar = pv.lines_from_points(nodes + np.outer(np.asarray(u[0][root:, 2]), [0, 0, 1]))
spar.cell_data["sigma"] = np.asarray(frame.beams[0].stress(u[0]))[root:] / 1e6 / sigma_yield_mpa
spar = spar.tube(radius=d_tube / 2, n_sides=48, capping=True)
undeflected = pv.lines_from_points(nodes[[0, -1]]).tube(radius=0.97 * d_tube / 2, n_sides=96)

# the airframe without its wing, the propeller blades and wheels (OpenVSP names are "<id>, <component>, ...", so
# the commas match whole component names), and the wing, twisted to the optimum like the vortex lattice
gray_parts = [", prop,", ", gear,"]
airframe, airframe_edges = sampled_mesh(ranger.declare_component(function_search_names=[""],
                                                                 ignore_names=["WingGeom"] + gray_parts),
                                        samples_per_span=8)
prop_and_wheels, prop_and_wheels_edges = sampled_mesh(ranger.declare_component(function_search_names=gray_parts),
                                                      samples_per_span=8)
wing, wing_edges = sampled_mesh(twist_ffd.twisted_wing(twist_cp_star), samples_per_span=8)

shaded = dict(smooth_shading=True, diffuse=0.7, ambient=0.3)
plotter = pv.Plotter(off_screen=True, window_size=(2800, 1500))
plotter.set_background("white")
plotter.enable_anti_aliasing("ssaa")
plotter.enable_depth_peeling()  # sort the translucent surfaces correctly
plotter.add_mesh(airframe, color="white", opacity=0.7, **shaded)
plotter.add_mesh(prop_and_wheels, color="lightgray", opacity=0.7, **shaded)
plotter.add_mesh(wing, color="white", opacity=0.25, **shaded)
# the edges of the STEP surfaces, as in jeometry's Geometry.plot
for edges in (airframe_edges, prop_and_wheels_edges, wing_edges):
    plotter.add_mesh(edges, color=EDGE_COLOR, line_width=1.0, opacity=0.6, lighting=False)
plotter.add_mesh(aero, scalars="Cp", cmap="Blues", clim=(cp_norm.vmin, cp_norm.vmax), show_edges=True,
                 edge_color="#6f6f6f", line_width=0.6, show_scalar_bar=False, lighting=False)
plotter.add_mesh(skin, color="#f2f2f2", opacity=0.55, show_edges=True, edge_color="#9a9a9a", line_width=0.6,
                 lighting=False)
plotter.add_mesh(undeflected, color="#9a9a9a", opacity=0.35, show_edges=True, edge_color="#5a5a5a", lighting=False)
plotter.add_mesh(spar, scalars="sigma", cmap="Oranges", clim=(0, 1), show_scalar_bar=False)

# camera ahead of and above the left wing, looking aft
center = np.array([0.9, -0.3, -0.2])
az, el = np.radians(205), np.radians(28)
direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
plotter.camera_position = [center + 12.5 * direction, center, (0, 0, 1)]
plotter.camera.view_angle = 30
image = plotter.screenshot(return_img=True)
plotter.close()

# crop the white margins, then add the colorbars below the image
ink = (image < 250).any(axis=2)
rows, cols = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
image = image[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
height = 7 * image.shape[0] / image.shape[1]  # of the image (in), 7 in wide
with plt.rc_context({"font.family": "STIXGeneral", "font.size": 14, "xtick.labelsize": 13, "axes.linewidth": 0.6,
                     "xtick.direction": "in", "xtick.major.size": 2.5, "xtick.major.width": 0.6,
                     "pdf.fonttype": 42}):
    fig = plt.figure(figsize=(7, height + 0.55))
    ax = fig.add_axes([0, 0.55 / (height + 0.55), 1, height / (height + 0.55)])
    ax.imshow(image)
    ax.axis("off")
    # the camera sees the right (structure) semispan on the left of the image
    for left, mappable, label in ((0.08, ScalarMappable(Normalize(0, 1), "Oranges"), "Normalized spar stress"),
                                  (0.58, ScalarMappable(cp_norm, "Blues"), "Pressure coefficient")):
        fig.colorbar(mappable, cax=fig.add_axes([left, 0.06, 0.34, 0.035]), orientation="horizontal").set_label(
            label, labelpad=2)
    fig.savefig(os.path.join(HERE, "fig_problem_render.pdf"), bbox_inches="tight", pad_inches=0.02, dpi=300)
