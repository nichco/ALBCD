import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import modopt as mo
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.cm import ScalarMappable
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator
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

# This script is for iterating on the figures, so by default it reuses the solution monolithic.py saved; set
# RERUN_OPTIMIZATION (or delete monolithic_solution.npz) to solve the problem again.
RERUN_OPTIMIZATION = False
solution_file = os.path.join(HERE, 'monolithic_solution.npz')

if RERUN_OPTIMIZATION or not os.path.exists(solution_file):
    optimizer = mo.SLSQP(prob, solver_options={'maxiter': 300, 'ftol': 1e-8}, turn_off_outputs=True)
    optimizer.solve()
    optimizer.print_results()

    x = optimizer.results['x'] / x_scaler
    twist_cp_star = x[TWIST_SLICE]
    thickness_cp_star = x[THICKNESS_SLICE]

    CD_star = float(objective(x))

    np.savez(solution_file,
             twist_cp=twist_cp_star,
             thickness_cp=thickness_cp_star,
             CD=CD_star,
             )
else:
    solution = np.load(solution_file)
    twist_cp_star = solution['twist_cp']
    thickness_cp_star = solution['thickness_cp']
    x = np.concatenate([twist_cp_star, thickness_cp_star])


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


# Problem figure (fig_problem_render_2.pdf): the optimal design on the airframe, split at the root. Both semispans
# show the wing itself (the STEP surfaces, twisted to the optimum). The left semispan is undeflected and colored by
# the VLM pressure coefficient, mapped from the vortex lattice onto the surface, on a square-root color scale. The
# right semispan is deflected by the spar's displacements and rotations under the design load (true scale), over a
# translucent outline of the undeflected wing, and colored by the von Mises stress of the spar beneath it, relative
# to yield. The airframe is opaque.
DEFLECTION_SCALE = 1.0        # 1 is true scale; increase to exaggerate the deflection
COLOR_WING_BY_STRESS = True   # otherwise the deflected right semispan is a plain color
mesh = np.asarray(out["mesh"])
cp = np.asarray(cp)
cp_norm = PowerNorm(0.5, 0.0, np.ceil(cp.max() * 10) / 10)  # square-root color scale


def surface_cp(points, mesh, cp):
    """The panel pressure coefficients ``cp`` (chordwise, spanwise panels) of the vortex lattice ``mesh``
    (chordwise, spanwise, 3) at wing surface points, interpolated linearly between the panel centers in the
    chordwise fraction and the spanwise coordinate (held constant beyond the outermost centers). The VLM pressure is
    the jump across the camber surface, so the upper and lower surfaces get the same value."""
    centers = (mesh[:-1, :-1] + mesh[1:, :-1] + mesh[:-1, 1:] + mesh[1:, 1:]) / 4
    y_centers = centers[0, :, 1]
    stations = mesh[0, :, 1]
    x_le = np.interp(points[:, 1], stations, mesh[0, :, 0])
    x_te = np.interp(points[:, 1], stations, mesh[-1, :, 0])
    frac_points = (points[:, 0] - x_le) / (x_te - x_le)
    frac_centers = ((centers[:, :, 0] - mesh[0, :-1, 0]) / (mesh[-1, :-1, 0] - mesh[0, :-1, 0])).mean(axis=1)
    interpolator = RegularGridInterpolator((frac_centers, y_centers), cp)
    query = np.stack([np.clip(frac_points, frac_centers[0], frac_centers[-1]),
                      np.clip(points[:, 1], y_centers[0], y_centers[-1])], axis=-1)
    return interpolator(query)

# the airframe without its wing, the propeller blades and wheels (OpenVSP names are "<id>, <component>, ...", so
# the commas match whole component names), and the wing, twisted to the optimum like the vortex lattice
# (the nose gear bogie is spelled "nodegearbogie" in the STEP file)
colored_parts = {"#8c8c8c": [", prop,", ", gear,"],   # propeller blades and tires: gray
                 "#e8d9a0": [", nodegearbogie,"],      # nose gear bogie: pale gold
                 "silver": [", spinner,"]}
airframe, airframe_edges = sampled_mesh(
    ranger.declare_component(function_search_names=[""],
                             ignore_names=["WingGeom"] + sum(colored_parts.values(), [])),
    samples_per_span=8)
parts = {color: sampled_mesh(ranger.declare_component(function_search_names=names), samples_per_span=8)
         for color, names in colored_parts.items()}
wing, wing_edges = sampled_mesh(twist_ffd.twisted_wing(twist_cp_star), samples_per_span=8)


def beam_deflection(points, nodes, u_nodes, scale=1.0):
    """The displacement of wing surface points carried by the beam: each point moves rigidly with the beam section
    at its spanwise station, by the section's translation plus its (small) rotation about the beam axis there,
    both interpolated linearly between the nodes. ``u_nodes`` is (num_nodes, 6): translations, then rotations."""
    span = nodes[:, 1]
    yp = np.clip(points[:, 1], span[0], span[-1])
    interp = lambda values: np.stack([np.interp(yp, span, v) for v in values.T], axis=-1)
    translation, rotation, axis_point = interp(u_nodes[:, :3]), interp(u_nodes[:, 3:]), interp(nodes)
    return scale * (translation + np.cross(rotation, points - axis_point))


def semispan(polydata, right):
    """The cells of ``polydata`` on the right (y >= 0) or left (y <= 0) of the root."""
    y_cells = polydata.cell_centers().points[:, 1]
    return polydata.extract_cells(np.flatnonzero(y_cells >= 0 if right else y_cells <= 0)).extract_surface()


u_beam = np.asarray(u[0])
sigma_elem = np.asarray(frame.beams[0].stress(u[0])) / 1e6 / sigma_yield_mpa
wing_left, wing_right = semispan(wing, right=False), semispan(wing, right=True)
edges_left, edges_right = semispan(wing_edges, right=False), semispan(wing_edges, right=True)

def deflected(polydata):
    """A copy of ``polydata`` deflected by the spar."""
    polydata = polydata.copy()
    polydata.points = polydata.points + beam_deflection(polydata.points, struct_mesh, u_beam, DEFLECTION_SCALE)
    return polydata


deflected_wing = deflected(wing_right)
deflected_wing.point_data["sigma"] = np.interp(deflected_wing.points[:, 1], y_elem, sigma_elem)
deflected_edges = deflected(edges_right)

# the left semispan stays undeflected, where the surface lies over the vortex lattice. PyVista has no power-law
# color scale, so it is given sqrt(Cp) on a linear scale, which matches cp_norm in the colorbar.
wing_left.point_data["sqrt_Cp"] = np.sqrt(np.clip(surface_cp(np.asarray(wing_left.points), mesh, cp), 0.0, None))

shaded = dict(smooth_shading=True, diffuse=0.7, ambient=0.3)
plotter = pv.Plotter(off_screen=True, window_size=(2800, 1500))
plotter.set_background("white")
plotter.enable_anti_aliasing("ssaa")
plotter.enable_depth_peeling()  # sort the translucent surfaces correctly
plotter.add_mesh(airframe, color="white", **shaded)
for color, (part, _) in parts.items():
    # the spinner gets a specular highlight, to read as metal
    plotter.add_mesh(part, color=color, **shaded, **(dict(specular=0.6, specular_power=30) if color == "silver" else {}))
plotter.add_mesh(wing_right, color="white", opacity=0.15, **shaded)  # the undeflected outline
# the edges of the STEP surfaces, as in jeometry's Geometry.plot
for edges in (airframe_edges, *(part_edges for _, part_edges in parts.values())):
    plotter.add_mesh(edges, color=EDGE_COLOR, line_width=1.0, lighting=False)
plotter.add_mesh(edges_right, color=EDGE_COLOR, line_width=1.0, opacity=0.3, lighting=False)
plotter.add_mesh(wing_left, scalars="sqrt_Cp", cmap="Blues", clim=(np.sqrt(cp_norm.vmin), np.sqrt(cp_norm.vmax)),
                 show_scalar_bar=False, **shaded)
plotter.add_mesh(edges_left, color="#5a5a5a", line_width=1.2, lighting=False)
if COLOR_WING_BY_STRESS:
    plotter.add_mesh(deflected_wing, scalars="sigma", cmap="Oranges", clim=(0, 1), show_scalar_bar=False, **shaded)
else:
    plotter.add_mesh(deflected_wing, color="#f2b880", **shaded)
plotter.add_mesh(deflected_edges, color="#5a5a5a", line_width=1.2, lighting=False)

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
colorbars = [(0.58, ScalarMappable(cp_norm, "Blues"), "Pressure coefficient")]
if COLOR_WING_BY_STRESS:
    # the camera sees the right (structure) semispan on the left of the image
    colorbars.insert(0, (0.08, ScalarMappable(Normalize(0, 1), "Oranges"), "Normalized spar stress"))
with plt.rc_context({"font.family": "STIXGeneral", "font.size": 14, "xtick.labelsize": 13, "axes.linewidth": 0.6,
                     "xtick.direction": "in", "xtick.major.size": 2.5, "xtick.major.width": 0.6,
                     "pdf.fonttype": 42}):
    fig = plt.figure(figsize=(7, height + 0.55))
    ax = fig.add_axes([0, 0.55 / (height + 0.55), 1, height / (height + 0.55)])
    ax.imshow(image)
    ax.axis("off")
    for left, mappable, label in colorbars:
        fig.colorbar(mappable, cax=fig.add_axes([left, 0.06, 0.34, 0.035]), orientation="horizontal").set_label(
            label, labelpad=2)
    fig.savefig(os.path.join(HERE, "fig_problem_render_2.pdf"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    fig.savefig(os.path.join(HERE, "fig_problem_render_2.png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
