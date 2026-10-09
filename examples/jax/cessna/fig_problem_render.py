"""Problem figure, option C: a 3D view of the optimal wing on the Cessna 182 airframe,
split at the root: the left semispan shows the vortex lattice colored by the pressure
coefficient (the aerodynamics), and the right semispan shows the spar tube colored by its
von Mises stress at the design load, relative to yield (the structure). Rendered off
screen with PyVista, then labeled with matplotlib. Reads cessna_problem.npz (from
problem_data.py)."""

import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 14,
                     "xtick.labelsize": 13, "axes.linewidth": 0.6, "xtick.direction": "in", "pdf.fonttype": 42})

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "cessna_problem.npz"))

mesh, cp = d["mesh_opt"], d["cp_opt"]      # (nx, ny, 3) nodes, (nx-1, ny-1) panels
y, y_elem, radius = d["y"], d["y_elem"], d["radius"]
stress_ratio = d["sigma_mpa_opt"] / float(d["sigma_yield"])
cmap_aero, cmap_struct = "Blues", "Oranges"
cp_norm = Normalize(0.0, float(np.ceil(cp.max() * 10) / 10))
sigma_norm = Normalize(0.0, 1.0)


def lattice(mesh, cells):
    """PolyData of the panels (i, j) with cells[i, j] True."""
    nx, ny, _ = mesh.shape
    faces = [[4, i * ny + j, i * ny + j + 1, (i + 1) * ny + j + 1, (i + 1) * ny + j]
             for i in range(nx - 1) for j in range(ny - 1) if cells[i, j]]
    return pv.PolyData(mesh.reshape(-1, 3), np.array(faces).ravel())


panel_y = 0.25 * (mesh[:-1, :-1, 1] + mesh[1:, :-1, 1] + mesh[:-1, 1:, 1] + mesh[1:, 1:, 1])
left = panel_y < 0
aero = lattice(mesh, left)
aero.cell_data["Cp"] = cp[left]
skin = lattice(mesh, ~left)

# spar tube along the quarter chord of the right semispan, one cylinder per beam element
x_qc = mesh[0, :, 0] + 0.25 * (mesh[-1, :, 0] - mesh[0, :, 0])
z_qc = mesh[0, :, 2] + 0.25 * (mesh[-1, :, 2] - mesh[0, :, 2])
spar = []
for e in np.where(y_elem > 0)[0]:
    p0 = np.array([x_qc[e], y[e], z_qc[e]])
    p1 = np.array([x_qc[e + 1], y[e + 1], z_qc[e + 1]])
    cyl = pv.Cylinder(center=0.5 * (p0 + p1), direction=p1 - p0, radius=radius[e],
                      height=np.linalg.norm(p1 - p0), resolution=40, capping=False)
    cyl.cell_data["sigma"] = np.full(cyl.n_cells, stress_ratio[e])
    spar.append(cyl)
spar = pv.merge(spar)

# the STL is in feet; scale to meters and place it under the wing, as in cessna.py
airframe = pv.read(os.path.join(HERE, "cessna182_no_wing.stl"))
airframe = airframe.scale(0.3048, inplace=False).translate((-1.4, 0.0, -0.9), inplace=False)

pv.global_theme.font.family = "times"
plotter = pv.Plotter(off_screen=True, window_size=(2800, 1500))
plotter.set_background("white")
plotter.enable_anti_aliasing("ssaa")
plotter.add_mesh(airframe, color="white", smooth_shading=True, specular=0.0, diffuse=0.7, ambient=0.3, show_scalar_bar=False)
plotter.add_mesh(aero, scalars="Cp", cmap=cmap_aero, clim=(cp_norm.vmin, cp_norm.vmax), show_edges=True,
                 edge_color="#6f6f6f", line_width=0.6, show_scalar_bar=False, lighting=False)
plotter.add_mesh(skin, color="#f2f2f2", opacity=0.55, show_edges=True, edge_color="#9a9a9a", line_width=0.6,
                 lighting=False)
plotter.add_mesh(spar, scalars="sigma", cmap=cmap_struct, clim=(0, 1), smooth_shading=True, show_scalar_bar=False)

# camera ahead of and above the left wing, looking aft at the airframe
plotter.camera_position = "iso"
center = np.array([0.6, -0.3, -0.3])
az, el, distance = np.radians(205), np.radians(28), 14.5
direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
plotter.camera_position = [tuple(center + distance * direction), tuple(center), (0.0, 0.0, 1.0)]
plotter.camera.view_angle = 30
image = plotter.screenshot(return_img=True, transparent_background=False)
plotter.close()

# crop the white margins
rows = np.where((image < 250).any(axis=(1, 2)))[0]
cols = np.where((image < 250).any(axis=(0, 2)))[0]
pad = 20
image = image[max(rows[0] - pad, 0):rows[-1] + pad, max(cols[0] - pad, 0):cols[-1] + pad]

h, w = image.shape[:2]
fig = plt.figure(figsize=(7, 7 * h / w * 1.0 + 0.55))
ax = fig.add_axes([0, 0.55 / (7 * h / w + 0.55), 1, 7 * h / w / (7 * h / w + 0.55)])
ax.imshow(image)
ax.axis("off")

# the camera sees the right (structure) semispan on the left of the image
for left_edge, mappable, label in (
        (0.08, ScalarMappable(sigma_norm, cmap_struct), "Normalized spar stress"),
        (0.58, ScalarMappable(cp_norm, cmap_aero), "Pressure coefficient")):
    cax = fig.add_axes([left_edge, 0.06, 0.34, 0.035])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(length=2.5, width=0.6)
    cb.set_label(label, fontsize=14, labelpad=2)

fig.savefig(os.path.join(HERE, "fig_problem_render.pdf"), bbox_inches="tight", pad_inches=0.02, dpi=300)
fig.savefig(os.path.join(HERE, "fig_problem_render.png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
plt.show()
