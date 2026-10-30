"""Problem figure: a 3D view of the optimal wing on the Vashon Ranger R7 airframe, split at
the root: the left semispan shows the vortex lattice colored by the pressure coefficient
(the aerodynamics), and the right semispan shows the square-tube wing box colored by its
von Mises stress at the design load, relative to yield (the structure). The wing box is
drawn deflected under the design load (true scale), over an outline of the undeflected box. Rendered off
screen with PyVista, then labeled with matplotlib. Reads ranger_problem.npz (from
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
d = np.load(os.path.join(HERE, "ranger_problem.npz"))

mesh, cp = d["mesh_opt"], d["cp_opt"]           # (nx, ny, 3) nodes, (nx-1, ny-1) panels
box_mesh, h = d["box_mesh"], float(d["box_side"])  # wing box nodes and outer side
w = d["w_opt"]                                   # vertical deflection of the box nodes at the design load
box_stress = d["sigma_mpa_opt"] / float(d["sigma_yield"])
cmap_aero, cmap_struct = "Blues", "Oranges"
cp_norm = Normalize(0.0, float(np.ceil(cp.max() * 10) / 10))
sigma_norm = Normalize(0.0, 1.0)


def lattice(mesh, cells):
    """PolyData of the panels (i, j) with cells[i, j] True."""
    nx, ny, _ = mesh.shape
    faces = [[4, i * ny + j, i * ny + j + 1, (i + 1) * ny + j + 1, (i + 1) * ny + j]
             for i in range(nx - 1) for j in range(ny - 1) if cells[i, j]]
    return pv.PolyData(mesh.reshape(-1, 3), np.array(faces).ravel())


def swept_tube(nodes, side, stress):
    """A square tube through the spanwise nodes (its section kept level), one segment per element
    colored by its stress."""
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * side / 2
    points = np.array([[p[0] + cx, p[1], p[2] + cz] for p in nodes for cx, cz in corners])
    faces, values = [], []
    for k, s in enumerate(stress):
        a, b = 4 * k, 4 * (k + 1)
        for m in range(4):
            faces.append([4, a + m, a + (m + 1) % 4, b + (m + 1) % 4, b + m])
            values.append(s)
    faces += [[4, 0, 1, 2, 3], [4, *(4 * (len(nodes) - 1) + np.arange(4))]]  # end caps
    values += [stress[0], stress[-1]]
    tube = pv.PolyData(points, np.array(faces).ravel())
    tube.cell_data["sigma"] = np.array(values)
    return tube


panel_y = 0.25 * (mesh[:-1, :-1, 1] + mesh[1:, :-1, 1] + mesh[:-1, 1:, 1] + mesh[1:, 1:, 1])
left = panel_y < 0
aero = lattice(mesh, left)
aero.cell_data["Cp"] = cp[left]
skin = lattice(mesh, ~left)

# the right semispan of the wing box, from the centerline to the tip, deflected under the design load
# (true scale), and a ghost of it undeflected, drawn slightly smaller so that near the root, where the
# two nearly coincide, it stays inside the deflected box instead of flickering through its faces
right_nodes = box_mesh[:, 1] >= 0
right_elems = right_nodes[:-1] & right_nodes[1:]   # both end nodes on the right semispan
nodes = box_mesh[right_nodes]
box = swept_tube(nodes + np.outer(w[right_nodes], [0.0, 0.0, 1.0]), h, box_stress[right_elems])
g = 0.97 * h / 2
undeflected = pv.Box(bounds=(nodes[0, 0] - g, nodes[0, 0] + g, nodes[0, 1], nodes[-1, 1], -g, g))

# The STL is in millimeters, with the span along x, the nose toward +y and z up. As in ranger.py: scale
# it to meters, cut away its wing panels outboard of the cabin (drawn here from the models instead),
# turn it so the span is along y and the nose toward -x, and move its wing's leading edge and mid-depth
# to the VLM mesh's
airframe = pv.read(os.path.join(HERE, "Vashon_Ranger_R7.stl")).scale(1e-3, inplace=False)
for side in (1, -1):
    airframe = airframe.clip_box(sorted([side * 0.62, side * 4.6]) + [-3.15, -0.8, 0.4, 1.15], invert=True)
airframe = airframe.rotate_z(90, inplace=False).translate((-1.43, 0.0, -0.68), inplace=False)
# trim the top of the wing center section over the cabin down to the wing's mid-depth, so the roots
# of the vortex lattice and the wing box show through it
airframe = airframe.clip_box([-0.15, 1.55, -0.63, 0.63, 0.0, 1.0], invert=True)

pv.global_theme.font.family = "times"
plotter = pv.Plotter(off_screen=True, window_size=(2800, 1500))
plotter.set_background("white")
plotter.enable_anti_aliasing("ssaa")
plotter.enable_depth_peeling()  # sort the translucent skin and undeflected box correctly
plotter.add_mesh(airframe, color="white", smooth_shading=True, specular=0.0, diffuse=0.7, ambient=0.3, show_scalar_bar=False, opacity=1.0)
plotter.add_mesh(aero, scalars="Cp", cmap=cmap_aero, clim=(cp_norm.vmin, cp_norm.vmax), show_edges=True,
                 edge_color="#6f6f6f", line_width=0.6, show_scalar_bar=False, lighting=False)
plotter.add_mesh(skin, color="#f2f2f2", opacity=0.55, show_edges=True, edge_color="#9a9a9a", line_width=0.6,
                 lighting=False)
plotter.add_mesh(undeflected, color="#9a9a9a", opacity=0.35, show_edges=True, edge_color="#5a5a5a", line_width=1.0,
                 lighting=False)
plotter.add_mesh(box, scalars="sigma", cmap=cmap_struct, clim=(0, 1), show_scalar_bar=False)

# camera ahead of and above the left wing, looking aft at the airframe
plotter.camera_position = "iso"
center = np.array([0.9, -0.3, -0.2])
az, el, distance = np.radians(205), np.radians(28), 12.5
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

h_img, w_img = image.shape[:2]
fig = plt.figure(figsize=(7, 7 * h_img / w_img + 0.55))
ax = fig.add_axes([0, 0.55 / (7 * h_img / w_img + 0.55), 1, 7 * h_img / w_img / (7 * h_img / w_img + 0.55)])
ax.imshow(image)
ax.axis("off")

# the camera sees the right (structure) semispan on the left of the image
for left_edge, mappable, label in (
        (0.08, ScalarMappable(sigma_norm, cmap_struct), "Normalized wing box stress"),
        (0.58, ScalarMappable(cp_norm, cmap_aero), "Pressure coefficient")):
    cax = fig.add_axes([left_edge, 0.06, 0.34, 0.035])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(length=2.5, width=0.6)
    cb.set_label(label, fontsize=14, labelpad=2)

fig.savefig(os.path.join(HERE, "fig_problem_render.pdf"), bbox_inches="tight", pad_inches=0.02, dpi=300)
fig.savefig(os.path.join(HERE, "fig_problem_render.png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
plt.show()
