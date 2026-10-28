"""Load and plot the Vashon Ranger R7 STL with PyVista.

The STL is in millimeters, with the span along x, the fuselage along y and z up.
It is scaled to meters here.
"""

import os
import numpy as np
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__))

airframe = pv.read(os.path.join(HERE, "Vashon_Ranger_R7.stl"))
airframe = airframe.scale(1e-3, inplace=False)  # mm -> m

x_min, x_max, y_min, y_max, z_min, z_max = airframe.bounds
print(airframe)
print(f"span   (x): {x_max - x_min:.2f} m")
print(f"length (y): {y_max - y_min:.2f} m")
print(f"height (z): {z_max - z_min:.2f} m")

plotter = pv.Plotter(window_size=(1600, 1000))
plotter.set_background("white")
plotter.enable_anti_aliasing("ssaa")
plotter.add_mesh(airframe, color="white", smooth_shading=True, specular=0.2, diffuse=0.7, ambient=0.3)
plotter.show_axes()

# camera ahead of (+y is forward), beside and above the aircraft, looking at its center
center = np.array(airframe.center)
az, el, distance = np.radians(55), np.radians(22), 12.5
direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
plotter.camera_position = [tuple(center + distance * direction), tuple(center), (0.0, 0.0, 1.0)]
plotter.show()
