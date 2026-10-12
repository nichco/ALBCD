"""Optional PyVista visualization helpers for VLM meshes.

Not needed for the VLM solve itself (``vlm_jax.analysis``/``vlm_jax.aerodynamics``
have no pyvista dependency) -- only import this module, or call its
functions, if you want to plot a mesh. ``pyvista`` itself is imported lazily
inside each function so the rest of this package stays usable without it
installed.
"""
import numpy as np


def to_pyvista_mesh(mesh, cell_data=None):
    """Build a ``pyvista.PolyData`` quad mesh from a VLM ``(nx, ny, 3)`` node
    mesh -- one quad cell per panel, ``(nx - 1) * (ny - 1)`` total.

    Parameters
    ----------
    mesh : (nx, ny, 3) array
        A VLM node mesh, e.g. from ``vlm_jax.meshing.generate_mesh`` or
        ``vlm_jax.analysis.solve_aero``'s ``"mesh"`` output.
    cell_data : dict[str, array], optional
        Per-panel scalars to attach, each shaped ``(nx-1, ny-1)`` or
        ``((nx-1)*(ny-1),)`` -- e.g.
        ``{"Pressure (Pa)": vlm_jax.analysis.panel_pressure(...)}``. Stored as
        ``grid.cell_data[name]``.

    Returns
    -------
    pyvista.PolyData
        Cells are ordered row-major by ``(panel row i, panel column j)``,
        the same order ``sec_forces``/``panel_pressure`` use, so
        ``cell_data`` lines up without any reordering.
    """
    import pyvista as pv

    mesh_np = np.asarray(mesh)
    nx, ny, _ = mesh_np.shape
    pts = mesh_np.reshape(-1, 3)   # node (i, j) -> flat index i * ny + j

    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            p00 = i * ny + j
            p01 = i * ny + (j + 1)
            p11 = (i + 1) * ny + (j + 1)
            p10 = (i + 1) * ny + j
            faces.extend([4, p00, p01, p11, p10])

    grid = pv.PolyData(pts, np.array(faces))
    if cell_data is not None:
        for name, values in cell_data.items():
            grid.cell_data[name] = np.asarray(values).reshape(-1)
    return grid
