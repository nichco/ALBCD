"""Mesh geometry manipulation: control points -> deformed aerodynamic mesh.

Ports ``openaerostruct.geometry.geometry_group.Geometry`` /
``openaerostruct.geometry.geometry_mesh.GeometryMesh`` for the specific
surface configuration used by ``run_uCRM.py``: only ``twist_cp`` (and
``t_over_c_cp``, handled separately in ``structures.py``/``aerodynamics.py``
since it never touches the mesh) is an active geometric design variable.

``GeometryMesh`` chains nine mesh-transform components in this order: Taper,
ScaleX, Sweep, ShearX, Stretch, ShearY, Dihedral, ShearZ, Rotate. Given this
surface dict (no ``taper``/``chord_cp``/``sweep``/``xshear_cp``/``span``/
``yshear_cp``/``dihedral``/``zshear_cp`` keys), the first eight components
each run with the parameter value that makes them an exact numerical
no-op (taper ratio 1.0, chord scaler 1.0, sweep/shear/dihedral 0.0, and a
"stretch" target span equal to the mesh's own current span) -- verified by
inspecting ``openaerostruct/geometry/geometry_mesh_transformations.py``.
Only ``Rotate`` (applying ``twist``) has a visible effect here, so that is
the only transform this module implements.
"""
import numpy as np
import torch

from vlm_torch.bsplines import bspline_interp_mtx


def build_twist_bspline_matrix(mesh, num_twist_cp):
    """Constant matrix mapping ``twist_cp`` (deg) to per-node ``twist`` (deg).

    Matches the ``twist_bsp`` ``om.SplineComp`` built in
    ``Geometry.setup()`` (``geometry_group.py:88-96``): order
    ``min(num_twist_cp, 4)``, queried at the mesh's own normalized spanwise
    node locations (no ``mid_panel``, no custom ``x_cp_start``/``x_cp_end``).
    """
    return torch.as_tensor(bspline_interp_mtx(mesh, num_twist_cp, mid_panel=False))


def compute_twist(twist_cp, b_twist):
    """``twist[ny] = B_twist @ twist_cp``, degrees. Linear, so this is the
    entire "derivative" of the twist B-spline -- no need for anything else."""
    return b_twist @ twist_cp


def apply_twist_rotation(mesh, twist, ref_axis_pos=0.25, symmetry=True, rotate_x=True):
    """Port of ``Rotate.compute()``
    (``geometry_mesh_transformations.py:1247-1321``): twists the mesh about
    the reference-axis line (``ref_axis_pos`` fraction of chord, quarter-chord
    by default). When ``rotate_x`` is True (OpenAeroStruct's default), an
    additional rotation about the x-axis is layered on, derived from the
    z-displacement of the reference axis between adjacent spanwise stations,
    so that twist is applied perpendicular to a dihedraled/curved wing rather
    than strictly about the global y-axis.

    Parameters
    ----------
    mesh : (nx, ny, 3) array
        Mesh before twist.
    twist : (ny,) array
        Twist angle at each spanwise station, degrees (root/tip convention
        matches the mesh's own node ordering).

    Returns
    -------
    (nx, ny, 3) array
    """
    te = mesh[-1]
    le = mesh[0]
    ref_axis = ref_axis_pos * te + (1 - ref_axis_pos) * le  # (ny, 3)

    ny = mesh.shape[1]

    if rotate_x:
        if symmetry:
            dz_qc = ref_axis[:-1, 2] - ref_axis[1:, 2]
            dy_qc = ref_axis[:-1, 1] - ref_axis[1:, 1]
            theta_x = torch.arctan(dz_qc / dy_qc)
            # Root (last station) is not rotated.
            rad_theta_x = torch.cat([theta_x, torch.zeros(1, dtype=mesh.dtype)])
        else:
            root_index = (ny - 1) // 2
            dz_qc_left = ref_axis[:root_index, 2] - ref_axis[1 : root_index + 1, 2]
            dy_qc_left = ref_axis[:root_index, 1] - ref_axis[1 : root_index + 1, 1]
            theta_x_left = torch.arctan(dz_qc_left / dy_qc_left)
            dz_qc_right = ref_axis[root_index + 1 :, 2] - ref_axis[root_index:-1, 2]
            dy_qc_right = ref_axis[root_index + 1 :, 1] - ref_axis[root_index:-1, 1]
            theta_x_right = torch.arctan(dz_qc_right / dy_qc_right)
            rad_theta_x = torch.cat([theta_x_left, torch.zeros(1, dtype=mesh.dtype), theta_x_right])
    else:
        rad_theta_x = torch.zeros(ny, dtype=mesh.dtype)

    rad_theta_y = twist * torch.pi / 180.0

    cos_rtx = torch.cos(rad_theta_x)
    cos_rty = torch.cos(rad_theta_y)
    sin_rtx = torch.sin(rad_theta_x)
    sin_rty = torch.sin(rad_theta_y)

    # Per-station rotation matrix R = Rx(theta_x) @ Ry(theta_y).
    mats = torch.zeros((ny, 3, 3), dtype=mesh.dtype)
    mats[:, 0, 0] = cos_rty
    mats[:, 0, 2] = sin_rty
    mats[:, 1, 0] = sin_rtx * sin_rty
    mats[:, 1, 1] = cos_rtx
    mats[:, 1, 2] = -sin_rtx * cos_rty
    mats[:, 2, 0] = -cos_rtx * sin_rty
    mats[:, 2, 1] = sin_rtx
    mats[:, 2, 2] = cos_rtx * cos_rty

    return torch.einsum("ikj,mij->mik", mats, mesh - ref_axis) + ref_axis


def compute_mesh(mesh0, twist_cp, b_twist, ref_axis_pos=0.25, symmetry=True):
    """Full geometry pipeline: control points -> deformed (twisted) mesh.

    ``mesh0`` is the fixed baseline mesh from ``generate_mesh`` (a plain
    numpy constant, not differentiated). ``twist_cp`` is the only active
    geometric design variable for this problem.
    """
    twist = compute_twist(twist_cp, b_twist)
    return apply_twist_rotation(
        torch.as_tensor(mesh0, dtype=torch.get_default_dtype()),
        twist, ref_axis_pos=ref_axis_pos, symmetry=symmetry, rotate_x=True,
    )
