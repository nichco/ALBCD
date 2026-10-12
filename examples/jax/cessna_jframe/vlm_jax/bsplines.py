"""B-spline interpolation matrix construction.

OpenAeroStruct maps design-variable control points (twist_cp, thickness_cp,
t_over_c_cp, ...) onto the mesh's spanwise stations with OpenMDAO's
``om.SplineComp(method="bsplines")``. That interpolation is a fixed LINEAR
operator: ``y = B @ y_cp``, where ``B`` depends only on the number of control
points, the (fixed) query locations, and the spline order -- never on the
control-point values themselves. So ``B`` can be built once with plain numpy
and then used as a constant matrix inside the JAX-differentiated graph
(``jnp.asarray(B) @ y_cp``); JAX only ever needs to differentiate the
matmul, not the matrix-construction algorithm.

``get_bspline_mtx`` below is a direct copy of the matrix-construction
algorithm in
``openmdao.components.interp_util.interp_bsplines.InterpBSplines.get_bspline_mtx``
(itself following Hwang & Martins, "GeoMACH", AIAA 2012-5605) -- reproduced
here so this package has no runtime dependency on OpenMDAO.
"""
import numpy as np


def get_bspline_mtx(num_cp, t_vec, order=4):
    """Build the dense B-spline evaluation matrix mapping control points to
    interpolated values at ``t_vec``.

    Parameters
    ----------
    num_cp : int
        Number of control points.
    t_vec : (num_pt,) ndarray
        Query locations, normalized to [0, 1].
    order : int
        B-spline order (degree + 1). OpenAeroStruct always uses
        ``min(num_cp, 4)``.

    Returns
    -------
    (num_pt, num_cp) ndarray
        Dense matrix ``B`` such that ``y_interp = B @ y_cp``.
    """
    knots = np.zeros(num_cp + order)
    knots[order - 1 : num_cp + 1] = np.linspace(0, 1, num_cp - order + 2)
    knots[num_cp + 1 :] = 1.0

    basis = np.zeros(order)
    arange = np.arange(order)

    num_pt = len(t_vec)
    mtx = np.zeros((num_pt, num_cp))

    for ipt in range(num_pt):
        t = t_vec[ipt]

        i0 = -1
        if t == knots[-1]:
            i0 = num_cp - order
        else:
            for ind in range(order, num_cp + 1):
                if knots[ind - 1] <= t < knots[ind]:
                    i0 = ind - order
                    break

        basis[:] = 0.0
        basis[-1] = 1.0

        for i in range(2, order + 1):
            ll = i - 1
            j1 = order - ll
            j2 = order
            n = i0 + j1

            if knots[n + ll] != knots[n]:
                basis[j1 - 1] = (knots[n + ll] - t) / (knots[n + ll] - knots[n]) * basis[j1]
            else:
                basis[j1 - 1] = 0.0

            for j in range(j1 + 1, j2):
                n = i0 + j

                if knots[n + ll - 1] != knots[n - 1]:
                    basis[j - 1] = (t - knots[n - 1]) / (knots[n + ll - 1] - knots[n - 1]) * basis[j - 1]
                else:
                    basis[j - 1] = 0.0

                if knots[n + ll] != knots[n]:
                    basis[j - 1] += (knots[n + ll] - t) / (knots[n + ll] - knots[n]) * basis[j]

            n = i0 + j2
            if knots[n + ll - 1] != knots[n - 1]:
                basis[j2 - 1] = (t - knots[n - 1]) / (knots[n + ll - 1] - knots[n - 1]) * basis[j2 - 1]
            else:
                basis[j2 - 1] = 0.0

        mtx[ipt, i0 + arange] = basis

    return mtx


def get_normalized_span_coords(mesh, mid_panel=False):
    """Normalized [0, 1] spanwise coordinate of each mesh node (or panel
    midpoint), tip=0 to root=1. Matches
    ``openaerostruct.utils.interpolation.get_normalized_span_coords``.
    """
    spanwise_coord = mesh[0, :, 1]
    span_range = spanwise_coord[-1] - spanwise_coord[0]
    span_offset = spanwise_coord[0]
    if mid_panel:
        x_real = (spanwise_coord[:-1] + spanwise_coord[1:]) / 2
    else:
        x_real = spanwise_coord
    return (x_real - span_offset) / span_range


def bspline_interp_mtx(mesh, num_cp, mid_panel=False, x_cp_start=None, x_cp_end=None):
    """Build the constant B-spline matrix mapping ``num_cp`` control points to
    the mesh's spanwise stations (or panel midpoints if ``mid_panel=True``),
    matching how ``openaerostruct.geometry.geometry_group.Geometry`` and
    ``openaerostruct.structures.tube_group.TubeGroup`` configure their
    ``om.SplineComp`` instances.
    """
    x_interp = get_normalized_span_coords(mesh, mid_panel=mid_panel)
    order = min(num_cp, 4)
    # Matches InterpBSplines.evaluate_vectorized's "map onto [0, 1]" step exactly.
    start = x_cp_start if x_cp_start is not None else x_interp[0]
    end = x_cp_end if x_cp_end is not None else x_interp[-1]
    scale = end - start
    shift = min(start, end)
    x_mapped = (x_interp - shift) / scale
    return get_bspline_mtx(num_cp, x_mapped, order=order)
