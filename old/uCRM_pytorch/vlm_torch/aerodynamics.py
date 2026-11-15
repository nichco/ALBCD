"""Vortex-lattice aerodynamics: deformed mesh -> circulations -> panel forces
-> CL/CD.

Ports (single-surface, symmetry=True, no ground effect, no rotational
velocities, incompressible i.e. non-Prandtl-Glauert states):
  openaerostruct/aerodynamics/geometry.py             (VLMGeometry)
  openaerostruct/aerodynamics/collocation_points.py    (CollocationPoints)
  openaerostruct/aerodynamics/vortex_mesh.py           (VortexMesh)
  openaerostruct/aerodynamics/get_vectors.py           (GetVectors)
  openaerostruct/aerodynamics/eval_mtx.py              (EvalVelMtx, Biot-Savart AIC)
  openaerostruct/aerodynamics/mtx_rhs.py               (VLMMtxRHSComp)
  openaerostruct/aerodynamics/solve_matrix.py          (SolveMatrix -- a single
                                                          direct linear solve,
                                                          not iterative)
  openaerostruct/aerodynamics/horseshoe_circulations.py
  openaerostruct/aerodynamics/eval_velocities.py       (EvalVelocities)
  openaerostruct/aerodynamics/panel_forces.py          (PanelForces)
  openaerostruct/aerodynamics/convert_velocity.py      (ConvertVelocity)
  openaerostruct/aerodynamics/functionals.py           (VLMFunctionals)
  openaerostruct/aerodynamics/lift_coeff_2D.py, lift_drag.py, coeffs.py,
    total_lift.py, viscous_drag.py, wave_drag.py (with_wave=False -> CDw=0),
    total_drag.py

Three torch-vs-jax API differences show up repeatedly below:

* ``torch.linalg.cross`` (not ``torch.cross``) is used everywhere, because
  only the former defaults to ``dim=-1`` the way ``jnp.cross`` does;
  ``torch.cross``'s legacy default picks the first size-3 dimension instead,
  which silently cross-products the wrong axis.
* Torch tensors cannot be sliced with a negative step, so JAX's ``[..., ::-1,
  ...]`` span reversals become ``torch.flip(..., dims=[axis])``.
* Torch has no functional ``.at[idx].set()/.add()``; the equivalent is
  ``clone()`` followed by ordinary indexed assignment, which autograd tracks
  through ``index_put_``. Cloning first is what keeps these updates
  out-of-place (and safe to differentiate) exactly as in the JAX original.
"""
import torch

_TOL = 1e-10


def _as_tensor(x):
    """Promote a Python/numpy scalar (or array) to a tensor, passing existing
    tensors through untouched.

    ``jnp.cos(0.5)`` works on a bare Python float; ``torch.cos(0.5)`` raises
    ``TypeError``. Flight-condition inputs (``alpha``, ``beta``) are routinely
    passed as plain floats here, so they get promoted at the point of use.
    Tensors pass through unchanged, so a ``requires_grad=True`` ``alpha``
    stays differentiable.
    """
    if torch.is_tensor(x):
        return x
    return torch.as_tensor(x, dtype=torch.get_default_dtype())


def _compute_finite_vortex(r1, r2, r1_norm=None, r2_norm=None):
    """Biot-Savart contribution of a finite straight vortex filament from
    r1 to r2 (both are vectors FROM the filament endpoint TO the field
    point). Port of ``eval_mtx.py::_compute_finite_vortex``.

    The near-singular (``den`` -> 0, field point on the filament's line)
    case is guarded by substituting a safe denominator *before* dividing,
    rather than only gating the final `torch.where`'s selected value: like
    JAX's, torch's `where` still evaluates (and back-propagates through)
    both branches, so a literal `num / den` in the discarded branch can
    still produce a NaN gradient -- even though its forward value is never
    selected -- via `0 * nan/inf = nan` in the reverse pass.

    ``r1_norm``/``r2_norm`` are optional precomputed ``|r1|``/``|r2|``
    (shape ``(..., 1)``). ``eval_vel_mtx`` calls this on four *overlapping*
    slices of one array, so it passes the norms in rather than letting every
    call recompute them -- see the note there. Left as ``None`` (the
    standalone case) they are computed here as before.
    """
    if r1_norm is None:
        r1_norm = torch.linalg.norm(r1, dim=-1, keepdim=True)
    if r2_norm is None:
        r2_norm = torch.linalg.norm(r2, dim=-1, keepdim=True)
    r1_x_r2 = torch.linalg.cross(r1, r2)
    r1_d_r2 = torch.sum(r1 * r2, dim=-1, keepdim=True)
    den = r1_norm * r2_norm + r1_d_r2
    singular = torch.abs(den) <= _TOL
    safe_den = torch.where(singular, 1.0, den)
    # Fold the whole scalar factor into one (..., 1) coefficient applied once
    # to the (..., 3) cross product. Written as `num / (safe_den * 4 * pi)`
    # with `num = (1/|r1| + 1/|r2|) * r1_x_r2` this costs three divides per
    # element; as a coefficient it costs one. Same expression, fewer divides.
    coeff = (1.0 / r1_norm + 1.0 / r2_norm) / (safe_den * (4 * torch.pi))
    return torch.where(singular, 0.0, coeff * r1_x_r2)


def _compute_semi_infinite_vortex(u, r, r_norm=None):
    """Port of ``eval_mtx.py::_compute_semi_infinite_vortex``: trailing leg
    from the field point along direction ``u`` to infinity. ``r_norm`` is an
    optional precomputed ``|r|``, as in ``_compute_finite_vortex``."""
    if r_norm is None:
        r_norm = torch.linalg.norm(r, dim=-1, keepdim=True)
    u_x_r = torch.linalg.cross(u, r)
    u_d_r = torch.sum(u * r, dim=-1, keepdim=True)
    den = r_norm * (r_norm - u_d_r)
    # `u_x_r / den / 4 / pi` is nine divides per element (three per `/`);
    # one reciprocal and a multiply is the same value for far less work.
    return u_x_r * (1.0 / (den * (4 * torch.pi)))


def vlm_geometry(def_mesh, S_ref_type="wetted", symmetry=True):
    """Port of ``VLMGeometry.compute()``.

    Returns
    -------
    b_pts, normals, chords, widths, lengths_spanwise, lengths, S_ref
    """
    mesh = def_mesh
    b_pts = mesh[:-1, :, :] * 0.75 + mesh[1:, :, :] * 0.25

    quarter_chord = 0.25 * mesh[-1] + 0.75 * mesh[0]
    lengths_spanwise = torch.linalg.norm(quarter_chord[1:, :] - quarter_chord[:-1, :], dim=1)
    widths = torch.linalg.norm(quarter_chord[1:, [1, 2]] - quarter_chord[:-1, [1, 2]], dim=1)

    dx = mesh[1:, :, 0] - mesh[:-1, :, 0]
    dy = mesh[1:, :, 1] - mesh[:-1, :, 1]
    dz = mesh[1:, :, 2] - mesh[:-1, :, 2]
    lengths = torch.sum(torch.sqrt(dx**2 + dy**2 + dz**2), dim=0)

    normals_raw = torch.linalg.cross(mesh[:-1, 1:, :] - mesh[1:, :-1, :], mesh[:-1, :-1, :] - mesh[1:, 1:, :], dim=2)
    norms = torch.sqrt(torch.sum(normals_raw**2, dim=2))
    normals = normals_raw / norms[:, :, None]

    if S_ref_type == "wetted":
        S_ref = 0.5 * torch.sum(norms)
    elif S_ref_type == "projected":
        proj_mesh = mesh.clone()
        proj_mesh[:, :, 2] = 0.0
        proj_normals = torch.linalg.cross(
            proj_mesh[:-1, 1:, :] - proj_mesh[1:, :-1, :], proj_mesh[:-1, :-1, :] - proj_mesh[1:, 1:, :], dim=2
        )
        proj_norms = torch.sqrt(torch.sum(proj_normals**2, dim=2))
        S_ref = 0.5 * torch.sum(proj_norms)
    else:
        raise ValueError(f"Unknown S_ref_type {S_ref_type!r}")

    if symmetry:
        S_ref = S_ref * 2

    chords = torch.linalg.norm(mesh[0, :, :] - mesh[-1, :, :], dim=1)

    return b_pts, normals, chords, widths, lengths_spanwise, lengths, S_ref


def collocation_points(def_mesh):
    """Port of ``CollocationPoints.compute()``. Returns flattened
    ``(coll_pts, force_pts, bound_vecs)``, each ``((nx-1)*(ny-1), 3)``."""
    mesh = def_mesh
    coll_pts = (
        0.25 * 0.5 * mesh[:-1, :-1, :]
        + 0.75 * 0.5 * mesh[1:, :-1, :]
        + 0.25 * 0.5 * mesh[:-1, 1:, :]
        + 0.75 * 0.5 * mesh[1:, 1:, :]
    ).reshape(-1, 3)
    force_pts = (
        0.75 * 0.5 * mesh[:-1, :-1, :]
        + 0.25 * 0.5 * mesh[1:, :-1, :]
        + 0.75 * 0.5 * mesh[:-1, 1:, :]
        + 0.25 * 0.5 * mesh[1:, 1:, :]
    ).reshape(-1, 3)
    bound_vecs = (
        0.75 * mesh[:-1, :-1, :] + 0.25 * mesh[1:, :-1, :] - 0.75 * mesh[:-1, 1:, :] - 0.25 * mesh[1:, 1:, :]
    ).reshape(-1, 3)
    return coll_pts, force_pts, bound_vecs


def left_wing_from_mesh(mesh0):
    """Determine mirroring convention once from the (fixed) base mesh, as
    ``VortexMesh``/``EvalVelMtx`` do at trace time. This depends only on
    which spanwise index is closer to the tip vs. the symmetry plane -- a
    topological property of the mesh that twist does not change -- so it is
    safe to resolve with plain Python/numpy on the static base mesh rather
    than inside the traced autograd graph."""
    import numpy as np

    return bool(np.abs(mesh0[0, 0, 1]) > np.abs(mesh0[0, -1, 1]))


def compute_vortex_mesh(def_mesh, left_wing, symmetry=True):
    """Port of ``VortexMesh.compute()`` (symmetry=True, no ground effect)."""
    if symmetry:
        mirror = torch.as_tensor([1.0, -1.0, 1.0], dtype=def_mesh.dtype)
        if left_wing:
            mirrored = torch.flip(def_mesh[:, :-1, :], dims=[1]) * mirror
            full = torch.cat([def_mesh, mirrored], dim=1)
        else:
            mirrored = torch.flip(def_mesh[:, 1:, :], dims=[1]) * mirror
            full = torch.cat([mirrored, def_mesh], dim=1)
    else:
        full = def_mesh

    vmesh = torch.zeros_like(full)
    vmesh[:-1, :, :] = 0.75 * full[:-1, :, :] + 0.25 * full[1:, :, :]
    vmesh[-1, :, :] = full[-1, :, :]
    return vmesh


def get_vectors(vortex_mesh, eval_pts):
    """Port of ``GetVectors.compute()`` (symmetry, no ground effect branch):
    vector from every vortex-mesh node to every evaluation point."""
    return eval_pts[:, None, None, :] - vortex_mesh[None, :, :, :]


def eval_vel_mtx(vectors, alpha_deg, ny, symmetry=True, right_wing=False):
    """Port of ``EvalVelMtx.compute()``: assembles the AIC (unit-circulation
    induced-velocity) matrix from vortex rings, folded for symmetry, with the
    trailing-edge row corrected into true horseshoes (finite bound + two
    semi-infinite trailing legs). No ground effect (single pass, vortex
    multiplier 1.0).

    Parameters
    ----------
    vectors : (num_eval, nx, ny_full, 3)
        From ``get_vectors``, using the vortex mesh (``ny_full = 2*ny-1``
        when ``symmetry=True``).
    ny : int
        Number of spanwise nodes in the *half-wing* mesh (not the mirrored
        vortex mesh).

    Returns
    -------
    (num_eval, nx-1, ny-1, 3) array, units 1/m.
    """
    alpha = _as_tensor(alpha_deg) * torch.pi / 180.0
    cosa = torch.cos(alpha)
    sina = torch.sin(alpha)
    u_dir = torch.stack([cosa, torch.zeros_like(cosa), sina])

    # |r| for every vortex-mesh-node -> eval-point vector, computed ONCE on the
    # whole array. vert_A..vert_D below are four overlapping slices of
    # `vectors`, and each ring leg needs the norm at both of its endpoints, so
    # every vertex's norm is needed twice; letting each _compute_finite_vortex
    # call compute its own would evaluate eight slice-sized norms (with their
    # square roots) where one array-sized norm covers all of them -- roughly a
    # 7x reduction in that work, and it is the dominant cost in this function.
    norms = torch.linalg.norm(vectors, dim=-1, keepdim=True)

    vert_A, norm_A = vectors[:, 0:-1, 1:, :], norms[:, 0:-1, 1:, :]
    vert_B, norm_B = vectors[:, 0:-1, 0:-1, :], norms[:, 0:-1, 0:-1, :]
    vert_C, norm_C = vectors[:, 1:, 0:-1, :], norms[:, 1:, 0:-1, :]
    vert_D, norm_D = vectors[:, 1:, 1:, :], norms[:, 1:, 1:, :]

    # kept separately: the trailing-edge correction below reuses its last row
    rear = _compute_finite_vortex(vert_C, vert_D, norm_C, norm_D)

    result = (
        _compute_finite_vortex(vert_A, vert_B, norm_A, norm_B)  # front (bound, quarter-chord)
        + _compute_finite_vortex(vert_B, vert_C, norm_B, norm_C)  # right trailing leg (near-field)
        + rear  # rear (TE-closing artifact; cancelled below)
        + _compute_finite_vortex(vert_D, vert_A, norm_D, norm_A)  # left trailing leg (near-field)
    )

    if symmetry:
        vel_mtx = result[:, :, : ny - 1, :] + torch.flip(result[:, :, ny - 1 :, :], dims=[2])
    else:
        vel_mtx = result

    vert_D_last = vert_D[:, -1:, :, :]
    vert_C_last = vert_C[:, -1:, :, :]
    u = torch.broadcast_to(u_dir, vert_D_last.shape)

    # Swapping a filament's endpoints negates its cross product and leaves
    # `den` (symmetric in r1, r2) alone, singular guard included, so
    # _compute_finite_vortex(D, C) == -_compute_finite_vortex(C, D) exactly.
    # The bound leg of the trailing-edge horseshoe is therefore just the ring's
    # already-computed "rear" term on the last chordwise row, negated -- no
    # need to evaluate that filament a second time.
    lr1 = -rear[:, -1:, :, :]  # cancels the "rear" artifact above
    lr2 = _compute_semi_infinite_vortex(u, vert_D_last, norm_D[:, -1:, :, :])
    lr3 = _compute_semi_infinite_vortex(u, vert_C_last, norm_C[:, -1:, :, :])

    if symmetry:
        res1 = lr1[:, :, : ny - 1, :] + torch.flip(lr1[:, :, ny - 1 :, :], dims=[2])
        res2 = lr2[:, :, : ny - 1, :] + torch.flip(lr2[:, :, ny - 1 :, :], dims=[2])
        res3 = lr3[:, :, : ny - 1, :] + torch.flip(lr3[:, :, ny - 1 :, :], dims=[2])
        last_row = res1 - res2 + res3
    else:
        last_row = lr1 - lr2 + lr3

    vel_mtx = vel_mtx.clone()
    vel_mtx[:, -1:, :, :] += last_row

    if symmetry and right_wing:
        vel_mtx = torch.flip(vel_mtx, dims=[2])

    return vel_mtx


def build_mtx_rhs(vel_mtx_coll, normals, freestream_velocities):
    """Port of ``VLMMtxRHSComp.compute()`` (single surface).

    ``mtx[i, j]`` = velocity induced at collocation point ``i`` by panel
    ``j``'s unit circulation, dotted with the normal AT COLLOCATION POINT
    ``i`` (i.e. panel ``i``'s own normal, since collocation point i sits on
    panel i) -- NOT panel j's normal. Source: ``np.einsum("ijk,ik->ij",
    mtx_n_n_3, normals_n_3)`` -- both operands indexed by ``i``, not ``j``.
    """
    system_size = vel_mtx_coll.shape[0]
    vel_flat = vel_mtx_coll.reshape(system_size, -1, 3)
    normals_flat = normals.reshape(-1, 3)
    mtx = torch.einsum("ijk,ik->ij", vel_flat, normals_flat)
    rhs = -torch.sum(freestream_velocities * normals_flat, dim=-1)
    return mtx, rhs


def solve_circulations(mtx, rhs):
    """Port of ``SolveMatrix``: a single direct linear solve (the source uses
    dense LU via ``scipy.linalg.lu_factor``/``lu_solve``; this is exactly
    ``mtx^-1 @ rhs``, not iterative)."""
    return torch.linalg.solve(mtx, rhs)


def compute_horseshoe_circulations(circulations, nx, ny):
    """Port of ``HorseshoeCirculations``: chordwise-adjacent ring circulation
    differencing (identity for ``nx-1==1``, i.e. a single chordwise panel)."""
    arr = circulations.reshape(nx - 1, ny - 1)
    horseshoe = arr.clone()
    horseshoe[1:, :] -= arr[:-1, :]
    return horseshoe.reshape(-1)


def eval_velocities(freestream_velocities, circulations, vel_mtx_force):
    """Port of ``EvalVelocities.compute()``: total velocity (freestream +
    induced) at the force points, using the ring ``circulations`` (not the
    horseshoe ones)."""
    num_eval = vel_mtx_force.shape[0]
    vel_flat = vel_mtx_force.reshape(num_eval, -1, 3)
    return freestream_velocities + torch.einsum("ijk,j->ik", vel_flat, circulations)


def compute_panel_forces(rho, horseshoe_circulations, force_pts_velocities, bound_vecs):
    """Port of ``PanelForces.compute()``: Kutta-Joukowski, ``F = rho * Gamma *
    (V x l_bound)``."""
    return rho * horseshoe_circulations[:, None] * torch.linalg.cross(force_pts_velocities, bound_vecs)


def convert_velocity(v, alpha_deg, beta_deg, system_size):
    """Port of ``ConvertVelocity.compute()`` (non-rotational)."""
    alpha = _as_tensor(alpha_deg) * torch.pi / 180.0
    beta = _as_tensor(beta_deg) * torch.pi / 180.0
    cosa, sina = torch.cos(alpha), torch.sin(alpha)
    cosb, sinb = torch.cos(beta), torch.sin(beta)
    v_inf = v * torch.stack([cosa * cosb, -sinb, sina * cosb])
    return torch.broadcast_to(v_inf, (system_size, 3))


# ---------------------------------------------------------------------------
# VLM functionals: sec_forces -> Cl, L, D, CL1, CDi, CL, CDv, CDw, CD
# ---------------------------------------------------------------------------


def lift_coeff_2d(sec_forces, alpha_deg, widths, chords, v, rho):
    """Port of ``LiftCoeff2D.compute()``: spanwise sectional lift coefficient."""
    alpha = _as_tensor(alpha_deg) * torch.pi / 180.0
    cosa, sina = torch.cos(alpha), torch.sin(alpha)
    forces = torch.sum(sec_forces, dim=0)  # sum over chordwise axis, (ny-1,3)
    lift_dist = (-forces[:, 0] * sina + forces[:, 2] * cosa) / widths
    chord = 0.5 * (chords[1:] + chords[:-1])
    return lift_dist / (0.5 * rho * v**2 * chord)


def lift_drag(sec_forces, alpha_deg, beta_deg, symmetry=True):
    """Port of ``LiftDrag.compute()``: total dimensional lift/drag."""
    alpha = _as_tensor(alpha_deg) * torch.pi / 180.0
    beta = _as_tensor(beta_deg) * torch.pi / 180.0
    cosa, sina = torch.cos(alpha), torch.sin(alpha)
    cosb, sinb = torch.cos(beta), torch.sin(beta)
    forces = sec_forces.reshape(-1, 3)

    L = torch.sum(-forces[:, 0] * sina + forces[:, 2] * cosa)
    D = torch.sum(forces[:, 0] * cosa * cosb - forces[:, 1] * sinb + forces[:, 2] * sina * cosb)

    if symmetry:
        L = L * 2.0
        D = D * 2.0
    return L, D


def aero_coeffs(S_ref, L, D, v, rho):
    """Port of ``Coeffs.compute()``."""
    q = 0.5 * rho * v**2 * S_ref
    return L / q, D / q  # CL1, CDi


def total_lift(CL1, CL0=0.0):
    """Port of ``TotalLift.compute()``."""
    return CL1 + CL0


def viscous_drag(re, mach, S_ref, widths, lengths_spanwise, lengths, t_over_c, k_lam, c_max_t, with_viscous=True, symmetry=True):
    """Port of ``ViscousDrag.compute()`` (``with_viscous`` branch)."""
    if not with_viscous:
        return 0.0

    cos_sweep = widths / lengths_spanwise
    chords = (lengths[1:] + lengths[:-1]) / 2.0
    Re_c = re * chords

    cdturb_total = 0.455 / (torch.log10(Re_c)) ** 2.58 / (1.0 + 0.144 * mach**2) ** 0.65

    if k_lam == 0:
        cdlam_tr = 0.0
        cdturb_tr = 0.0
    elif k_lam < 1.0:
        cdlam_tr = 1.328 / torch.sqrt(Re_c * k_lam)
        cdturb_tr = 0.455 / (torch.log10(Re_c * k_lam)) ** 2.58 / (1.0 + 0.144 * mach**2) ** 0.65
    else:
        cdlam_tr = 1.328 / torch.sqrt(Re_c * k_lam)
        cdturb_total = 0.0
        cdturb_tr = 0.0

    cd = (cdlam_tr - cdturb_tr) * k_lam + cdturb_total
    d_over_q = 2 * cd * chords

    k_FF = 1.34 * mach**0.18 * (1.0 + 0.6 * t_over_c / c_max_t + 100 * t_over_c**4)
    FF = k_FF * cos_sweep**0.28

    D_over_q = torch.sum(d_over_q * widths * FF)
    CDv = D_over_q / S_ref
    if symmetry:
        CDv = CDv * 2
    return CDv


def total_drag(CDi, CDv, CDw, CD0):
    """Port of ``TotalDrag.compute()``."""
    return CDi + CDv + CDw + CD0


# ---------------------------------------------------------------------------
# Aero -> structure load transfer (not an OpenAeroStruct port -- generic
# glue for coupling this package's panel forces to an external structural
# model that has one node per spanwise VLM station, e.g. a beam/spar mesh).
# ---------------------------------------------------------------------------


def sectional_forces(sec_forces):
    """Sum panel forces over the chordwise axis: an ``(nx-1, ny-1, 3)`` panel
    lattice -> one net force per spanwise strip, ``(ny-1, 3)``. Use this
    before coupling to a structural model with a single node per spanwise
    station (one beam/shell element per strip) regardless of how many
    chordwise VLM panels were used to resolve that strip's loading."""
    return torch.sum(sec_forces, dim=0)


def lump_to_nodes(strip_forces):
    """``(ny-1, 3)`` sectional (per spanwise-strip) forces -> ``(ny, 3)``
    nodal forces, via the standard half-to-each-endpoint lumping (exactly
    conserves total force: ``sum(nodal) == sum(strip_forces)``). Strip ``i``
    sits between nodes ``i`` and ``i + 1``, matching a structural mesh whose
    element ``i`` connects those same two nodes (e.g. a beam model built
    with ``[[i, i + 1] for i in range(n)]`` connectivity)."""
    n_nodes = strip_forces.shape[0] + 1
    nodal = torch.zeros((n_nodes, 3), dtype=strip_forces.dtype)
    nodal[:-1] += 0.5 * strip_forces
    nodal[1:] += 0.5 * strip_forces
    return nodal
