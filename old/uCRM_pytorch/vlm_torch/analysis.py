"""High-level aero-only VLM analysis.

Bundles ``vlm_torch.aerodynamics``/``vlm_torch.geometry``/
``vlm_torch.functionals`` into a single per-case evaluation (mesh -> vortex
mesh -> AIC -> circulations -> panel forces -> CL/CD -> fuel burn/trim), so a
run/optimization script doesn't need to hand-assemble the VLM pipeline
itself.

Two entry points, at two levels:

* ``solve_aero`` -- the generic, mission-independent aerodynamic solve:
  an already-built mesh plus a flight condition (``v``, ``rho``, ``alpha``,
  ``beta``, ...) in, forces/CL/CD out. No ``VLMProblem``, no fuel-burn/trim
  mission parameters required. This is what any VLM consumer needs,
  including a script that couples the VLM to its own (non-``vlm_torch``)
  structural model -- e.g. one lumping panel forces onto a beam mesh -- and
  has no use for Breguet-range fuel burn or L=W trim.
* ``build_problem``/``evaluate`` -- the whole-aircraft-performance API for
  the fixed-flight-condition, mission-level use case (fuel burn, L=W trim
  across a design-variable sweep). ``evaluate`` calls ``solve_aero``
  internally and layers the mission functionals on top.

There is no beam/structural model anywhere in this package: the wing is
always rigid (whatever mesh you pass in), so a solve here is a single VLM
evaluation, not an iterated aerostructural coupling loop.
"""
from dataclasses import dataclass

import numpy as np
import torch

from vlm_torch import geometry
from vlm_torch import aerodynamics as aero
from vlm_torch import functionals as func


def solve_aero(mesh, v, rho, alpha=0.0, beta=0.0, symmetry=True, left_wing=True,
                S_ref_type="wetted", CL0=0.0, CD0=0.0, with_viscous=False,
                k_lam=0.05, c_max_t=0.303, t_over_c=None, re=0.0, mach=0.0):
    """Run the VLM once on an already-built (e.g. twisted) mesh and return
    aerodynamic forces and coefficients. No ``VLMProblem`` or mission
    (fuel-burn, trim) parameters required -- this is the generic, reusable
    aerodynamic solve that both a standalone VLM run and an aerostructural
    coupling loop need; ``build_problem``/``evaluate`` below are the
    higher-level, whole-aircraft-performance API layered on top of it.

    Parameters
    ----------
    mesh : (nx, ny, 3) array
        Already-deformed aerodynamic mesh (apply
        ``geometry.apply_twist_rotation``/``geometry.compute_mesh`` first if
        twist is a design variable). A plain numpy mesh (e.g. straight from
        ``vlm_torch.meshing.generate_mesh``) is accepted and promoted to a
        tensor here; unlike ``jax.numpy``, torch ops do not accept numpy
        arrays directly. A tensor is passed through untouched, so a
        ``requires_grad=True`` mesh stays differentiable.
    v, rho : float
        Freestream speed (m/s) and density (kg/m^3).
    alpha, beta : float
        Angle of attack / sideslip, degrees.
    symmetry : bool
        Whether ``mesh`` is a half-span mesh that should be mirrored about
        ``y=0`` (the usual ``vlm_torch.meshing.generate_mesh`` case), or an
        already-full-span mesh that needs no mirroring (``False`` -- e.g. a
        mesh built to line up 1:1, node-for-node, with a full-span
        structural mesh that isn't itself mirrored).
    left_wing : bool
        Only meaningful when ``symmetry=True``. Precompute this once, from
        the *undeformed* base mesh, with ``aerodynamics.left_wing_from_mesh``
        (a plain-numpy call on a mesh that is a genuine constant) and pass
        the same value on every call -- don't recompute it from ``mesh``
        here, since ``mesh`` may be a tensor carrying an autograd graph
        inside an optimization loop, and topology doesn't change with twist
        anyway.
    S_ref_type : {"wetted", "projected"}
        Passed through to ``aerodynamics.vlm_geometry``.
    CL0, CD0 : float
        Fixed increments added to the VLM's own (induced) CL/CD, e.g. for a
        fuselage/tail contribution not modeled by this surface.
    with_viscous, k_lam, c_max_t, t_over_c, re, mach :
        Passed through to ``aerodynamics.viscous_drag``. ``with_viscous=False``
        (the default here) returns ``CDv=0`` without touching ``re``/``mach``/
        ``t_over_c`` -- ``t_over_c`` defaults to a uniform ``0.15`` if left
        unset when viscous drag *is* requested.

    Returns
    -------
    dict with ``mesh``, ``normals``, ``sec_forces``, ``chords``, ``widths``,
    ``lengths_spanwise``, ``lengths``, ``S_ref``, ``CL``, ``CD``, ``L``, ``D``.
    """
    mesh = aero._as_tensor(mesh)
    nx, ny = mesh.shape[0], mesh.shape[1]
    right_wing = not left_wing

    b_pts, normals, chords, widths, lengths_spanwise, lengths, S_ref = aero.vlm_geometry(
        mesh, S_ref_type=S_ref_type, symmetry=symmetry)

    vmesh = aero.compute_vortex_mesh(mesh, left_wing, symmetry=symmetry)
    coll_pts, force_pts, bound_vecs = aero.collocation_points(mesh)

    vectors_coll = aero.get_vectors(vmesh, coll_pts)
    vel_mtx_coll = aero.eval_vel_mtx(vectors_coll, alpha, ny, symmetry=symmetry, right_wing=right_wing)

    freestream = aero.convert_velocity(v, alpha, beta, coll_pts.shape[0])
    mtx, rhs = aero.build_mtx_rhs(vel_mtx_coll, normals, freestream)
    circulations = aero.solve_circulations(mtx, rhs)
    horseshoe = aero.compute_horseshoe_circulations(circulations, nx, ny)

    vectors_force = aero.get_vectors(vmesh, force_pts)
    vel_mtx_force = aero.eval_vel_mtx(vectors_force, alpha, ny, symmetry=symmetry, right_wing=right_wing)
    force_pts_velocities = aero.eval_velocities(freestream, circulations, vel_mtx_force)

    panel_forces = aero.compute_panel_forces(rho, horseshoe, force_pts_velocities, bound_vecs)
    sec_forces = panel_forces.reshape(nx - 1, ny - 1, 3)

    L, D = aero.lift_drag(sec_forces, alpha, beta, symmetry=symmetry)
    CL1, CDi = aero.aero_coeffs(S_ref, L, D, v, rho)
    CL = aero.total_lift(CL1, CL0=CL0)

    if t_over_c is None:
        t_over_c = 0.15 * torch.ones(ny - 1)
    CDv = aero.viscous_drag(
        re, mach, S_ref, widths, lengths_spanwise, lengths, t_over_c, k_lam, c_max_t,
        with_viscous=with_viscous, symmetry=symmetry,
    )
    CD = aero.total_drag(CDi, CDv, 0.0, CD0)

    return dict(
        mesh=mesh, normals=normals, sec_forces=sec_forces, chords=chords, widths=widths,
        lengths_spanwise=lengths_spanwise, lengths=lengths, S_ref=S_ref, CL=CL, CD=CD, L=L, D=D,
    )


@dataclass
class VLMProblem:
    """Everything about the wing/mission that is NOT a design variable."""

    mesh0: np.ndarray             # (nx, ny, 3) baseline mesh, half-wing
    b_twist: torch.Tensor          # twist_cp (deg) -> per-node twist (deg)
    ny: int
    left_wing: bool
    right_wing: bool
    symmetry: bool = True
    CL0: float = 0.0
    CD0: float = 0.0
    k_lam: float = 0.05
    c_max_t: float = 0.303
    with_viscous: bool = True
    t_over_c: torch.Tensor = None  # (ny-1,), fixed panel thickness-to-chord

    # flight condition / mission (Breguet range + L=W trim)
    v: float = 0.0
    beta: float = 0.0
    Mach_number: float = 0.0
    re: float = 0.0
    rho: float = 0.0
    speed_of_sound: float = 0.0
    CT: float = 0.0
    R: float = 0.0
    W0: float = 0.0
    load_factor: float = 1.0
    # No beam model here, so there is no computed wing_structural_mass --
    # fold the entire non-fuel aircraft weight into W0 and leave this 0.
    wing_structural_mass: float = 0.0


def build_problem(mesh0, num_twist_cp, **kwargs) -> VLMProblem:
    """Construct a ``VLMProblem`` from a baseline mesh.

    ``mesh0``/``num_twist_cp`` derive ``b_twist``/``ny``/``left_wing``
    (topological properties of the fixed mesh); every other ``VLMProblem``
    field is supplied via ``kwargs`` (e.g. ``v=248.136, CD0=0.015, ...``).
    """
    mesh0 = np.asarray(mesh0)
    ny = mesh0.shape[1]
    b_twist = geometry.build_twist_bspline_matrix(mesh0, num_twist_cp)
    left_wing = aero.left_wing_from_mesh(mesh0)
    right_wing = kwargs.pop("right_wing", not left_wing)
    t_over_c = kwargs.pop("t_over_c", 0.15 * torch.ones(ny - 1))
    return VLMProblem(
        mesh0=mesh0, b_twist=b_twist, ny=ny, left_wing=left_wing, right_wing=right_wing,
        t_over_c=t_over_c, **kwargs,
    )


def evaluate(p: VLMProblem, twist_cp, alpha):
    """``twist_cp, alpha -> CL, CD, fuelburn, L_equals_W``, plus the raw VLM
    outputs (``mesh``, ``sec_forces``, ``normals``) needed to visualize or
    post-process (e.g. with ``panel_pressure``) the solved case."""
    mesh = geometry.compute_mesh(p.mesh0, twist_cp, p.b_twist, symmetry=p.symmetry)
    aux = solve_aero(
        mesh, p.v, p.rho, alpha=alpha, beta=p.beta, symmetry=p.symmetry, left_wing=p.left_wing,
        CL0=p.CL0, CD0=p.CD0, with_viscous=p.with_viscous, k_lam=p.k_lam, c_max_t=p.c_max_t,
        t_over_c=p.t_over_c, re=p.re, mach=p.Mach_number,
    )

    S_ref_total = func.sum_areas(aux["S_ref"])
    CL, CD, Ltot, Dtot = func.total_lift_drag(aux["CL"], aux["CD"], aux["S_ref"], S_ref_total, p.rho, p.v)
    fuelburn = func.breguet_range_fuelburn(
        p.wing_structural_mass, p.CT, p.speed_of_sound, p.R, CL, CD, p.Mach_number, p.W0
    )
    L_equals_W, total_weight = func.equilibrium(
        p.wing_structural_mass, fuelburn, p.W0, p.load_factor, CL, S_ref_total, p.v, p.rho
    )

    return dict(
        CL=CL, CD=CD, fuelburn=fuelburn, L_equals_W=L_equals_W,
        mesh=aux["mesh"], normals=aux["normals"], sec_forces=aux["sec_forces"],
    )


def panel_pressure(mesh, sec_forces, normals, orient=None):
    """Per-panel pressure jump (Pa): the panel-normal component of the
    Kutta-Joukowski panel force (``sec_forces``, from
    ``aerodynamics.compute_panel_forces``) divided by panel area (the same
    diagonal-cross-product formula ``vlm_geometry`` uses for ``S_ref`` --
    exact for a planar quad).

    Preferred over raw circulation strength for visualization: circulation
    has an inherent chordwise sawtooth shape from the VLM horseshoe/ring
    formulation, which reads as a modeling artifact rather than physics.
    Pressure is the standard quantity these tools show (what AVL/XFLR5-style
    post-processing displays) and makes a twist distribution's effect on the
    spanwise load visually legible.

    Parameters
    ----------
    orient : {None, "up"}
        ``normals`` come from a mesh-index cross product (``vlm_geometry``),
        so their sign follows the mesh's *index* ordering, not a fixed
        physical direction -- a mesh whose spanwise index runs in the
        opposite direction from ``vlm_torch.meshing.generate_mesh``'s usual
        increasing-y-with-index convention (e.g. one built to match a
        structural mesh's own node order) can end up with panel normals
        pointing down instead of up, silently flipping the sign of every
        returned pressure. Default (``None``) uses ``normals`` exactly as
        given, matching prior behavior. Pass ``orient="up"`` to instead
        auto-flip the sign so the result is positive where the wing is
        lift-loaded (mean normal z-component >= 0), regardless of the
        mesh's index convention -- most useful before visualizing/coloring
        panels, where the sign is meant to read as "high/low pressure", not
        as a literal record of which way the mesh happened to be indexed.
    """
    mesh = aero._as_tensor(mesh)
    diag1 = mesh[1:, 1:, :] - mesh[:-1, :-1, :]
    diag2 = mesh[:-1, 1:, :] - mesh[1:, :-1, :]
    area = 0.5 * torch.linalg.norm(torch.linalg.cross(diag1, diag2), dim=-1)

    if orient == "up":
        if float(torch.mean(normals[..., 2])) < 0:
            normals = -normals
    elif orient is not None:
        raise ValueError(f"Unknown orient {orient!r}, expected None or 'up'.")

    normal_force = torch.sum(sec_forces * normals, dim=-1)
    return normal_force / area
