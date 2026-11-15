"""Total aircraft performance: per-surface CL/CD -> fuel burn, L=W, cg, CM.

Ports (single-surface specialization of):
  openaerostruct/functionals/sum_areas.py         (SumAreas)
  openaerostruct/functionals/total_lift_drag.py   (TotalLiftDrag)
  openaerostruct/functionals/breguet_range.py     (BreguetRange)
  openaerostruct/functionals/equilibrium.py       (Equilibrium)
  openaerostruct/functionals/center_of_gravity.py (CenterOfGravity)
  openaerostruct/functionals/moment_coefficient.py(MomentCoefficient)

For a single surface, ``SumAreas``/``TotalLiftDrag`` reduce to identities on
CL/CD (dividing the single surface's contribution by itself); they are kept
as explicit steps here (rather than skipped) since ``S_ref_total``, ``L``,
and ``D`` are genuine, separately-used outputs.

``BreguetRange`` looks self-referential (fuel burn depends on total weight,
which includes fuel burn) but is not: the classic closed-form Breguet
solution only needs the pre-fuel weight (W0 + structural mass), so this is a
single explicit evaluation, not an inner solve.
"""
import torch

GRAV_CONSTANT = 9.80665  # openaerostruct/utils/constants.py


def sum_areas(wing_S_ref):
    """Port of ``SumAreas.compute()`` (single surface)."""
    return wing_S_ref


def total_lift_drag(wing_CL, wing_CD, wing_S_ref, S_ref_total, rho, v):
    """Port of ``TotalLiftDrag.compute()`` (single surface). Preserves the
    source's exact evaluation order (L, D computed from the *unnormalized*
    area-weighted sum before dividing by S_ref_total) for bit-for-bit-style
    fidelity, even though with one surface CL_total == wing_CL exactly."""
    CL_unnorm = wing_CL * wing_S_ref
    CD_unnorm = wing_CD * wing_S_ref
    L = CL_unnorm * 0.5 * rho * v**2
    D = CD_unnorm * 0.5 * rho * v**2
    CL = CL_unnorm / S_ref_total
    CD = CD_unnorm / S_ref_total
    return CL, CD, L, D


def breguet_range_fuelburn(wing_structural_mass, CT, speed_of_sound, R, CL, CD, Mach_number, W0):
    """Port of ``BreguetRange.compute()``. Closed-form; no inner solve."""
    Ws = wing_structural_mass
    return (W0 + Ws) * (torch.exp(R * CT / speed_of_sound / Mach_number * CD / CL) - 1)


def equilibrium(wing_structural_mass, fuelburn, W0, load_factor, CL, S_ref_total, v, rho):
    """Port of ``Equilibrium.compute()``. Returns (L_equals_W, total_weight[N])."""
    g = GRAV_CONSTANT * load_factor
    total_weight = (wing_structural_mass + fuelburn + W0) * g
    L_equals_W = 1 - (0.5 * rho * v**2 * S_ref_total) * CL / total_weight
    return L_equals_W, total_weight


def center_of_gravity(wing_structural_mass, wing_cg_location, total_weight, fuelburn, W0, load_factor, empty_cg):
    """Port of ``CenterOfGravity.compute()`` (single surface)."""
    g = GRAV_CONSTANT * load_factor
    W0_cg = W0 * empty_cg
    spar_cg = wing_cg_location * wing_structural_mass
    return (W0_cg + spar_cg) / (total_weight / g - fuelburn)


def moment_coefficient(wing_b_pts, wing_widths, wing_chords, wing_S_ref, wing_sec_forces, cg, v, rho, S_ref_total, symmetry=True):
    """Port of ``MomentCoefficient.compute()`` (single surface, ``j==0`` path
    always taken -- ``MAC``/``S_ref_wing`` are always defined from this one
    surface)."""
    panel_chords = (wing_chords[1:] + wing_chords[:-1]) * 0.5
    MAC = 1.0 / wing_S_ref * torch.sum(panel_chords**2 * wing_widths)
    if symmetry:
        MAC = MAC * 2.0

    pts = (wing_b_pts[:, 1:, :] + wing_b_pts[:, :-1, :]) * 0.5
    diff = pts - cg[None, None, :]
    moment = torch.sum(torch.linalg.cross(diff, wing_sec_forces, dim=2), dim=0)  # (ny-1, 3)

    if symmetry:
        moment = moment.clone()
        moment[:, 0] = 0.0
        moment[:, 1] *= 2.0
        moment[:, 2] = 0.0

    M = torch.sum(moment, dim=0)
    CM = M / (0.5 * rho * v**2 * S_ref_total * MAC)
    return CM, M
