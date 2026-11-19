"""Numerical core of QuasiNewtonALBCD: pure functions, numpy only, no solver state.

Conventions (IMPLEMENTATION_PLAN.md §1). A block's linearization ``lin`` is the dict returned by
QNSubproblem.grad_phi, plus ``lin["v"]``, the block's (unscaled) variables at which it was taken. All linear
algebra works in the scaled variables x_hat = D * x: gradients and Jacobian columns are divided by D, and a
scaled step is divided by D to give the unscaled one.

    Lagrangian       L = sum_i f_i + lam^T phi + sum_i kappa_i^T (h_iA - b_iA)
    stationarity     r_i = grad f_i + J_i^T lam + Jh_i^T kappa_i      (kappa_i is 0 on inactive rows)
"""

import numpy as np


# ----------------------------------------------------------------------------------------------------------
# damped BFGS (§1.7)
# ----------------------------------------------------------------------------------------------------------

def initial_scale(s, v):
    """sigma = clip(v^T v / s^T v, 1e-6, 1e6), the scale of B = sigma I taken on the first pair with s^T v > 0."""
    return float(np.clip(v @ v / (s @ v), 1e-6, 1e6))


def damped_bfgs_update(B, s, v, xhat_norm=0.0):
    """Powell-damped BFGS update of the SPD matrix B with the pair (s, v); returns the new matrix.

    The pair is skipped (B returned unchanged) if ||s|| <= 1e-12 max(1, ||x_hat||), with ||x_hat|| = xhat_norm.
    """
    s = np.asarray(s, dtype=float)
    v = np.asarray(v, dtype=float)
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(v))):
        return B
    if np.linalg.norm(s) <= 1e-12 * max(1.0, xhat_norm):
        return B
    Bs = B @ s
    sBs = s @ Bs
    if not sBs > 0.0:
        return B
    sv = s @ v
    if sv < 0.2 * sBs:
        omega = 0.8 * sBs / (sBs - sv)
        v = omega * v + (1.0 - omega) * Bs
        sv = s @ v
    B = B - np.outer(Bs, Bs) / sBs + np.outer(v, v) / sv
    return 0.5 * (B + B.T)


def apply_pair(B, s, v, initialized, xhat_norm=0.0):
    """One BFGS pair with the initialization rule of §1.7: B stays as it is (I by default) until the first pair
    with s^T v > 0, which first resets B to sigma I. Returns (B, initialized, used)."""
    s = np.asarray(s, dtype=float)
    v = np.asarray(v, dtype=float)
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(v))) or np.linalg.norm(s) <= 1e-12 * max(1.0, xhat_norm):
        return B, initialized, False
    if not initialized:
        if not s @ v > 0.0:
            return B, initialized, False
        B = initial_scale(s, v) * np.eye(len(s))
        initialized = True
    return damped_bfgs_update(B, s, v, xhat_norm), initialized, True


def penalty_update(B, J, D, dmu):
    """B + J_hat^T diag(dmu) J_hat, with J_hat = J / D (columnwise): the change of the penalty curvature."""
    Jh = np.asarray(J, dtype=float) / D
    return B + Jh.T @ (np.asarray(dmu, dtype=float)[:, None] * Jh)


# ----------------------------------------------------------------------------------------------------------
# gradients, active sets and the KKT measure (§1.2-1.6)
# ----------------------------------------------------------------------------------------------------------

def stationarity(lin, lam, kappa):
    """r = grad f + J^T lam + Jh^T kappa (unscaled), over all of the block's variables."""
    r = np.asarray(lin["grad_f"], dtype=float) + np.asarray(lin["J"]).T @ lam
    if len(kappa):
        r = r + np.asarray(lin["Jh"]).T @ kappa
    return r


def kappa_or_zeros(kappa, nh):
    """kappa if it has one entry per local constraint (nh of them), else zeros (no multipliers yet)."""
    return kappa if kappa is not None and len(kappa) == nh else np.zeros(nh)


def al_gradient(lin, lam, mu, phi_tilde, kappa):
    """Gradient of the block's augmented Lagrangian f_i + lam^T phi + 1/2 sum mu phi^2 + kappa^T h (unscaled),
    with phi_tilde = c_i(v) + sum_{j != i} c_j the coupling at this block's point and the others' current ones."""
    return stationarity(lin, np.asarray(lam) + np.asarray(mu) * np.asarray(phi_tilde), kappa)


def active_sets(lin, kappa, r, act_tol=1e-8, bnd_tol=1e-8, tight_tol=None):
    """Free variables and active local constraints of one block (§1.3-1.4).

    Rows: equalities (cl == cu) are always active. An inequality row is active if
    |kappa_j| > act_tol max(1, ||kappa||_inf), or, when tight_tol is given, if it is within tight_tol of one of
    its bounds. The bound it is active at is cu if kappa_j > 0, cl if kappa_j < 0 (the finite or the nearer one
    otherwise).

    Variables: on a bound if within bnd_tol max(1, |bound|) of it; fixed if on the lower bound with r_j >= 0 or
    on the upper bound with r_j <= 0. All others are free.

    Returns (free_mask, active_rows, bound_at), with bound_at the b value of each active row.
    """
    h = np.asarray(lin["h"], dtype=float)
    cl = np.asarray(lin["cl"], dtype=float)
    cu = np.asarray(lin["cu"], dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    rows, bvals = [], []
    kmax = max(1.0, np.max(np.abs(kappa), initial=0.0))
    for j in range(len(h)):
        if cl[j] == cu[j]:
            rows.append(j)
            bvals.append(cl[j])
            continue
        if abs(kappa[j]) > act_tol * kmax:
            if kappa[j] > 0 and np.isfinite(cu[j]):
                b = cu[j]
            elif kappa[j] < 0 and np.isfinite(cl[j]):
                b = cl[j]
            else:
                b = cu[j] if np.isfinite(cu[j]) else cl[j]
            rows.append(j)
            bvals.append(b)
        elif tight_tol is not None:
            dl = abs(h[j] - cl[j]) if np.isfinite(cl[j]) else np.inf
            du = abs(h[j] - cu[j]) if np.isfinite(cu[j]) else np.inf
            if min(dl, du) <= tight_tol:
                rows.append(j)
                bvals.append(cl[j] if dl <= du else cu[j])

    v = np.asarray(lin["v"], dtype=float)
    xl = np.asarray(lin["xl"], dtype=float)
    xu = np.asarray(lin["xu"], dtype=float)
    with np.errstate(invalid="ignore"):
        on_lower = np.isfinite(xl) & (v <= xl + bnd_tol * np.maximum(1.0, np.abs(xl)))
        on_upper = np.isfinite(xu) & (v >= xu - bnd_tol * np.maximum(1.0, np.abs(xu)))
    fixed = (on_lower & (r >= 0)) | (on_upper & (r <= 0)) | (on_lower & on_upper)
    return ~fixed, np.array(rows, dtype=int), np.array(bvals, dtype=float)


def local_violation(lin):
    """Largest violation of cl <= h <= cu and of xl <= v <= xu (0 if satisfied)."""
    h = np.asarray(lin["h"], dtype=float)
    v = np.asarray(lin["v"], dtype=float)
    viol = 0.0
    if len(h):
        viol = max(viol, np.max(np.asarray(lin["cl"]) - h, initial=0.0), np.max(h - np.asarray(lin["cu"]), initial=0.0))
    viol = max(viol, np.max(np.asarray(lin["xl"]) - v, initial=0.0), np.max(v - np.asarray(lin["xu"]), initial=0.0))
    return float(viol)


def kkt_measure(lins, lam, kappas, free_masks, phi, parts=False):
    """theta = max(max_i ||(r_i)_F||_inf, ||phi||_inf, max_i localviol_i) (§1.6), unscaled.

    With parts=True, returns (theta, stationarity, feasibility, local violation)."""
    stat = max((np.max(np.abs(stationarity(lin, lam, k)[F]), initial=0.0)
                for lin, k, F in zip(lins, kappas, free_masks)), default=0.0)
    feas = float(np.max(np.abs(phi), initial=0.0))
    viol = max((local_violation(lin) for lin in lins), default=0.0)
    theta = max(stat, feas, viol)
    return (theta, stat, feas, viol) if parts else theta


# ----------------------------------------------------------------------------------------------------------
# the coupled step (§1.8)
# ----------------------------------------------------------------------------------------------------------

def block_condense(B, lin, D, lam, kappa, free, rows, bvals, check=1e-8):
    """One block's part of the condensed system (§1.8).

    Returns a dict with W = K^-1 [J_hat^T; 0], u = K^-1 [r_hat; g], S = J_hat W[:nF], s = J_hat u[:nF] and the
    pieces back_substitute needs. Raises numpy.linalg.LinAlgError if K is singular or the solve's residual
    exceeds ``check`` relative.
    """
    D = np.asarray(D, dtype=float)
    free = np.asarray(free, dtype=bool)
    rows = np.asarray(rows, dtype=int)
    Df = D[free]
    nF, nA = int(free.sum()), len(rows)

    BF = np.asarray(B)[np.ix_(free, free)]
    Jhat = (np.asarray(lin["J"], dtype=float) / D)[:, free]
    Ghat = (np.asarray(lin["Jh"], dtype=float).reshape(-1, len(D))[rows] / D)[:, free] if nA else np.zeros((0, nF))
    k = np.zeros(len(lin["h"]))
    if nA:
        k[rows] = np.asarray(kappa, dtype=float)[rows]
    rhat = stationarity(lin, lam, k)[free] / Df
    g = np.asarray(lin["h"], dtype=float)[rows] - bvals if nA else np.zeros(0)

    K = np.block([[BF, Ghat.T], [Ghat, np.zeros((nA, nA))]])
    rhs = np.hstack([np.vstack([Jhat.T, np.zeros((nA, Jhat.shape[0]))]), np.concatenate([rhat, g])[:, None]])
    sol = np.linalg.solve(K, rhs)  # raises LinAlgError if K is exactly singular
    res = np.linalg.norm(K @ sol - rhs) / max(np.linalg.norm(K) * np.linalg.norm(sol), np.linalg.norm(rhs), 1e-300)
    if not np.all(np.isfinite(sol)) or res > check:
        raise np.linalg.LinAlgError(f"singular K (relative residual {res:.2e})")
    W, u = sol[:, :-1], sol[:, -1]
    return dict(K=K, W=W, u=u, S=Jhat @ W[:nF], s=Jhat @ u[:nF], Jhat=Jhat, free=free, rows=rows, D=D,
                nF=nF, nA=nA, n=len(D), nh=len(lin["h"]))


def coordinator_solve(S_list, s_list, phi, eps=1e-14):
    """dlam = (S + eps tr(S)/m I)^-1 (phi - sum s_i), S = sum S_i symmetrized."""
    S = sum(S_list)
    S = 0.5 * (S + S.T)
    m = S.shape[0]
    rhs = np.asarray(phi, dtype=float) - sum(s_list)
    return np.linalg.solve(S + eps * np.trace(S) / m * np.eye(m), rhs)


def back_substitute(bd, dlam, u=None):
    """[dx_hat_F; dkappa_A] = -(u + W dlam). Returns (dx, dkappa): dx unscaled over all the block's variables (0 on
    fixed ones), dkappa over all its local constraints (0 on inactive ones). ``u`` overrides bd["u"]."""
    z = -((bd["u"] if u is None else u) + bd["W"] @ dlam)
    dx = np.zeros(bd["n"])
    dx[bd["free"]] = z[:bd["nF"]] / bd["D"][bd["free"]]
    dkappa = np.zeros(bd["nh"])
    dkappa[bd["rows"]] = z[bd["nF"]:]
    return dx, dkappa


def coupled_step(block_data, phi, mu, refine=1, eps=1e-14):
    """The coupled quasi-Newton step from the condensed blocks, with ``refine`` correction rounds for the
    cross-block penalty curvature that the block matrices B_i (which hold J_i^T diag(mu) J_i) double-count.

    Each round solves the condensed system again with the same factorizations, with r_hat = -J_hat^T diag(mu)
    (J_hat dx_hat_F) built from the latest increment, g = 0 and phi = 0. Since that right-hand side lies in the
    range of J_hat^T, its K^-1 solve is -W (mu * J_hat dx_hat_F), so no new factorization is needed.

    Returns (dx_list, dkappa_list, dlam).
    """
    mu = np.asarray(mu, dtype=float)
    S_list = [bd["S"] for bd in block_data]
    dlam = coordinator_solve(S_list, [bd["s"] for bd in block_data], phi, eps)
    steps = [back_substitute(bd, dlam) for bd in block_data]
    dx = [st[0] for st in steps]
    dk = [st[1] for st in steps]

    incr = dx
    for _ in range(int(refine)):
        us = [-bd["W"] @ (mu * (bd["Jhat"] @ (d[bd["free"]] * bd["D"][bd["free"]]))) for bd, d in zip(block_data, incr)]
        ss = [bd["Jhat"] @ u[:bd["nF"]] for bd, u in zip(block_data, us)]
        dlam_c = coordinator_solve(S_list, ss, np.zeros_like(dlam), eps)
        corr = [back_substitute(bd, dlam_c, u) for bd, u in zip(block_data, us)]
        dx = [a + c[0] for a, c in zip(dx, corr)]
        dk = [a + c[1] for a, c in zip(dk, corr)]
        dlam = dlam + dlam_c
        incr = [c[0] for c in corr]
    return dx, dk, dlam
