"""Tests of qn_core: damped BFGS, the condensed coupled step, its refinement and the active sets (Task 2)."""

import numpy as np
import pytest

import qn_core as qc


# ----------------------------------------------------------------------------------------------------------
# 1. BFGS
# ----------------------------------------------------------------------------------------------------------

def random_spd(rng, n, lo=0.5, hi=5.0):
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return Q @ np.diag(rng.uniform(lo, hi, n)) @ Q.T


@pytest.mark.parametrize("seed", range(5))
def test_bfgs_secant_condition(seed):
    rng = np.random.default_rng(seed)
    n = 7
    B = random_spd(rng, n)
    A = random_spd(rng, n)
    s = rng.standard_normal(n)
    v = A @ s  # s^T v > 0; make sure the undamped branch is taken
    if s @ v < 0.2 * s @ B @ s:
        v = v + (0.2 * s @ B @ s - s @ v + 1.0) * s / (s @ s)
    B1 = qc.damped_bfgs_update(B, s, v)
    np.testing.assert_allclose(B1 @ s, v, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(B1, B1.T)


def test_bfgs_stays_spd_under_negative_curvature():
    rng = np.random.default_rng(1)
    n = 6
    for trial in range(20):
        B = np.eye(n)
        for _ in range(10):
            s = rng.standard_normal(n)
            v = rng.standard_normal(n)
            if s @ v > 0:
                v = -v
            B = qc.damped_bfgs_update(B, s, v)
            assert np.min(np.linalg.eigvalsh(B)) > 0
    np.testing.assert_allclose(B, B.T)


def test_bfgs_skips_tiny_steps_and_initializes():
    B = 3.0 * np.eye(2)
    assert qc.damped_bfgs_update(B, np.array([1e-14, 0.0]), np.array([1.0, 0.0]), xhat_norm=1.0) is B
    # no initialization before a pair with s^T v > 0
    B1, init, used = qc.apply_pair(np.eye(2), np.array([1.0, 0.0]), np.array([-1.0, 0.0]), False)
    assert not init and not used and np.array_equal(B1, np.eye(2))
    s, v = np.array([1.0, 1.0]), np.array([4.0, 4.0])
    B1, init, used = qc.apply_pair(np.eye(2), s, v, False)
    assert init and used
    assert qc.initial_scale(s, v) == pytest.approx(4.0)
    np.testing.assert_allclose(B1 @ s, v)
    assert qc.initial_scale(np.array([1.0]), np.array([1e-9])) == 1e-6  # clipped


def test_penalty_update():
    rng = np.random.default_rng(2)
    J = rng.standard_normal((3, 5))
    D = rng.uniform(0.1, 10, 5)
    dmu = rng.uniform(0, 2, 3)
    B = random_spd(rng, 5)
    Jh = J / D
    np.testing.assert_allclose(qc.penalty_update(B, J, D, dmu), B + Jh.T @ np.diag(dmu) @ Jh)


# ----------------------------------------------------------------------------------------------------------
# random block problems
# ----------------------------------------------------------------------------------------------------------

def random_problem(rng, m_c, sizes=(5, 8, 12), mu_scale=1.0, H_given=False):
    """Three random blocks with 0-3 active rows (equality, lower and upper), one inactive inequality row, 0-2
    fixed variables, random scaling and random SPD B_i. With H_given, B_i = H_i + J_hat^T diag(mu) J_hat for a
    random SPD H_i, which is returned too."""
    mu = mu_scale * rng.uniform(0.5, 1.5, m_c)
    lam = rng.standard_normal(m_c)
    blocks = []
    for n in sizes:
        nA = int(rng.integers(0, 4))
        nh = nA + 1
        D = 10.0 ** rng.uniform(-2, 2, n)
        v = rng.standard_normal(n)
        Jh = rng.standard_normal((nh, n))
        h = rng.standard_normal(nh)
        kappa = np.zeros(nh)
        cl = np.full(nh, -np.inf)
        cu = np.full(nh, np.inf)
        for j in range(nA):
            kind = j % 3
            if kind == 0:   # equality
                cl[j] = cu[j] = h[j] + 0.1 * rng.standard_normal()
                kappa[j] = rng.standard_normal()
            elif kind == 1:  # active at a lower bound
                cl[j] = h[j] - 0.01
                kappa[j] = -rng.uniform(0.5, 2)
            else:            # active at an upper bound
                cu[j] = h[j] + 0.01
                kappa[j] = rng.uniform(0.5, 2)
        cu[nA] = h[nA] + 5.0  # inactive
        # with H_given, J_hat = J / D has O(1) entries, so the penalty curvature is O(mu) next to H (eig. 1-10)
        J = rng.standard_normal((m_c, n)) * (D / np.sqrt(n) if H_given else 1.0)
        lin = dict(grad_f=rng.standard_normal(n), c=rng.standard_normal(m_c), J=J,
                   h=h, Jh=Jh, cl=cl, cu=cu, xl=np.full(n, -np.inf), xu=np.full(n, np.inf), v=v, outputs={})
        r = qc.stationarity(lin, lam, kappa)
        for j in rng.choice(n, int(rng.integers(0, 3)), replace=False):
            if r[j] >= 0:
                lin["xl"][j] = v[j]
            else:
                lin["xu"][j] = v[j]
        Jhat = lin["J"] / D
        if H_given:
            H = random_spd(rng, n, 1.0, 10.0)
            B = H + Jhat.T @ np.diag(mu) @ Jhat
        else:
            H = None
            B = random_spd(rng, n)
        free, rows, bvals = qc.active_sets(lin, kappa, r)
        assert len(rows) == nA and np.sum(~free) <= 2
        blocks.append(dict(lin=lin, D=D, kappa=kappa, B=B, H=H, free=free, rows=rows, bvals=bvals))
    phi = rng.standard_normal(m_c)
    return blocks, lam, mu, phi


def assembled_step(blocks, lam, phi, which="B"):
    """The full bordered system over every block's free variables, active rows and the coupling (test only)."""
    nFs = [int(b["free"].sum()) for b in blocks]
    nAs = [len(b["rows"]) for b in blocks]
    m = len(phi)
    nx, na = sum(nFs), sum(nAs)
    N = nx + na + m
    K = np.zeros((N, N))
    rhs = np.zeros(N)
    ox, oa = 0, nx
    for b, nF, nA in zip(blocks, nFs, nAs):
        F, rows, D, lin = b["free"], b["rows"], b["D"], b["lin"]
        M = b[which]
        Jhat = (lin["J"] / D)[:, F]
        Ghat = (lin["Jh"][rows] / D)[:, F]
        k = np.zeros(len(lin["h"]))
        k[rows] = b["kappa"][rows]
        K[ox:ox + nF, ox:ox + nF] = M[np.ix_(F, F)]
        K[ox:ox + nF, oa:oa + nA] = Ghat.T
        K[oa:oa + nA, ox:ox + nF] = Ghat
        K[ox:ox + nF, nx + na:] = Jhat.T
        K[nx + na:, ox:ox + nF] = Jhat
        rhs[ox:ox + nF] = -qc.stationarity(lin, lam, k)[F] / D[F]
        rhs[oa:oa + nA] = -(lin["h"][rows] - b["bvals"])
        ox += nF
        oa += nA
    rhs[nx + na:] = -phi
    z = np.linalg.solve(K, rhs)
    dxs, dks = [], []
    ox, oa = 0, nx
    for b, nF, nA in zip(blocks, nFs, nAs):
        dx = np.zeros(len(b["D"]))
        dx[b["free"]] = z[ox:ox + nF] / b["D"][b["free"]]
        dk = np.zeros(len(b["lin"]["h"]))
        dk[b["rows"]] = z[oa:oa + nA]
        dxs.append(dx)
        dks.append(dk)
        ox += nF
        oa += nA
    return dxs, dks, z[nx + na:]


def condensed_step(blocks, lam, mu, phi, refine):
    bds = [qc.block_condense(b["B"], b["lin"], b["D"], lam, b["kappa"], b["free"], b["rows"], b["bvals"]) for b in blocks]
    return qc.coupled_step(bds, phi, mu, refine=refine)


def step_error(a, b):
    dxa, dka, dla = a
    dxb, dkb, dlb = b
    return max(max(np.max(np.abs(x - y), initial=0) for x, y in zip(dxa, dxb)),
               max(np.max(np.abs(x - y), initial=0) for x, y in zip(dka, dkb)),
               np.max(np.abs(dla - dlb)))


def step_size(a):
    dx, dk, dl = a
    return max(max(np.max(np.abs(x), initial=0) for x in dx), max(np.max(np.abs(x), initial=0) for x in dk),
               np.max(np.abs(dl)))


# ----------------------------------------------------------------------------------------------------------
# 2. condensed step = assembled step
# ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("m_c", [1, 4])
@pytest.mark.parametrize("seed", range(10))
def test_condensed_equals_assembled(m_c, seed):
    rng = np.random.default_rng(100 * m_c + seed)
    blocks, lam, mu, phi = random_problem(rng, m_c)
    cond = condensed_step(blocks, lam, mu, phi, refine=0)
    full = assembled_step(blocks, lam, phi, "B")
    assert step_error(cond, full) <= 1e-10 * max(1.0, step_size(full))
    # fixed variables do not move
    for b, dx in zip(blocks, cond[0]):
        assert np.all(dx[~b["free"]] == 0)


# ----------------------------------------------------------------------------------------------------------
# 3. refinement
# ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("m_c", [1, 4])
@pytest.mark.parametrize("seed", range(10))
def test_refinement_converges_to_true_system(m_c, seed):
    """B_i = H_i + J_hat_i^T diag(mu) J_hat_i with SPD H_i (eigenvalues 1-10) and mu = O(0.1). The refinement
    is a splitting iteration whose limit is the true system (H_i blocks, penalty-free coupling)."""
    rng = np.random.default_rng(1000 + 100 * m_c + seed)
    blocks, lam, mu, phi = random_problem(rng, m_c, mu_scale=0.1, H_given=True)
    true = assembled_step(blocks, lam, phi, "H")
    e0 = step_error(condensed_step(blocks, lam, mu, phi, refine=0), true)
    e1 = step_error(condensed_step(blocks, lam, mu, phi, refine=1), true)
    e_many = step_error(condensed_step(blocks, lam, mu, phi, refine=60), true)
    assert e_many <= 1e-10 * max(1.0, step_size(true))
    assert e1 <= 0.1 * e0


# ----------------------------------------------------------------------------------------------------------
# 4. active sets
# ----------------------------------------------------------------------------------------------------------

def lin_for(h, cl, cu, v, xl, xu):
    n = len(v)
    return dict(grad_f=np.zeros(n), c=np.zeros(1), J=np.zeros((1, n)), h=np.asarray(h, float),
                Jh=np.zeros((len(h), n)), cl=np.asarray(cl, float), cu=np.asarray(cu, float),
                xl=np.asarray(xl, float), xu=np.asarray(xu, float), v=np.asarray(v, float), outputs={})


def test_active_rows():
    inf = np.inf
    #        eq    lower  upper  two-sided(lower) two-sided(upper) inactive  tiny kappa
    h = [1.0, 0.0, 2.0, 0.0, 1.0, 0.5, 0.0]
    cl = [1.0, 0.0, -inf, 0.0, 0.0, 0.0, 0.0]
    cu = [1.0, inf, 2.0, 1.0, 1.0, 1.0, inf]
    kappa = np.array([0.0, -2.0, 3.0, -1.0, 0.5, 0.0, 1e-12])
    lin = lin_for(h, cl, cu, [0.0], [-inf], [inf])
    free, rows, b = qc.active_sets(lin, kappa, np.zeros(1))
    np.testing.assert_array_equal(rows, [0, 1, 2, 3, 4])  # the equality is active even with kappa = 0
    np.testing.assert_array_equal(b, [1.0, 0.0, 2.0, 0.0, 1.0])
    # a tight row with zero multiplier is active only with tight_tol
    _, rows, b = qc.active_sets(lin, kappa, np.zeros(1), tight_tol=1e-8)
    np.testing.assert_array_equal(rows, [0, 1, 2, 3, 4, 6])
    assert b[-1] == 0.0


def test_fixed_variables():
    inf = np.inf
    v = [0.0, 0.0, 1.0, 1.0, 0.5, 2.0, 1.0 + 1e-10]
    xl = [0.0, 0.0, -inf, -inf, 0.0, 2.0, 1.0]
    xu = [inf, inf, 1.0, 1.0, 1.0, 2.0, inf]
    r = np.array([1.0, -1.0, -1.0, 1.0, 5.0, 3.0, 0.0])
    lin = lin_for([], [], [], v, xl, xu)
    free, rows, _ = qc.active_sets(lin, np.zeros(0), r)
    # lower with r >= 0: fixed; lower with r < 0: free; upper with r <= 0: fixed; upper with r > 0: free;
    # interior: free; xl == xu: fixed; within bnd_tol of the lower bound with r = 0: fixed
    np.testing.assert_array_equal(free, [False, True, False, True, True, False, False])
    assert len(rows) == 0


def test_kkt_measure():
    lin = lin_for([0.5, 2.0], [0.0, -np.inf], [1.0, 1.5], [0.0, 3.0], [0.0, -np.inf], [np.inf, np.inf])
    lin["grad_f"] = np.array([2.0, -0.25])
    lin["J"] = np.array([[1.0, 0.0]])
    free = np.array([False, True])
    theta, stat, feas, viol = qc.kkt_measure([lin], np.array([1.0]), [np.zeros(2)], [free], np.array([0.1]), parts=True)
    assert stat == 0.25 and feas == 0.1 and viol == pytest.approx(0.5) and theta == pytest.approx(0.5)
