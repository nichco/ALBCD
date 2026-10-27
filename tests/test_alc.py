"""Tests of the ALC coordinator on the closed-form two-block problem of test_albcd.py:

    min (x0 - a)^2 + (x1 - b)^2  s.t.  phi = x0 - x1 = 0,  solution x0 = x1 = (a + b) / 2, y = a - b.
"""

import numpy as np

from albcd import ALBCD, ALC, initial_penalty
from test_albcd import Block, A, B


def objective(x, data):
    return (x[0] - A) ** 2 + (x[1] - B) ** 2


def blocks():
    return [Block(0, A), Block(1, B)]


def test_converges_to_known_solution():
    opt = ALC(blocks(), np.zeros(2), np.ones(1), objective, phi0=np.zeros(1), feas_tol=1e-8,
              inner_tol=1e-14, verbose=False)
    assert opt.solve() is opt
    assert opt.success
    np.testing.assert_allclose(opt.x, [(A + B) / 2] * 2, atol=1e-6)
    np.testing.assert_allclose(opt.y, [A - B], atol=1e-5)


def test_matches_albcd_with_equivalent_settings():
    """With an inner loop run to convergence, no penalty growth and the same mu, ALC's iterates are
    ALBCD's: the merit function and the multiplier update are identical."""
    kw = dict(feas_tol=1e-6, max_outer_iter=5, verbose=False)
    alc = ALC(blocks(), np.zeros(2), np.ones(1), objective, phi0=np.zeros(1), beta=1.0, inner_tol=1e-15,
              max_inner_iter=500, **kw).solve()
    albcd = ALBCD(blocks(), np.zeros(2), mu0=np.ones(1), rho=1.0, opt_tol=1e-12, max_inner_iter=500, **kw).solve()
    np.testing.assert_allclose(alc.y, albcd.y, atol=1e-8)
    np.testing.assert_allclose(alc.x, albcd.x, atol=1e-8)


def test_stops_early_with_one_sweep_per_update():
    """With one sweep per multiplier update (Tosserams' alternating-direction variant) and growing
    penalties, ALC's outer test is met away from the solution, yet it reports success."""
    opt = ALC(blocks(), np.zeros(2), np.ones(1), objective, beta=2.0, gamma=0.5, feas_tol=1e-8,
              max_inner_iter=1, max_outer_iter=400, verbose=False).solve()
    assert opt.success
    assert abs(opt.x[0] - (A + B) / 2) > 1e-3


def test_penalty_rule_has_no_floor():
    """Eq. 25 grows mu whenever |q| does not shrink by gamma, even when q is already satisfied."""
    opt = ALC(blocks(), np.zeros(2), np.ones(1), objective, beta=2.0, gamma=0.5, verbose=False)
    opt._update_mu(np.array([1e-12]), np.array([1e-12]))
    assert opt.mu[0] == 4.0  # mu = 2 w^2 grows by beta^2
    opt._update_mu(np.array([1e-12]), np.array([1e-11]))
    assert opt.mu[0] == 4.0


def test_mu0_not_aliased():
    mu0 = np.ones(1)
    ALC(blocks(), np.zeros(2), mu0, objective, beta=3.0, verbose=False, max_outer_iter=3).solve()
    assert mu0[0] == 1.0


def test_histories():
    opt = ALC(blocks(), np.zeros(2), np.ones(1), objective, feas_tol=1e-8, verbose=False).solve()
    n = len(opt.feas_history)
    assert len(opt.opt_history) == len(opt.merit_history) == len(opt.outer_history) == n
    assert np.all(np.isfinite(opt.opt_history))
    no_res = ALC(blocks(), np.zeros(2), np.ones(1), objective, record_residual=False, verbose=False).solve()
    assert np.all(np.isnan(no_res.opt_history))


def test_initial_penalty():
    # with v = 0 and a tiny penalty the blocks go to their targets, so q_hat = A - B
    mu0 = initial_penalty(blocks(), np.zeros(2), 1, objective, phi0=np.zeros(1), mu_small=1e-8,
                          inner_tol=1e-14, alpha=0.1)
    f_hat = objective(np.zeros(2), {})
    np.testing.assert_allclose(mu0, 2 * 0.1 * f_hat / (A - B) ** 2, rtol=1e-6)
