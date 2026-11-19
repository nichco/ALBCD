"""Tests of QuasiNewtonALBCD on small problems with exact block solves (Task 3).

Problem 1 (as in tests/test_albcd.py, but with a new Block class):

    min (x0 - 1)^2 + (x1 - 3)^2  s.t.  phi = x0 - x1 = 0,      solution x = [2, 2], y = -2.

Problem 2: two blocks with copies and local constraints (the structure ranger has)

    block A owns (a, w): f = (a - 1)^2,  a^2 - w = 0,           c_A = [a^3; -w]
    block B owns (l, t): f = (t - 2)^2,  l + t <= cu, t >= 0,    c_B = [-l; t^2]

with cu = 10 (the plan's problem, local inequality inactive at the solution) and cu = 4 (active). Both blocks'
augmented-Lagrangian subproblems reduce to roots of univariate polynomials, so the block solves are exact.
"""

import numpy as np
import pytest

from qn_albcd import QuasiNewtonALBCD
from qn_subproblem import QNSubproblem

A_, B_ = 1.0, 3.0
MODES = ["dual", "primal_dual", "adaptive"]


# ----------------------------------------------------------------------------------------------------------
# problem 1
# ----------------------------------------------------------------------------------------------------------

class Block(QNSubproblem):
    """Minimizes (v - target)^2 + y * phi + mu / 2 * phi^2 over this block's entry v (closed form)."""

    def __init__(self, i, target):
        self.i = i
        self.target = target
        self.sign = 1.0 if i == 0 else -1.0  # d(phi)/dv
        super().__init__(slice(i, i + 1))

    def solve(self, x, y, mu, data, outputs):
        y, mu = y[0], mu[0]
        v = (2 * self.target - self.sign * y + mu * x[1 - self.i]) / (2 + mu)
        outputs["x"] = self.recompose(x, [v])
        outputs["phi"] = np.array([outputs["x"][0] - outputs["x"][1]])

    def gradient(self, x, y, mu, data):
        y, mu = y[0], mu[0]
        phi = x[0] - x[1]
        return np.atleast_1d(2 * (x[self.i] - self.target) + self.sign * (y + mu * phi))

    def grad_phi(self, x, data):
        v = x[self.i]
        return dict(grad_f=np.array([2 * (v - self.target)]), c=np.array([self.sign * v]), J=np.array([[self.sign]]),
                    h=np.zeros(0), Jh=np.zeros((0, 1)), cl=np.zeros(0), cu=np.zeros(0),
                    xl=np.array([-np.inf]), xu=np.array([np.inf]), outputs={})

    def local_multipliers(self):
        return np.zeros(0)


def make_solver(**kwargs):
    options = dict(subproblems=[Block(0, A_), Block(1, B_)], x0=np.zeros(2), mu0=np.ones(1),
                   feas_tol=1e-8, opt_tol=1e-12, max_inner_iter=100, max_outer_iter=100, kkt_tol=1e-10,
                   verbose=False)
    options.update(kwargs)
    return QuasiNewtonALBCD(**options)


def n_sweeps(opt):
    return opt.history["event"].count("sweep")


@pytest.mark.parametrize("B0, refine", [(2.0, 0), (3.0, 60)])
def test_exact_curvature_converges_in_one_step(B0, refine):
    """B_0 = 2 (the Lagrangian Hessian) with no refinement solves the true KKT system. B_0 = 2 + mu (the exact
    block AL Hessian) solves it once the refinement has converged (one round leaves a factor mu / (2 + mu))."""
    opt = make_solver(mode="primal_dual", B0={0: [[B0]], 1: [[B0]]}, refine=refine, kkt_tol=1e-12).solve()
    events = opt.history["event"]
    assert opt.success
    assert events[:2] == ["sweep", "qn_accept"]
    assert opt.history["theta"][1] <= 1e-12
    np.testing.assert_allclose(opt.x, [2, 2], atol=1e-12)
    np.testing.assert_allclose(opt.y, [-2], atol=1e-12)


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_converges(mode):
    opt = make_solver(mode=mode).solve()
    assert opt.success, opt.status
    assert opt.theta <= 1e-10
    np.testing.assert_allclose(opt.x, [2, 2], atol=1e-9)
    np.testing.assert_allclose(opt.y, [-2], atol=1e-8)
    # history: one entry per event, aligned keys
    h = opt.history
    n = len(h["event"])
    assert all(len(h[k]) == n for k in ("theta", "time", "t_parallel", "n_solves", "n_linearize", "n_pairs"))
    assert np.all(np.diff(h["t_parallel"]) >= 0) and np.all(np.diff(h["time"]) >= 0)
    assert len(h["optimality"]) == len(h["feasibility"]) == n_sweeps(opt)
    assert len(h["x"]) == 1 + h["n_solves"][-1]


def test_adaptive_skips_sweeps():
    pd = make_solver(mode="primal_dual").solve()
    ad = make_solver(mode="adaptive").solve()
    assert pd.success and ad.success
    assert n_sweeps(ad) < n_sweeps(pd)


def test_all_steps_rejected_falls_back_to_albcd():
    opt = make_solver(mode="primal_dual", accept=0.0, max_outer_iter=200).solve()
    events = opt.history["event"]
    assert opt.success and opt.theta <= 1e-10
    assert "qn_reject" in events and "qn_accept" not in events
    assert set(events) <= {"sweep", "qn_reject", "dual_fallback", "first_order", "mu_growth"}
    np.testing.assert_allclose(opt.x, [2, 2], atol=1e-9)


def test_requires_qn_blocks():
    from albcd import Subproblem

    class Plain(Subproblem):
        pass

    with pytest.raises(TypeError, match="QNSubproblem"):
        make_solver(subproblems=[Plain(slice(0, 1)), Block(1, B_)]).solve()


def test_rejects_nonseparable_coupling():
    class WrongC(Block):
        def grad_phi(self, x, data):
            lin = super().grad_phi(x, data)
            lin["c"] = lin["c"] + 1.0
            return lin

    with pytest.raises(ValueError, match="separable"):
        make_solver(subproblems=[WrongC(0, A_), Block(1, B_)]).solve()


# ----------------------------------------------------------------------------------------------------------
# problem 2: copies and local constraints
# ----------------------------------------------------------------------------------------------------------

def real_roots(coeffs):
    r = np.roots(coeffs)
    return np.real(r[np.abs(np.imag(r)) <= 1e-9 * np.maximum(1.0, np.abs(r))])


def polish_root(p, x):
    """Newton on the polynomial p (in np.polyval order) from x, a root of it (to clean np.roots' round-off)."""
    dp = np.polyder(p)
    for _ in range(5):
        d = np.polyval(dp, x)
        if d == 0:
            break
        x = x - np.polyval(p, x) / d
    return x


class BlockA(QNSubproblem):
    """Owns (a, w): f = (a - 1)^2, local a^2 - w = 0, c_A = [a^3, -w]."""

    def solve(self, x, y, mu, data, outputs):
        _, _, l, t = x
        # eliminate w = a^2: g(a) = (a-1)^2 + y1 (a^3 - l) + y2 (t^2 - a^2) + mu1/2 (a^3 - l)^2 + mu2/2 (t^2 - a^2)^2
        P = np.polynomial.Polynomial
        a = P([0, 1])
        g = (a - 1) ** 2 + y[0] * (a ** 3 - l) + y[1] * (t ** 2 - a ** 2) + 0.5 * mu[0] * (a ** 3 - l) ** 2 \
            + 0.5 * mu[1] * (t ** 2 - a ** 2) ** 2
        dg = g.deriv()
        roots = [polish_root(dg.coef[::-1], r) for r in real_roots(dg.coef[::-1])]
        av = min(roots, key=lambda r: g(r))
        wv = av ** 2
        # stationarity in w: dAL/dw - kappa = 0 (h = a^2 - w, dh/dw = -1)
        self._kappa = np.array([-y[1] - mu[1] * (t ** 2 - wv)])
        outputs["x"] = np.array([av, wv, l, t])
        outputs["phi"] = np.array([av ** 3 - l, t ** 2 - wv])

    def gradient(self, x, y, mu, data):
        a, w, l, t = x
        phi = np.array([a ** 3 - l, t ** 2 - w])
        lam = y + mu * phi
        g = np.array([2 * (a - 1) + 3 * a ** 2 * lam[0], -lam[1]])
        return g + np.array([2 * a, -1.0]) * self._kappa[0]

    def grad_phi(self, x, data):
        a, w = x[0], x[1]
        return dict(grad_f=np.array([2 * (a - 1), 0.0]), c=np.array([a ** 3, -w]),
                    J=np.array([[3 * a ** 2, 0.0], [0.0, -1.0]]), h=np.array([a ** 2 - w]),
                    Jh=np.array([[2 * a, -1.0]]), cl=np.zeros(1), cu=np.zeros(1),
                    xl=np.full(2, -np.inf), xu=np.full(2, np.inf), outputs={})

    def local_multipliers(self):
        return self._kappa.copy()


class BlockB(QNSubproblem):
    """Owns (l, t): f = (t - 2)^2, local l + t <= cu, t >= 0, c_B = [-l, t^2]."""

    def __init__(self, index, cu):
        self.cu = cu
        super().__init__(index)

    def solve(self, x, y, mu, data, outputs):
        a, w = x[0], x[1]
        P = np.polynomial.Polynomial
        T = P([0, 1])
        # unconstrained in l: dAL/dl = -y1 - mu1 (a^3 - l) = 0; t from a cubic
        gt = (T - 2) ** 2 + y[1] * (T ** 2 - w) + 0.5 * mu[1] * (T ** 2 - w) ** 2

        def best_t(g):
            dg = g.deriv()
            cands = [polish_root(dg.coef[::-1], r) for r in real_roots(dg.coef[::-1])]
            cands = [c for c in cands if c >= 0] + [0.0]
            return min(cands, key=lambda r: g(r))

        t = best_t(gt)
        l = a ** 3 + y[0] / mu[0]
        kappa = 0.0
        if l + t > self.cu:
            # active: l = cu - t
            g2 = gt + y[0] * (a ** 3 - (self.cu - T)) + 0.5 * mu[0] * (a ** 3 - (self.cu - T)) ** 2
            t = best_t(g2)
            l = self.cu - t
            kappa = y[0] + mu[0] * (a ** 3 - l)  # dAL/dl + kappa = 0, dAL/dl = -(y1 + mu1 (a^3 - l))
        self._kappa = np.array([kappa])
        outputs["x"] = np.array([a, w, l, t])
        outputs["phi"] = np.array([a ** 3 - l, t ** 2 - w])

    def gradient(self, x, y, mu, data):
        a, w, l, t = x
        phi = np.array([a ** 3 - l, t ** 2 - w])
        lam = y + mu * phi
        g = np.array([-lam[0], 2 * (t - 2) + 2 * t * lam[1]]) + self._kappa[0]
        if t <= 0:
            g[1] = min(g[1], 0.0)
        return g

    def grad_phi(self, x, data):
        l, t = x[2], x[3]
        return dict(grad_f=np.array([0.0, 2 * (t - 2)]), c=np.array([-l, t ** 2]),
                    J=np.array([[-1.0, 0.0], [0.0, 2 * t]]), h=np.array([l + t]), Jh=np.array([[1.0, 1.0]]),
                    cl=np.array([-np.inf]), cu=np.array([self.cu]),
                    xl=np.array([-np.inf, 0.0]), xu=np.full(2, np.inf), outputs={})

    def local_multipliers(self):
        return self._kappa.copy()


def reference(cu):
    from scipy.optimize import minimize
    cons = [dict(type="eq", fun=lambda z: np.array([z[0] ** 2 - z[1], z[0] ** 3 - z[2], z[3] ** 2 - z[1]]),
                 jac=lambda z: np.array([[2 * z[0], -1, 0, 0], [3 * z[0] ** 2, 0, -1, 0], [0, -1, 0, 2 * z[3]]])),
            dict(type="ineq", fun=lambda z: np.array([cu - z[2] - z[3]]), jac=lambda z: np.array([[0, 0, -1.0, -1.0]]))]
    res = minimize(lambda z: (z[0] - 1) ** 2 + (z[3] - 2) ** 2, np.array([1.0, 1.0, 1.0, 1.0]),
                   jac=lambda z: np.array([2 * (z[0] - 1), 0, 0, 2 * (z[3] - 2)]), method="SLSQP",
                   bounds=[(None, None)] * 3 + [(0, None)], constraints=cons,
                   options=dict(ftol=1e-15, maxiter=1000))
    assert res.success
    return res.x


def solve_copies(mode, cu, mu0, **kwargs):
    return QuasiNewtonALBCD(subproblems=[BlockA(slice(0, 2)), BlockB(slice(2, 4), cu)],
                            x0=np.array([1.0, 1.0, 1.0, 1.0]), mu0=np.full(2, mu0), feas_tol=1e-8, opt_tol=1e-11,
                            max_inner_iter=100, max_outer_iter=100, kkt_tol=1e-9, mode=mode, verbose=False,
                            **kwargs).solve()


@pytest.mark.parametrize("cu", [10.0, 4.0])
@pytest.mark.parametrize("mode", MODES)
def test_copies_and_local_constraints(mode, cu):
    """mu0 = 0.1. See test_copies_penalty_dominated_tail for mu0 = 1."""
    z_ref = reference(cu)
    opt = solve_copies(mode, cu, 0.1)
    assert opt.success, opt.status
    assert opt.theta <= 1e-9
    np.testing.assert_allclose(opt.x, z_ref, atol=1e-6)
    if mode != "dual":
        assert "qn_accept" in opt.history["event"]


@pytest.mark.xfail(strict=True, reason="known limitation: with mu0 = 1 the block-diagonal penalty curvature "
                   "(Z^T P Z ~ 109) swamps the true reduced Hessian (Z^T H Z = 4) along the problem's one null-space "
                   "direction; one refinement round contracts that error only by 109/113, so V2 converges linearly "
                   "(theta ~1e-6 after 100 outer iterations)")
def test_copies_penalty_dominated_tail():
    opt = solve_copies("primal_dual", 10.0, 1.0)
    assert opt.success
