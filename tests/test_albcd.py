"""Tests of the ALBCD core on a two-block problem with closed-form block solves.

    min (x0 - a)^2 + (x1 - b)^2  s.t.  phi = x0 - x1 = 0

has the solution x0 = x1 = (a + b) / 2 with multiplier y = a - b.
"""

import numpy as np
import pytest

from albcd import ALBCD, Subproblem

A, B = 1.0, 3.0


class Block(Subproblem):
    """Minimizes (v - target)^2 + y * phi + mu / 2 * phi^2 over this block's entry v."""

    def __init__(self, index, target=None):
        self.target = target
        self.sign = 1.0 if index == 0 else -1.0  # d(phi)/dv
        super().__init__(index)

    def _target(self, data):
        return data["target"] if self.target is None else self.target

    def solve(self, x, y, mu, data, outputs):
        y, mu = y[0], mu[0]
        v = (2 * self._target(data) - self.sign * y + mu * self.other(x)[0]) / (2 + mu)
        outputs["x"] = self.recompose(x, v)
        outputs["phi"] = np.array([outputs["x"][0] - outputs["x"][1]])
        outputs[f"v{self.index}"] = v

    def gradient(self, x, y, mu, data):
        y, mu = y[0], mu[0]
        phi = x[0] - x[1]
        return np.atleast_1d(2 * (x[self.index] - self._target(data)) + self.sign * (y + mu * phi))


def make_solver(**kwargs):
    options = dict(subproblems=[Block(0, A), Block(1, B)], x0=np.zeros(2), mu0=np.ones(1),
                   feas_tol=1e-8, opt_tol=1e-10, max_inner_iter=100, verbose=False)
    options.update(kwargs)
    return ALBCD(**options)


def test_converges_to_known_solution():
    opt = make_solver()
    assert opt.solve() is opt
    assert opt.success
    np.testing.assert_allclose(opt.x, [(A + B) / 2] * 2, atol=1e-7)
    np.testing.assert_allclose(opt.y, [A - B], atol=1e-6)
    assert np.max(np.abs(opt.phi)) <= opt.feas_tol
    # ALBCD stops only once the last sweep is both feasible and optimal
    assert opt.history["feasibility"][-1] <= opt.feas_tol
    assert opt.history["optimality"][-1] <= opt.opt_tol
    assert opt.tf > 0


@pytest.mark.parametrize("kwargs", [
    {"max_outer_iter": 2},                                         # stops before reaching feasibility
    {"max_inner_iter": 1, "feas_tol": 1.0, "max_outer_iter": 3},  # feasible, but never optimal
])
def test_not_converged(kwargs):
    opt = make_solver(**kwargs)
    opt.solve()
    assert not opt.success
    assert opt.history["optimality"][-1] > opt.opt_tol


def test_inexact_inner_loop():
    """kappa > 0 loosens the inner tolerance while infeasible: fewer sweeps, the same solution."""
    exact, inexact = make_solver(kappa=0.0).solve(), make_solver(kappa=10.0).solve()
    assert exact.success and inexact.success
    np.testing.assert_allclose(inexact.x, exact.x, atol=1e-7)
    assert len(inexact.history["optimality"]) < len(exact.history["optimality"])


def test_verbose_output(capsys):
    make_solver(verbose=False).solve()
    assert capsys.readouterr().out == ""

    make_solver(verbose=True).solve()
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1].startswith("ALBCD converged")
    assert all(line.startswith("inner loop ") for line in lines[:-1])  # one line per outer iteration


def test_history():
    opt = make_solver(x0=[0, 0])  # a list x0 is accepted
    opt.solve()
    h = opt.history
    np.testing.assert_array_equal(h["x"][0], [0, 0])
    np.testing.assert_array_equal(h["x"][-1], opt.x)
    assert (len(h["x"]) - 1) % 2 == 0  # one entry per subproblem solve

    # feasibility and optimality are logged once per sweep (here, every two block solves)
    assert len(h["feasibility"]) == len(h["optimality"]) == (len(h["x"]) - 1) // 2
    np.testing.assert_allclose(h["feasibility"][-1], np.max(np.abs(opt.phi)))
    assert np.all(np.isfinite(h["optimality"]))


def test_data_exchange():
    opt = make_solver(subproblems=[Block(0), Block(1, B)], data0={"target": A})
    opt.solve()
    np.testing.assert_allclose(opt.x, [(A + B) / 2] * 2, atol=1e-7)
    assert opt.data["v0"] == opt.x[0] and opt.data["v1"] == opt.x[1]


def test_update_mu():
    opt = make_solver(tau=0.5, rho=2.0, max_mu=1e3, feas_tol=1e-3)
    mu = opt._update_mu(np.array([1.0, 1.0, 1.0, 900.0]), phi_new=np.array([0.4, 0.6, 1e-4, 0.6]),
                        phi_old=np.array([1.0, 1.0, 1e-4, 1.0]))
    # decreased enough / too little decrease / feasible / capped at max_mu
    np.testing.assert_array_equal(mu, [1.0, 2.0, 1.0, 1e3])


def test_update_y():
    opt = make_solver(max_y=1.0)
    y = opt._update_y(np.array([0.0, 0.5, -0.5]), mu=np.array([2.0, 2.0, 2.0]), phi=np.array([0.1, 1.0, -1.0]))
    np.testing.assert_allclose(y, [0.2, 1.0, -1.0])  # y + mu * phi, clipped to [-max_y, max_y]


def test_multipliers_are_bounded():
    opt = make_solver(max_y=0.5, max_outer_iter=5)
    opt.solve()
    assert np.all(np.abs(opt.y) <= 0.5)


def test_does_not_modify_x0():
    x0 = np.zeros(2)
    make_solver(x0=x0).solve()
    np.testing.assert_array_equal(x0, [0, 0])


def test_phi_not_set():
    class ForgetsPhi(Block):
        """Sets phi on its first solve only."""
        solves = 0

        def solve(self, x, y, mu, data, outputs):
            super().solve(x, y, mu, data, outputs)
            self.solves += 1
            if self.solves > 1:
                del outputs["phi"]

    with pytest.raises(ValueError, match="reshape"):
        make_solver(subproblems=[ForgetsPhi(0, A), Block(1, B)]).solve()


def test_phi_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        make_solver(mu0=np.ones(2)).solve()


def test_subproblem_helpers():
    sub = Block(slice(1, 3), A)
    x = np.array([0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(sub.decompose(x), [1, 2])
    np.testing.assert_array_equal(sub.other(x), [0, 3])
    np.testing.assert_array_equal(sub.recompose(x, [5, 6]), [0, 5, 6, 3])
    np.testing.assert_array_equal(x, [0, 1, 2, 3])  # recompose returns a copy


def test_subproblem_requires_solve_and_gradient():
    sub = Subproblem(slice(0, 1))  # setup() is optional
    with pytest.raises(NotImplementedError):
        sub.solve(np.zeros(1), np.zeros(0), np.zeros(0), {}, {})
    with pytest.raises(NotImplementedError):
        sub.gradient(np.zeros(1), np.zeros(0), np.zeros(0), {})
