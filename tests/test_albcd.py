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

    def setup(self):
        if self.target is None:
            self.add_input("target")  # read from the solver's data dictionary
        self.add_output("x")
        self.add_output("phi")
        self.add_output(f"v{self.index}")  # written to the solver's data dictionary

    def _target(self, inputs):
        return inputs["target"] if self.target is None else self.target

    def solve(self, inputs, outputs):
        x, y, mu = inputs["x"], inputs["y"][0], inputs["mu"][0]
        v = (2 * self._target(inputs) - self.sign * y + mu * self.other(x)[0]) / (2 + mu)
        outputs["x"] = self.recompose(x, v)
        outputs["phi"] = np.array([outputs["x"][0] - outputs["x"][1]])
        outputs[f"v{self.index}"] = v

    def residual(self, inputs):
        x, y, mu = inputs["x"], inputs["y"][0], inputs["mu"][0]
        phi = x[0] - x[1]
        return abs(2 * (x[self.index] - self._target(inputs)) + self.sign * (y + mu * phi))


def make_solver(**kwargs):
    options = dict(subproblems=[Block(0, A), Block(1, B)], x0=np.zeros(2), mu0=np.ones(1),
                   feas_tol=1e-8, opt_tol=1e-10, max_inner_iter=100, verbose=False)
    options.update(kwargs)
    return ALBCD(**options)


@pytest.mark.parametrize("opt_tol", [1e-10, [1e-2, 1e-10]])
def test_converges_to_known_solution(opt_tol):
    opt = make_solver(opt_tol=opt_tol)
    opt.solve()
    assert opt.success
    np.testing.assert_allclose(opt.x, [(A + B) / 2] * 2, atol=1e-7)
    np.testing.assert_allclose(opt.y, [A - B], atol=1e-6)
    assert np.max(np.abs(opt.phi)) <= opt.feas_tol
    assert opt.feas_history[-1] <= opt.feas_tol
    assert opt.tf > 0


@pytest.mark.parametrize("kwargs", [
    {"max_outer_iter": 2},                    # stops before reaching feasibility
    {"max_inner_iter": 1, "feas_tol": 1e-1},  # feasible, but the inner loop never reaches opt_tol
])
def test_not_converged(kwargs):
    opt = make_solver(**kwargs)
    opt.solve()
    assert not opt.success


def test_verbose_output(capsys):
    opt = make_solver(opt_tol=[1e-2, 1e-10], verbose=True)
    opt.solve()
    lines = capsys.readouterr().out.splitlines()
    n = len(opt.feas_history)  # outer iterations performed
    assert "Phase 1 complete, starting phase 2" in lines
    assert lines[-2].startswith(f"outer {n:3d} | feas ")
    assert lines[-1].startswith(f"Converged after {n} outer iterations")


def test_history():
    opt = make_solver(x0=[0, 0])  # a list x0 is accepted
    opt.solve()
    np.testing.assert_array_equal(opt.history[0], [0, 0])
    np.testing.assert_array_equal(opt.history[-1], opt.x)
    assert (len(opt.history) - 1) % 2 == 0  # one entry per subproblem solve

    opt = make_solver(save=False)
    opt.solve()
    assert opt.history == [] and opt.feas_history == []


def test_data_exchange():
    opt = make_solver(subproblems=[Block(0), Block(1, B)], data0={"target": A})
    opt.solve()
    np.testing.assert_allclose(opt.x, [(A + B) / 2] * 2, atol=1e-7)
    assert opt.data["v0"] == opt.x[0] and opt.data["v1"] == opt.x[1]


def test_update_mu():
    opt = make_solver(mu0=[1.0, 1.0, 1.0, 900.0], tau=0.5, rho=2.0, max_mu=1e3, feas_tol=1e-3)
    opt._update_mu(c_new=np.array([0.4, 0.6, 1e-4, 0.6]), c_old=np.array([1.0, 1.0, 1e-4, 1.0]))
    # decreased enough / too little decrease / feasible / capped at max_mu
    np.testing.assert_array_equal(opt.mu, [1.0, 2.0, 1.0, 1e3])


def test_multipliers_are_bounded():
    opt = make_solver(max_y=0.5, max_outer_iter=5)
    opt.solve()
    assert np.all(np.abs(opt.y) <= 0.5)


def test_does_not_modify_x0():
    x0 = np.zeros(2)
    make_solver(x0=x0).solve()
    np.testing.assert_array_equal(x0, [0, 0])


@pytest.mark.parametrize("kwargs", [{"opt_tol": []}, {"max_inner_iter": 0}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        make_solver(**kwargs)


def test_missing_phi_output():
    class NoPhi(Block):
        def setup(self):
            self.add_output("x")

    with pytest.raises(ValueError, match="phi"):
        make_solver(subproblems=[NoPhi(0, A), Block(1, B)])


def test_phi_not_set():
    class ForgetsPhi(Block):
        def solve(self, inputs, outputs):
            outputs["x"] = inputs["x"]

    with pytest.raises(RuntimeError, match="phi"):
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


def test_subproblem_requires_setup():
    with pytest.raises(NotImplementedError):
        Subproblem(slice(0, 1))
