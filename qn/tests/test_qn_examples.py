"""Tests of the example blocks' grad_phi() and local_multipliers() (Task 4). Slow: importing ranger's models takes
~30-40 s. Each problem is checked at the point after one ALBCD sweep (airliner with N = 2)."""

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("modopt")

import qn_core as qc

pytestmark = pytest.mark.slow


def after_one_sweep(module, *args):
    opt = module.make_solver(*args, max_inner_iter=1, max_outer_iter=1, verbose=False)
    module.compile_models(opt)
    y_sweep, mu_sweep = opt.y.copy(), opt.mu.copy()
    opt.solve()  # one sweep; ALBCD then updates y (and maybe mu) for the next outer iteration
    return opt, y_sweep, mu_sweep


@pytest.fixture(scope="module")
def airliner():
    import airliner_blocks as ab
    opt, _, _ = after_one_sweep(ab, 2)
    return ab, opt


@pytest.fixture(scope="module")
def ranger():
    import ranger_blocks as rb
    opt, _, _ = after_one_sweep(rb)
    return rb, opt


PROBLEMS = ["airliner", "ranger"]


def get(request, name):
    return request.getfixturevalue(name)


def block_objective(name, module, sp, v):
    """f_i(v), the block's objective term (no multiplier or penalty terms)."""
    if name == "airliner":
        return float(sp.outputs(np.array(v))[-1])
    if isinstance(sp, module.AeroSubproblem):
        return 1e2 * sp.grad_phi(sp.recompose(np.zeros(sp.index.stop), v), {})["outputs"]["CD"]
    return 0.0


# ----------------------------------------------------------------------------------------------------------
# 1. separability
# ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", PROBLEMS)
def test_separability(request, name):
    module, opt = get(request, name)
    lins = [sp.grad_phi(opt.x, opt.data) for sp in opt.subproblems]
    np.testing.assert_allclose(sum(lin["c"] for lin in lins), opt.phi, rtol=0, atol=1e-12)


# ----------------------------------------------------------------------------------------------------------
# 2. derivatives
# ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", PROBLEMS)
def test_derivatives(request, name):
    """Central differences of f, c and h in the scaled variables (step 1e-6) against grad_f, J and Jh."""
    module, opt = get(request, name)
    for sp in opt.subproblems:
        lin = sp.grad_phi(opt.x, opt.data)
        v = np.array(opt.x[sp.index], dtype=float)
        D = sp.scale(len(v))
        n = len(v)
        g_fd = np.zeros(n)
        J_fd = np.zeros_like(lin["J"])
        Jh_fd = np.zeros_like(lin["Jh"])
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1e-6 / D[j]
            xp, xm = opt.x.copy(), opt.x.copy()
            xp[sp.index] = v + e
            xm[sp.index] = v - e
            lp, lm = sp.grad_phi(xp, opt.data), sp.grad_phi(xm, opt.data)
            g_fd[j] = (block_objective(name, module, sp, v + e) - block_objective(name, module, sp, v - e)) / 2e-6
            J_fd[:, j] = (lp["c"] - lm["c"]) / 2e-6
            Jh_fd[:, j] = (lp["h"] - lm["h"]) / 2e-6
        # compare in scaled variables: derivative columns divided by D
        for ad, fd, label in ((lin["grad_f"] / D, g_fd, "grad_f"), (lin["J"] / D, J_fd, "J"), (lin["Jh"] / D, Jh_fd, "Jh")):
            scale = max(np.max(np.abs(ad), initial=0.0), 1e-12)
            err = np.max(np.abs(ad - fd), initial=0.0) / scale
            assert err <= 1e-5, f"{type(sp).__name__} {label}: relative error {err:.2e}"


# ----------------------------------------------------------------------------------------------------------
# 3. multiplier signs (validates §1.3)
# ----------------------------------------------------------------------------------------------------------

def solve_block_and_residuals(name, module, opt, sp):
    """Solve one block from the post-sweep point with nonzero multipliers and return the scaled residuals
    (grad_obj + Jh^T kappa) / D on the free variables for kappa, -kappa and least-squares multipliers on the
    same active rows, and the scaled tolerance of the block solver."""
    x, y, mu, data = opt.x.copy(), opt.y.copy(), opt.mu.copy(), dict(opt.data)
    sp.act_tol = 1e-8
    out = {}
    sp.solve(x, y, mu, data, out)
    v = np.array(out["x"][sp.index], dtype=float)
    kappa = sp.local_multipliers()
    xn = out["x"]
    lin = sp.grad_phi(xn, data)
    lin["v"] = v
    if name == "airliner":
        g = sp.objective_gradient(v, x, y, mu)
    else:
        import jax.numpy as jnp
        g = np.asarray(sp.grad(jnp.asarray(v), *sp.args(x, y, mu, data)))
    D = sp.scale(len(v))
    r = g + lin["Jh"].T @ kappa
    free, rows, _ = qc.active_sets(lin, kappa, r, 1e-8, 1e-8, getattr(sp, "tight_tol", None))
    res = np.max(np.abs((r / D)[free]), initial=0.0)
    res_flip = np.max(np.abs(((g - lin["Jh"].T @ kappa) / D)[free]), initial=0.0)
    G = (lin["Jh"][rows] / D)[:, free]
    if len(rows):
        k_ls = np.linalg.lstsq(G.T, -(g / D)[free], rcond=None)[0]
        res_ls = np.max(np.abs((g / D)[free] + G.T @ k_ls), initial=0.0)
    else:
        res_ls = np.max(np.abs((g / D)[free]), initial=0.0)
    if name == "ranger":
        pi = np.asarray(sp._optimizer_results["pi"])
        tol = 1e-8 * (1 + np.max(np.abs(pi), initial=0.0))  # OpenSQP's scaled opt_tol
    else:
        tol = None  # SLSQP's ftol is no stationarity tolerance
    return res, res_flip, res_ls, tol, kappa, rows


@pytest.mark.parametrize("name", PROBLEMS)
def test_multiplier_signs(request, name):
    module, opt = get(request, name)
    for sp in opt.subproblems:
        res, res_flip, res_ls, tol, kappa, rows = solve_block_and_residuals(name, module, opt, sp)
        label = type(sp).__name__
        if tol is not None:
            # OpenSQP: the residual is at the solver's own tolerance
            assert res <= 10 * tol, f"{label}: residual {res:.2e} > 10 x tolerance {tol:.2e}"
        else:
            # SLSQP stops on the change in the objective: its stationarity floor (~1e-6..1e-5 here) is the
            # reference, i.e. the converted multipliers are as good as the best multipliers on the same rows
            assert res <= 10 * max(res_ls, 1e-12), f"{label}: residual {res:.2e} vs least squares {res_ls:.2e}"
        if np.any(kappa != 0):
            assert res_flip > 10 * res, f"{label}: flipping kappa's sign does not hurt ({res_flip:.2e} vs {res:.2e})"


def test_airliner_kappa_signs(airliner):
    """kappa <= 0 on rows active at cl, >= 0 on rows active at cu."""
    module, opt = airliner
    for sp in opt.subproblems:
        kappa = sp.local_multipliers()
        lin = sp.grad_phi(opt.x, opt.data)
        two_sided = (lin["cl"] != lin["cu"])
        near_cl = two_sided & (np.abs(lin["h"] - lin["cl"]) <= 1e-6)
        near_cu = two_sided & (np.abs(lin["h"] - lin["cu"]) <= 1e-6)
        assert np.all(kappa[near_cl & ~near_cu] <= 1e-8)
        assert np.all(kappa[near_cu & ~near_cl] >= -1e-8)
        assert np.count_nonzero(kappa[two_sided]) > 0


# ----------------------------------------------------------------------------------------------------------
# 4. outputs
# ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", PROBLEMS)
def test_outputs_match_solve(request, name):
    module, opt = get(request, name)
    for sp in opt.subproblems:
        outputs = sp.grad_phi(opt.x, opt.data)["outputs"]
        for key, value in outputs.items():
            np.testing.assert_allclose(value, opt.data[key], rtol=1e-10, atol=1e-10)
    if name == "ranger":
        keys = set().union(*(sp.grad_phi(opt.x, opt.data)["outputs"] for sp in opt.subproblems))
        assert keys == {"aero_loads", "CD", "weight"}


# ----------------------------------------------------------------------------------------------------------
# 5. predicted outputs ("aladin" mode)
# ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", PROBLEMS)
def test_predict_outputs_first_order(request, name):
    """predict_outputs(lin, dv) matches grad_phi's outputs at v + dv to first order: the error shrinks ~4x when dv
    halves, and is small next to the change itself. Blocks without shared outputs predict nothing."""
    module, opt = get(request, name)
    rng = np.random.default_rng(0)
    for sp in opt.subproblems:
        lin = sp.grad_phi(opt.x, opt.data)
        v = np.array(opt.x[sp.index], dtype=float)
        D = sp.scale(len(v))
        if not lin["outputs"]:
            assert sp.predict_outputs(lin, np.zeros_like(v)) == {}
            continue
        direction = rng.standard_normal(len(v)) / D  # a unit-size move in the scaled variables
        errors, changes = [], []
        for size in (1e-3, 5e-4):
            dv = size * direction
            dv = np.clip(v + dv, lin["xl"], lin["xu"]) - v
            pred = sp.predict_outputs(lin, dv)
            assert set(pred) == set(lin["outputs"])
            actual = sp.grad_phi(sp.recompose(opt.x, v + dv), opt.data)["outputs"]
            err = max(np.max(np.abs(np.asarray(pred[k]) - np.asarray(actual[k]))) for k in pred)
            change = max(np.max(np.abs(np.asarray(actual[k]) - np.asarray(lin["outputs"][k]))) for k in pred)
            errors.append(err)
            changes.append(change)
        label = type(sp).__name__
        assert errors[0] <= 1e-2 * changes[0], f"{label}: error {errors[0]:.2e} vs change {changes[0]:.2e}"
        assert errors[1] <= 0.4 * errors[0] or errors[0] <= 1e-12, f"{label}: errors {errors} not quadratic"
