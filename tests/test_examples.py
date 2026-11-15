"""Run each example script and compare its solution with the known optimum.

Skipped unless matplotlib is installed, and, for the examples that use them,
modopt (LSDOlab) and the example's autodiff backend (jax or torch).
"""

import runpy
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # non-interactive, so plt.show() returns immediately
import matplotlib.pyplot as plt  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
R = np.sqrt(2) / 4  # x1^2 + x2^2 - 1.5 x1 x2 on the circle of radius 0.5 is minimized at x1 = x2 = +-R
# From x0 = (-0.5, 1), Rosenbrock reaches this local minimum on the circle, not the global one at (1, 1)
ROSENBROCK_LOCAL = [-0.4536, 0.2105]

# example file -> solution reached from the example's x0
SOLUTIONS = {
    "quadratic_global_linear.py": [0, 0, 0],
    "quadratic_global_linear_cvxopt.py": [0, 0, 0],
    "quadratic_global_failure.py": [-0.5, -1],  # where plain BCD stalls, not the optimum (the origin)
    "no_feasible_solution.py": [0, 0],
    "quadratic_global_circle.py": [R, 0, R],
    "quadratic_consensus_circle.py": [R, R, R, R],
    "quadratic_consensus_linear.py": [0, 0, 0, 0],
    "rosenbrock_consensus.py": ROSENBROCK_LOCAL * 2,
    "proximal_rosenbrock_consensus.py": ROSENBROCK_LOCAL * 2,
    "2d_rosenbrock.py": [1, 1],
}


@pytest.mark.parametrize("backend, folder", [("jax", "jax"), ("torch", "pytorch")])
@pytest.mark.parametrize("name", SOLUTIONS)
def test_example(name, backend, folder):
    pytest.importorskip("modopt")
    matplotlib.use("Agg")  # modopt switches the backend to TkAgg when first imported
    pytest.importorskip(backend)
    if "cvxopt" in name:
        pytest.importorskip("cvxopt")

    script = runpy.run_path(str(EXAMPLES / folder / name))
    plt.close("all")

    if "opt" in script:  # an ALBCD example
        opt = script["opt"]
        assert opt.success
        x = opt.x
    else:  # a plain BCD example: history holds x0, then x after each solve of a one-variable block
        history = script["history"]
        max_sweeps = script["bcd"].__defaults__[-1]
        assert len(history) - 1 < history.shape[1] * max_sweeps  # stopped on its tolerance
        x = history[-1]
    np.testing.assert_allclose(x, SOLUTIONS[name], atol=5e-3)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")  # plt.show() under Agg
def test_powell():
    """Unconstrained BCD cycles on Powell's example instead of converging."""
    script = runpy.run_path(str(EXAMPLES / "powell.py"))
    plt.close("all")

    history = script["history"]
    assert len(history) == 1 + 6 * 3  # all max_sweeps = 6 sweeps over 3 blocks, so it never converged
    np.testing.assert_allclose(np.abs(history[-1]), 1, atol=1e-3)  # near a vertex of [-1, 1]^3 ...
    assert np.max(np.abs(script["gradient"](history[-1]))) > 1.9  # ... where the gradient does not vanish


def test_circle_animation():
    """The animation's solve (the video itself is not written) reaches the quick start solution."""
    pytest.importorskip("modopt")
    matplotlib.use("Agg")
    opt, log = runpy.run_path(str(EXAMPLES / "albcd_circle_animation.py"))["run_albcd"]()

    assert opt.success
    np.testing.assert_allclose(opt.x, [R, 0, R], atol=5e-3)
    assert len(log) == len(opt.history["x"]) - 1  # one (block, y, mu) entry per block solve
