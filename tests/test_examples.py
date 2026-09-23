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
    "quadratic_global_circle.py": [R, 0, R],
    "quadratic_consensus_circle.py": [R, R, R, R],
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

    opt = runpy.run_path(str(EXAMPLES / folder / name))["opt"]
    plt.close("all")

    assert opt.success
    np.testing.assert_allclose(opt.x, SOLUTIONS[name], atol=5e-3)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")  # plt.show() under Agg
def test_powell():
    """Unconstrained BCD cycles on Powell's example instead of converging."""
    opt = runpy.run_path(str(EXAMPLES / "powell.py"))["opt"]
    plt.close("all")

    assert not opt.success
    assert len(opt.history) == 1 + 6 * 3  # max_inner_iter = 6 sweeps over 3 blocks
    np.testing.assert_allclose(np.abs(opt.x), 1, atol=1e-3)  # near a vertex of [-1, 1]^3 ...
    assert opt.opt_history[-1] > 1.9  # ... where the gradient does not vanish
