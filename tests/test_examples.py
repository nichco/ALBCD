"""Run each example script and compare its solution with the known optimum.

Skipped unless modopt (LSDOlab), matplotlib and the example's autodiff
backend (jax or torch) are installed.
"""

import runpy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("modopt")
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
}


@pytest.mark.parametrize("backend, folder", [("jax", "jax"), ("torch", "pytorch")])
@pytest.mark.parametrize("name", SOLUTIONS)
def test_example(name, backend, folder):
    pytest.importorskip(backend)
    if "cvxopt" in name:
        pytest.importorskip("cvxopt")

    opt = runpy.run_path(str(EXAMPLES / folder / name))["opt"]
    plt.close("all")

    assert opt.success
    np.testing.assert_allclose(opt.x, SOLUTIONS[name], atol=5e-3)
