# ALBCD

**Augmented Lagrangian block coordinate descent**

Augmented Lagrangian block coordinate descent (ALBCD) is a coordination scheme for distributed
multidisciplinary design optimization (MDO) problems. To use ALBCD, MDO problems are first decomposed into subproblems. These subproblems are formulated with relaxed constraints and a new augmemted Lagrangian merit function instead of the original objective function. The ALBCD coordination scheme iteratively solves each subproblem using the block coordinate descent algorithm in an inner loop, while an outer loop enforces the relaxed constraints using the augmented Lagrangian method. In some scenarios, ALBCD can be used to reduce computational cost and/or enable geographically distributed optimization.

## Installation

```bash
pip install git+https://github.com/nichco/ALBCD.git
```

The core package only depends on numpy. The examples require [modopt](https://github.com/LSDOlab/modopt) for optimization and either [JAX](https://docs.jax.dev) or [PyTorch](https://pytorch.org) for automatic differentiation. Some examples also require CVXOPT and/or matplotlib. To install all of the dependencies required to run the examples, install from a clone:

```bash
git clone https://github.com/nichco/ALBCD.git
cd ALBCD
pip install -e ".[examples]"
```

ModOpt can be installed from GitHub
(`pip install git+https://github.com/lsdolab/modopt.git@main`)

## Quick start

Minimize `x1^2 + x2^2 - 1.5 x1 x2` subject to `x1^2 + x2^2 >= 0.25`. The
constraint couples the two blocks and is written as the equality
`phi = 0.25 - x1^2 - x2^2 + s = 0` with a slack variable `s >= 0`. Block 1 owns the variables
`(x1, s)` and block 2 owns the variables `x2`. Each block's subproblem is solved with modopt's
SLSQP using gradients from PyTorch, so this needs the `[examples]` dependency install above.
It is a condensed version of
[examples/pytorch/quadratic_global_circle.py](https://github.com/nichco/ALBCD/blob/main/examples/pytorch/quadratic_global_circle.py).

```python
import numpy as np
import modopt as mo
import torch
from albcd import ALBCD, Subproblem

torch.set_default_dtype(torch.float64)  # modopt works in float64


def f(x1, x2):
    return x1**2 + x2**2 - 1.5 * x1 * x2


def phi(x1, s, x2):
    return 0.5**2 - (x1**2 + x2**2) + s


class Subproblem1(Subproblem):
    """Owns x1 and the slack s >= 0."""

    xl = np.array([-np.inf, 0.0])
    xu = np.array([np.inf, np.inf])

    def setup(self) -> None:
        self.add_input("x")  # [x1, s, x2]
        self.add_input("y")  # Lagrange multipliers
        self.add_input("mu")  # penalty parameters
        self.add_output("x")
        self.add_output("phi")  # coupling constraints at the new x

    def objective(self, v, other, y, mu):
        """Augmented Lagrangian as a function of this block's variables v."""
        x1, s, x2 = v[0], v[1], other[0]
        c = phi(x1, s, x2)
        return f(x1, x2) + torch.sum(y * c) + 0.5 * torch.sum(mu * c**2)

    def solve(self, inputs, outputs) -> None:
        x = inputs["x"]
        other, y, mu = (torch.as_tensor(a) for a in (self.other(x), inputs["y"], inputs["mu"]))
        obj = lambda v: self.objective(torch.as_tensor(v), other, y, mu)

        prob = mo.ProblemLite(x0=self.decompose(x), obj=lambda v: float(obj(v)),
                              grad=lambda v: torch.func.grad(obj)(torch.as_tensor(v)).numpy(),
                              xl=self.xl, xu=self.xu, name=type(self).__name__)
        optimizer = mo.SLSQP(prob, solver_options={'maxiter': 100, 'ftol': 1e-8}, turn_off_outputs=True)
        optimizer.solve()

        outputs["x"] = self.recompose(x, optimizer.results['x'])
        outputs["phi"] = np.array([phi(*outputs["x"])])

    def residual(self, inputs) -> float:
        """Projected-gradient stationarity residual: zero iff v is a KKT point for xl <= v <= xu."""
        x, y, mu = inputs["x"], inputs["y"], inputs["mu"]
        v = self.decompose(x)
        args = (torch.as_tensor(a) for a in (v, self.other(x), y, mu))
        grad = torch.func.grad(self.objective)(*args).numpy()
        return float(np.max(np.abs(v - np.clip(v - grad, self.xl, self.xu))))


class Subproblem2(Subproblem):
    """Owns x2, which is unbounded."""

    xl = np.array([-np.inf])
    xu = np.array([np.inf])

    def setup(self) -> None:
        self.add_input("x")  # [x1, s, x2]
        self.add_input("y")  # Lagrange multipliers
        self.add_input("mu")  # penalty parameters
        self.add_output("x")
        self.add_output("phi")  # coupling constraints at the new x

    def objective(self, v, other, y, mu):
        """Augmented Lagrangian as a function of this block's variables v."""
        x1, s, x2 = other[0], other[1], v[0]
        c = phi(x1, s, x2)
        return f(x1, x2) + torch.sum(y * c) + 0.5 * torch.sum(mu * c**2)

    def solve(self, inputs, outputs) -> None:
        x = inputs["x"]
        other, y, mu = (torch.as_tensor(a) for a in (self.other(x), inputs["y"], inputs["mu"]))
        obj = lambda v: self.objective(torch.as_tensor(v), other, y, mu)

        prob = mo.ProblemLite(x0=self.decompose(x), obj=lambda v: float(obj(v)),
                              grad=lambda v: torch.func.grad(obj)(torch.as_tensor(v)).numpy(),
                              xl=self.xl, xu=self.xu, name=type(self).__name__)
        optimizer = mo.SLSQP(prob, solver_options={'maxiter': 100, 'ftol': 1e-8}, turn_off_outputs=True)
        optimizer.solve()

        outputs["x"] = self.recompose(x, optimizer.results['x'])
        outputs["phi"] = np.array([phi(*outputs["x"])])

    def residual(self, inputs) -> float:
        """Projected-gradient stationarity residual: zero iff v is a KKT point for xl <= v <= xu."""
        x, y, mu = inputs["x"], inputs["y"], inputs["mu"]
        v = self.decompose(x)
        args = (torch.as_tensor(a) for a in (v, self.other(x), y, mu))
        grad = torch.func.grad(self.objective)(*args).numpy()
        return float(np.max(np.abs(v - np.clip(v - grad, self.xl, self.xu))))


opt = ALBCD(
    subproblems=[Subproblem1(index=slice(0, 2)), Subproblem2(index=slice(2, 3))],
    x0=np.array([-0.5, 0.0, 1.0]),
    mu0=np.array([1.0]),
    opt_tol=[1e-2, 1e-4],
    max_inner_iter=100,
)
opt.solve()
print(opt.success, opt.x)  # True, approximately [0.354 0. 0.354]: x1 = x2 = sqrt(2)/4 on the circle, s = 0
```

## Examples

See [examples/](https://github.com/nichco/ALBCD/tree/main/examples). Every problem below is included twice, once with gradients from JAX and once from PyTorch.

| Example | Notable features |
| --- | --- |
| `quadratic_global_circle.py` | Quadratic objective with a nonlinear inequality coupling two blocks via a slack variable; the quick start problem above. |
| `quadratic_global_linear.py` | Quadratic objective with a linear inequality coupling two blocks. |
| `quadratic_global_linear_cvxopt.py` | Same problem as `quadratic_global_linear.py`, but each block is solved by CVXOPT using exact Hessians instead of SLSQP. |
| `quadratic_consensus_circle.py` | Consensus form: each block owns a full copy of the variables and handles the circle constraint locally, with coupling constraints forcing the copies to agree. |
| `rosenbrock_consensus.py` | Consensus form of the nonconvex Rosenbrock function. |
| `proximal_rosenbrock_consensus.py` | Same problem as `rosenbrock_consensus.py`, with a proximal term added to each block's subproblem for stabilization. |

Two larger examples solve wing design problems and compare against a monolithic (single-solve) reference:

| Example | Notable features |
| --- | --- |
| [aerostruct/](https://github.com/nichco/ALBCD/tree/main/examples/pytorch/aerostruct) (JAX and PyTorch) | Aerostructural wing design: minimizes drag subject to lift = weight while sizing the spar wall thickness for a tip-deflection constraint. Aerodynamics (vortex-lattice) and structure (beam FE) are the two coupled blocks. |
| [uCRM/](https://github.com/nichco/ALBCD/tree/main/examples/pytorch/uCRM) (PyTorch only) | Multipoint wing design: minimizes average fuel burn over N missions with different payloads/ranges, each mission a block with its own copy of the wing twist coupled by consensus. |


## Tests

```bash
pip install -e ".[test]"
pytest
```

The example tests are skipped if modopt, matplotlib or the autodiff
library (JAX or PyTorch) isn't installed.

## References

The following papers describe early variants of the ALBCD coordination scheme:

```bibtex
@inproceedings{orndorff2026distributed,
  author    = {Orndorff, Nicholas C. and Lupp, Christopher and Hwang, John T.},
  title     = {{A Distributed Algorithm for Large-Scale Multidisciplinary Design Optimization With Global Constraints}},
  booktitle = {AIAA AVIATION 2026 Forum},
  year      = {2026},
  doi       = {10.2514/6.2026-4500},
}

@inproceedings{orndorff2025distributed,
  author    = {Orndorff, Nicholas C. and Hwang, John T.},
  title     = {{A Distributed Method for Solving Large-Scale Multidisciplinary Optimization Problems}},
  booktitle = {AIAA AVIATION Forum and ASCEND 2025},
  year      = {2025},
  doi       = {10.2514/6.2025-3736},
}
```