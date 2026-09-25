# Examples

Every example is included twice: in `jax/` with gradients from JAX, and in
`pytorch/` with gradients from PyTorch. The block subproblems are solved with
[modopt](https://github.com/LSDOlab/modopt). Install the dependencies from the
repository root with

```bash
pip install -e ".[examples]"
```

and run any example directly, e.g. `python examples/pytorch/rosenbrock_consensus.py`.

| Example | Notable features |
| --- | --- |
| `quadratic_global_circle.py` | Quadratic objective with a nonlinear inequality coupling two blocks via a slack variable. |
| `quadratic_global_linear.py` | Quadratic objective with a linear inequality coupling two blocks. |
| `quadratic_global_linear_cvxopt.py` | Same problem as `quadratic_global_linear.py`, but each block is solved by CVXOPT using exact Hessians instead of SLSQP. |
| `quadratic_consensus_circle.py` | Consensus form: each block owns a full copy of the variables and handles the circle constraint locally, with coupling constraints forcing the copies to agree. |
| `rosenbrock_consensus.py` | Consensus form of the nonconvex Rosenbrock function. |
| `proximal_rosenbrock_consensus.py` | Same problem as `rosenbrock_consensus.py`, with a proximal term added to each block's subproblem for stabilization. |
| `2d_rosenbrock.py` | Unconstrained (`unconstrained=True`) two-dimensional Rosenbrock function with one variable per block: plain block coordinate descent zigzags along the valley to the minimum. |
| `scalable_test_problem.py` | Monolithic reference for `scalable_test_problem_albcd.py`: `N` Rosenbrock subproblems sharing `n0` global variables, each with `ni` local variables and a sphere constraint, plus one global linear inequality, solved with a single SLSQP. |
| `scalable_test_problem_albcd.py` | ALBCD version of `scalable_test_problem.py`, with adjustable sizes `N`, `n0` and `ni`: each block owns a local copy of the global variables, kept in agreement by spanning-tree (chain) consensus constraints, and handles its sphere constraint locally. The global inequality enters the augmented Lagrangian through a slack variable. |

[powell.py](powell.py) is Powell's unconstrained example
(`unconstrained=True`), on which block coordinate descent cycles around six
vertices of a cube instead of converging. Each block's minimizer and residual
are explicit, so it needs only NumPy and matplotlib.

Two larger wing design examples are in
[jax/aerostruct/](jax/aerostruct/README.md) and [pytorch/aerostruct/](pytorch/aerostruct/README.md)
(aerostructural design) and in [jax/uCRM/](jax/uCRM/README.md) and
[pytorch/uCRM/](pytorch/uCRM/README.md) (multipoint design). A cart-pole
co-design problem under uncertainty is in [jax/cart_pole/](jax/cart_pole/README.md) and
[pytorch/cart_pole/](pytorch/cart_pole/README.md).