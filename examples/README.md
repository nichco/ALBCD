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

[powell.py](powell.py) is Powell's unconstrained example
(`unconstrained=True`), on which block coordinate descent cycles around six
vertices of a cube instead of converging. Each block's minimizer and residual
are explicit, so it needs only NumPy and matplotlib.

Two larger wing design examples are in
[jax/aerostruct/](jax/aerostruct/README.md) and [pytorch/aerostruct/](pytorch/aerostruct/README.md)
(aerostructural design) and, in PyTorch only, [pytorch/uCRM/](pytorch/uCRM/README.md)
(multipoint design).