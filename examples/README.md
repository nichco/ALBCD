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
| `quadratic_global_failure.py` | Same problem as `quadratic_global_linear.py`, but run as plain BCD with the coupling constraint handled locally in each block; it stalls on the constraint line at (-0.5, -1) instead of reaching the origin. |
| `no_feasible_solution.py` | Plain BCD with the two wedge constraints `x1 - 0.5 x2 >= 0` and `x1 + 0.5 x2 >= 0` handled locally in each block; block 2's local problem is infeasible when `x1 < 0`. From (1, 0.5) the iterates zigzag to the origin. |
| `quadratic_consensus_circle.py` | Consensus form: each block owns a full copy of the variables and handles the circle constraint locally, with coupling constraints forcing the copies to agree. |
| `quadratic_consensus_linear.py` | Same consensus setup as `quadratic_consensus_circle.py`, but each block handles the linear constraint `x1 - 0.5 x2 >= 0` locally. |
| `rosenbrock_consensus.py` | Consensus form of the nonconvex Rosenbrock function. |
| `proximal_rosenbrock_consensus.py` | Same problem as `rosenbrock_consensus.py`, with a proximal term added to each block's subproblem for stabilization. |
| `2d_rosenbrock.py` | Unconstrained (`unconstrained=True`) two-dimensional Rosenbrock function with one variable per block: plain block coordinate descent zigzags along the valley to the minimum. |
| `wing_design.py` | Unconstrained (`unconstrained=True`) two-variable wing design from [Engineering Design Optimization](https://mdobook.github.io/): minimize the power required over span and chord, one variable per block, with bounds; compared against a monolithic SLSQP solve. |
| `scalable_test_problem.py` | Monolithic reference for `scalable_test_problem_albcd.py`: `N` Rosenbrock subproblems sharing `n0` global variables, each with `ni` local variables and a sphere constraint, plus one global linear inequality, solved with a single SLSQP. |
| `scalable_test_problem_albcd.py` | ALBCD version of `scalable_test_problem.py`, with adjustable sizes `N`, `n0` and `ni`: each block owns a local copy of the global variables, kept in agreement by spanning-tree (chain) consensus constraints, and handles its sphere constraint locally. The global inequality enters the augmented Lagrangian through a slack variable. |

[powell.py](powell.py) is Powell's unconstrained example
(`unconstrained=True`), on which block coordinate descent cycles around six
vertices of a cube instead of converging. Each block's minimizer and residual
are explicit, so it needs only NumPy and matplotlib.

[circle_animation.py](circle_animation.py) renders a video of ALBCD solving the
`quadratic_global_circle.py` problem: the axis-aligned block updates on contours of the
augmented Lagrangian, the landscape morphing as the multiplier and penalty are updated
between outer iterations, and the feasibility and optimality histories. Its gradients are
written out by hand, so it needs modopt, matplotlib and ffmpeg, but no autodiff library.

Two larger wing design examples are in
[jax/aerostruct/](jax/aerostruct/README.md) and [pytorch/aerostruct/](pytorch/aerostruct/README.md)
(aerostructural design), in [jax/cessna/](jax/cessna/README.md) and
[pytorch/cessna/](pytorch/cessna/README.md) (aerostructural design of a Cessna 182-like
wing with stress and minimum-gauge constraints) and in [jax/uCRM/](jax/uCRM/README.md) and
[pytorch/uCRM/](pytorch/uCRM/README.md) (multipoint design). A cart-pole
co-design problem under uncertainty is in [jax/cart_pole/](jax/cart_pole/README.md) and
[pytorch/cart_pole/](pytorch/cart_pole/README.md).