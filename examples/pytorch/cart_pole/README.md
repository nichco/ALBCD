# Cart-pole co-design under uncertainty

A larger ALBCD example, with gradients from PyTorch; the JAX version is in
[jax/cart_pole/](../../jax/cart_pole/README.md) and converges to the same solution.
A cart swings a pole up from hanging below it to balancing above it while moving
0.8 m in a fixed time, with the dynamics transcribed by trapezoidal collocation. The pole's
length and mass are designed together with the trajectory to minimize the mean control
effort over N scenarios of the uncertain gravity and cart and pole friction, sampled with
a Latin hypercube. Each scenario is an ALBCD block with its own trajectory and its own
copy of the pole design, and the coupling constraints force the copies to agree.
N is 2 by default; pass another as an argument, e.g. `python cart_pole.py 3` (and
likewise for `monolithic.py`). The solve saves its optimality, feasibility and error histories
to `convergence_N{N}.npz`.

| File | Contents |
| --- | --- |
| `cart_pole.py` | The ALBCD solve. Run this. |
| `monolithic.py` | The same problem solved with a single SLSQP; writes the reference solution `monolithic_solution_N{N}.npz` |
| `models.py` | The scenarios, the cart-pole dynamics and the collocation constraints, shared by both scripts |

```bash
python examples/pytorch/cart_pole/cart_pole.py
```
