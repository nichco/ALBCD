# Aerostructural wing design

A larger ALBCD example, in JAX. It minimizes the drag of a
tapered wing over its twist, subject to lift = weight, while sizing the wall thickness of its
tubular spar so that both wing tips deflect by 0.1 m. The aerodynamics and the structure are
the two ALBCD blocks, coupled through copies of the aero loads and the weight.

Each subproblem compiles its objective, local constraints and their derivatives once with
`jax.jit`, passing the values that change between solves as arguments, so every block solve
reuses the same compiled code.

| File | Contents |
| --- | --- |
| `aerostruct.py` | The ALBCD solve. Run this. |
| `monolithic.py` | The same problem solved with a single SLSQP; writes the reference solution `monolithic_solution.npz` |
| `models.py` | Wing setup and the aerodynamic and structural models, shared by both scripts |

The aerodynamics use the `jvlm` package (from `jvlm-oas`), a vortex-lattice method in JAX
ported from OpenAeroStruct, and the spar uses `jframe`, a differentiable 3D frame solver in JAX.
Both are separate packages; install them into the same environment first:

```bash
pip install -e path/to/jvlm-oas path/to/jframe
```

```bash
python examples/jax/aerostruct/aerostruct.py
```
