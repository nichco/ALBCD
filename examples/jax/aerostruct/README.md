# Aerostructural wing design

A larger ALBCD example, in JAX; the same problem as
[../../pytorch/aerostruct/](../../pytorch/aerostruct/README.md). It minimizes the drag of a
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
| `vlm_jax/` | Vortex-lattice aerodynamics, a JAX port of the aerodynamics in OpenAeroStruct |
| `beam_jax.py` | Beam finite-element model with a thin-walled tube cross-section |

```bash
python examples/jax/aerostruct/aerostruct.py
```
