# Cessna wing design

An aerostructural ALBCD example on a Cessna 182-like wing (11 m span, constant
1.5 m chord over the inboard half of each semispan, tapering to 1.0 m at the tip),
in JAX; the same problem in PyTorch is in [../../pytorch/cessna/](../../pytorch/cessna/README.md).
It minimizes the drag of the wing over its twist, subject to lift = weight, while
sizing the wall thickness of its tubular spar so that the von Mises stress under a
5 g maneuver (with a 1.5 safety factor) stays below yield and the wall stays above
a 1 mm minimum gauge. Both structural constraints are aggregated with KS functions.
The aerodynamics and the structure are the two ALBCD blocks, coupled through
copies of the aero loads and the weight.

Each subproblem compiles its objective, local constraints and their derivatives once with
`jax.jit`, passing the values that change between solves as arguments, so every block solve
reuses the same compiled code.

| File | Contents |
| --- | --- |
| `cessna.py` | The ALBCD solve. Run this. |
| `monolithic.py` | The same problem solved with a single SLSQP; writes the reference solution `monolithic_solution.npz` |
| `models.py` | Wing setup and the aerodynamic and structural models, shared by both scripts |
| `vlm_jax/` | Vortex-lattice aerodynamics, a JAX port of the aerodynamics in OpenAeroStruct |
| `cessna182_no_wing.stl` | Cessna 182 fuselage (without wing), for the PyVista view of the optimized wing |
| `beam_jax.py` | Beam finite-element model with a thin-walled tube cross-section |

```bash
python examples/jax/cessna/cessna.py
```
