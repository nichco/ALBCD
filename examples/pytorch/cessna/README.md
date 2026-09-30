# Cessna wing design

An aerostructural ALBCD example on a Cessna 182-like wing (11 m span, constant
1.5 m chord over the inboard half of each semispan, tapering to 1.0 m at the tip),
in PyTorch; the same problem in JAX is in [../../jax/cessna/](../../jax/cessna/README.md).
It minimizes the drag of the wing over its twist, subject to lift = weight, while
sizing the wall thickness of its tubular spar so that the von Mises stress under a
5 g maneuver (with a 1.5 safety factor) stays below yield and the wall stays above
a 1 mm minimum gauge. Both structural constraints are aggregated with KS functions.
The aerodynamics and the structure are the two ALBCD blocks, coupled through
copies of the aero loads and the weight.

| File | Contents |
| --- | --- |
| `cessna.py` | The ALBCD solve. Run this. |
| `monolithic.py` | The same problem solved with a single SLSQP; writes the reference solution `monolithic_solution.npz` |
| `models.py` | Wing setup and the aerodynamic and structural models, shared by both scripts |
| `vlm_torch/` | Vortex-lattice aerodynamics, a PyTorch port of the aerodynamics in OpenAeroStruct |
| `cessna182_no_wing.stl` | Cessna 182 fuselage (without wing), for the PyVista view of the optimized wing |
| `beam_torch.py` | Beam finite-element model with a thin-walled tube cross-section |

```bash
python examples/pytorch/cessna/cessna.py
```
