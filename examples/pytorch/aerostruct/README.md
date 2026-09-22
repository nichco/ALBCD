# Aerostructural wing design

A larger ALBCD example, in PyTorch; the same problem in JAX is in
[../../jax/aerostruct/](../../jax/aerostruct/README.md). It minimizes the drag of a tapered wing
over its twist, subject to lift = weight, while sizing the wall thickness of its
tubular spar so that both wing tips deflect by 0.1 m. The aerodynamics and the
structure are the two ALBCD blocks, coupled through copies of the aero loads and
the weight.

| File | Contents |
| --- | --- |
| `aerostruct.py` | The ALBCD solve. Run this. |
| `monolithic.py` | The same problem solved with a single SLSQP; writes the reference solution `monolithic_solution.npz` |
| `models.py` | Wing setup and the aerodynamic and structural models, shared by both scripts |
| `vlm_torch/` | Vortex-lattice aerodynamics, a PyTorch port of the aerodynamics in OpenAeroStruct |
| `beam_torch.py` | Beam finite-element model with a thin-walled tube cross-section |

```bash
python examples/pytorch/aerostruct/aerostruct.py
```
