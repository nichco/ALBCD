# Multipoint uCRM wing design

A larger ALBCD example, in PyTorch only. It minimizes the average fuel burn of N
missions, with different payloads and ranges, flown by the same uCRM wing. The
design variables are the wing's twist and each mission's angle of attack, and
every mission is trimmed to lift = weight. Each mission is an ALBCD block with its
own copy of the twist, and the coupling constraints force the copies to agree.
N (2 by default) is set in `models.py`.

| File | Contents |
| --- | --- |
| `multipoint.py` | The ALBCD solve. Run this. |
| `monolithic.py` | The same problem solved with a single SLSQP; writes the reference solution `monolithic_solution_N{N}.npz` |
| `models.py` | The missions, the wing and the flight condition, shared by both scripts |
| `vlm_torch/` | Vortex-lattice aerodynamics, a PyTorch port of the aerodynamics in OpenAeroStruct |

```bash
python examples/pytorch/uCRM/multipoint.py
```
