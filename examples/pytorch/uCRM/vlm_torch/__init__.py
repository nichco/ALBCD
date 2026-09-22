"""Self-contained aero-only VLM package: mesh generation
(``vlm_torch.meshing``), vortex-lattice aerodynamics
(``vlm_torch.aerodynamics``), twist geometry (``vlm_torch.geometry``), and
mission/performance functionals (``vlm_torch.functionals``), bundled into a
single per-case evaluation by ``vlm_torch.analysis`` -- see
``vlm_torch.analysis.solve_aero`` for the generic, mission-independent entry
point (mesh + flight condition in, forces/CL/CD out) and
``build_problem``/``evaluate`` for the whole-aircraft-performance one (fuel
burn, L=W trim) layered on top of it.

Copied from the aerodynamics-relevant subset of ``openaerostruct_jax``
(dropping everything structural: no beam FEM, no aerostructural coupling),
so this package has no dependency on it or on OpenMDAO/OpenAeroStruct at
runtime.

PyTorch port of the ``vlm`` package: every ``jax.numpy`` operation is
replaced by its ``torch`` equivalent and differentiation is done with
PyTorch autograd instead of ``jax.grad``. The numerics are otherwise
identical -- see ``vlm_torch.aerodynamics`` for the three places where
torch's API forced a different spelling (no negative-stride slicing, no
functional ``.at[]`` updates, and ``torch.linalg.cross`` for a ``dim=-1``
default matching ``jnp.cross``).
"""
import torch

# Essential: OpenAeroStruct/OpenMDAO compute in float64, and this package's
# results are validated against that. Torch defaults to float32, which would
# silently produce a different (much less accurate) answer.
torch.set_default_dtype(torch.float64)
