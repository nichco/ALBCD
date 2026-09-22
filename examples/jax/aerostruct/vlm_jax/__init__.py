"""Self-contained aero-only VLM package: mesh generation
(``vlm_jax.meshing``), vortex-lattice aerodynamics
(``vlm_jax.aerodynamics``), twist geometry (``vlm_jax.geometry``), and
mission/performance functionals (``vlm_jax.functionals``), bundled into a
single per-case evaluation by ``vlm_jax.analysis`` -- see
``vlm_jax.analysis.solve_aero`` for the generic, mission-independent entry
point (mesh + flight condition in, forces/CL/CD out) and
``build_problem``/``evaluate`` for the whole-aircraft-performance one (fuel
burn, L=W trim) layered on top of it.

Copied from the aerodynamics-relevant subset of ``openaerostruct_jax``
(dropping everything structural: no beam FEM, no aerostructural coupling),
so this package has no dependency on it or on OpenMDAO/OpenAeroStruct at
runtime. The JAX counterpart of ``examples/pytorch/aerostruct/vlm_torch``;
the two compute the same quantities in the same order.
"""
import jax

# Essential: OpenAeroStruct/OpenMDAO compute in float64, and this package's
# results are validated against that. JAX defaults to float32, which would
# silently produce a different (much less accurate) answer.
jax.config.update("jax_enable_x64", True)
