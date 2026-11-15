"""Mesh-generation utilities, copied unchanged from OpenAeroStruct.

These are pure numpy (no OpenMDAO dependency) and run exactly once, before
optimization starts, to build the fixed baseline mesh. They are not part of
the JAX-differentiated computational graph, so there is nothing to port here.
"""
from vlm_jax.meshing.mesh_generator import generate_mesh

__all__ = ["generate_mesh"]
