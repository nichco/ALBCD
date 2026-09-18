"""Augmented Lagrangian block coordinate descent (ALBCD)."""

from .albcd import AugmentedLagrangianBlockCoordinateDescent
from .subproblem import Subproblem

ALBCD = AugmentedLagrangianBlockCoordinateDescent  # short alias

__version__ = "0.1.0"

__all__ = ["ALBCD", "AugmentedLagrangianBlockCoordinateDescent", "Subproblem"]
