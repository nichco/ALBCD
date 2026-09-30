# Distribution Statement A. Approved for public release: distribution is unlimited. Approved AFRL-2026-1671 28-09-2026.

"""
.. include:: ../../README.md
   :start-after: # Augmented Lagrangian block coordinate descent
"""

from .albcd import ALBCD
from .subproblem import Subproblem

__version__ = "0.1.0"

__all__ = ["ALBCD", "Subproblem"]
