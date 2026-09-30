# Distribution Statement A. Approved for public release: distribution is unlimited. Approved AFRL-2026-1671 28-09-2026.

"""Base class for ALBCD subproblems."""

import numpy as np


class Subproblem:
    """Base class for one block of an ALBCD problem.

    Subclasses implement :meth:`solve` and :meth:`residual`, and can override
    :meth:`setup` for one-time work such as compiling functions. The solver
    passes both the current design vector ``x``, multipliers ``y``, penalty
    parameters ``mu`` and its ``data`` dictionary, and writes any output of
    :meth:`solve` other than ``x`` and ``phi`` into ``data``, so blocks can
    exchange additional quantities.

    Parameters
    ----------
    index : slice or array_like of int
        Entries of the global design vector ``x`` owned by this block.
    """

    def __init__(self, index):
        """Set up the subproblem and call :meth:`setup`. ``index`` is described above."""
        self.index = index # slice/array selecting this block's entries out of x
        self.setup()

    def setup(self) -> None:
        """Optional one-time setup, called by the constructor."""

    def solve(self, x, y, mu, data, outputs) -> None:
        """Minimize the augmented Lagrangian over this block with the other blocks fixed.

        Must set ``outputs["x"]``, the global design vector with this block's
        entries updated (see :meth:`recompose`), and ``outputs["phi"]``, the
        coupling constraints evaluated at that same ``x``. For an unconstrained
        problem the augmented Lagrangian is just the objective, and there is
        no ``phi``.
        """
        raise NotImplementedError

    def residual(self, x, y, mu, data) -> float:
        """Return the max-norm KKT stationarity residual of this block's subproblem at ``x``."""
        raise NotImplementedError

    def decompose(self, x):
        """Convenience: this subproblem's own slice of the global x, via `index`."""
        return x[self.index]

    def other(self, x):
        """
        Convenience: the coupling variables this subproblem depends on but
        doesn't own -- everything *not* in `index`. Correct whenever the
        global x is fully partitioned across blocks (the usual case);
        override if a block only couples to some of the other variables.
        """
        mask = np.ones(len(x), dtype=bool)
        mask[self.index] = False
        return x[mask]

    def recompose(self, x, v):
        """
        Returns a copy of the global x with this subproblem's entries replaced by v
        """
        x_new = np.array(x, dtype=float, copy=True)
        x_new[self.index] = v
        return x_new
