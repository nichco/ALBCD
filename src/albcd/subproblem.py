"""Base class for ALBCD subproblems."""

import numpy as np


class Subproblem:
    """Base class for one block of an ALBCD problem.

    Subclasses implement :meth:`setup`, :meth:`solve` and :meth:`residual`.
    The solver passes ``x``, ``y`` and ``mu`` as inputs. Any other declared
    input is read from the solver's ``data`` dictionary, and any extra output
    is written back to it, so blocks can exchange additional quantities.

    Parameters
    ----------
    index : slice or array_like of int
        Entries of the global design vector ``x`` owned by this block.
    """

    def __init__(self, index):
        self.index = index # slice/array selecting this block's entries out of x
        self.inputs = {}
        self.outputs = {}
        self.setup()

    def setup(self) -> None:
        """Declare inputs and outputs with :meth:`add_input` and :meth:`add_output`.

        Outputs ``"x"`` and ``"phi"`` are required.
        """
        raise NotImplementedError

    def solve(self, inputs, outputs) -> None:
        """Minimize the augmented Lagrangian over this block with the other blocks fixed.

        Must set ``outputs["x"]``, the global design vector with this block's
        entries updated (see :meth:`recompose`), and ``outputs["phi"]``, the
        coupling constraints evaluated at that same ``x``.
        """
        raise NotImplementedError

    def residual(self, inputs) -> float:
        """Return the max-norm KKT stationarity residual of this block's subproblem at ``inputs``."""
        raise NotImplementedError

    def add_input(self, name) -> None:
        """Declare an input named ``name``."""
        self.inputs[name] = None

    def add_output(self, name) -> None:
        """Declare an output named ``name``."""
        self.outputs[name] = None

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
