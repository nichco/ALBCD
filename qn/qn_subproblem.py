"""Block interface for QuasiNewtonALBCD: albcd.Subproblem plus first-order information.

QuasiNewtonALBCD needs, besides the block solves, each block's objective gradient, coupling term and local
constraints with their derivatives at the current x (grad_phi), and the multipliers of its local constraints
from the latest solve (local_multipliers). Optionally, a block records the points its optimizer evaluates
(self.curvature), which the solver turns into free BFGS pairs.
"""

import numpy as np
from albcd import Subproblem


class QNSubproblem(Subproblem):
    """An albcd.Subproblem that also provides first-order information for QuasiNewtonALBCD.

    Optional harvesting contract: if ``self.harvest`` is true (QuasiNewtonALBCD sets it when it uses harvested
    pairs), ``solve()`` may set ``self.curvature = [(v, grad_obj, Jh), ...]``, one entry per point its optimizer
    evaluated, in evaluation order: the unscaled block variables v, the gradient of the subproblem objective (the
    block's augmented Lagrangian at the solve's y and mu, without any proximal term) and the local constraint
    Jacobian at v. The solver consumes and clears the list after each solve.
    """

    x_scale = None   # scale vector D for this block's variables (None -> ones)
    harvest = False  # set by QuasiNewtonALBCD: record self.curvature during solve()
    act_tol = 1e-8   # set by QuasiNewtonALBCD: activity threshold for local-constraint multipliers

    def grad_phi(self, x, data) -> dict:
        """First-order information of this block at the global x, for QuasiNewtonALBCD. Optional.

        Returns a dict with, for this block's variables v = x[self.index] (unscaled):
            grad_f : (n,)        gradient of this block's objective term f_i(v) (no multiplier or penalty terms,
                                 no proximal term)
            c      : (m_c,)      this block's coupling term c_i(v); sum over blocks = phi(x)
            J      : (m_c, n)    dc_i/dv
            h      : (n_h,)      local constraint values (n_h may be 0)
            Jh     : (n_h, n)    dh/dv
            cl, cu : (n_h,)      local constraint bounds (cl == cu for equalities)
            xl, xu : (n,)        variable bounds (+-inf where unbounded)
            outputs: dict        the shared data this block writes in solve(), recomputed at x without
                                 solving (may be empty)
        """
        raise NotImplementedError

    def local_multipliers(self) -> np.ndarray:
        """Multipliers kappa of the local constraints from the latest solve(), length n_h, in the convention
        grad_f + J^T lam + Jh^T kappa = 0 on the active rows, and 0 on inactive rows. Optional."""
        raise NotImplementedError

    def predict_outputs(self, lin, dv) -> dict:
        """Linear prediction of this block's shared outputs after its variables move by dv from the point of the
        linearization ``lin`` (a grad_phi result), without evaluating the block. Optional; used by
        QuasiNewtonALBCD's "aladin" mode, which never evaluates a block at the step it takes, so that a block
        reading another block's outputs before that block re-solves sees predicted rather than stale data.
        The default predicts no change (returns {}: data keeps the values from the linearization point)."""
        return {}

    def scale(self, n):
        """The scale vector D of this block's n variables."""
        return np.ones(n) if self.x_scale is None else np.asarray(self.x_scale, dtype=float)
