# Distribution Statement A. Approved for public release: distribution is unlimited. Approved AFRL-2026-1671 28-09-2026.

"""The alternating direction method of multipliers (ADMM) in its canonical form, for comparison with ALBCD.

S. Boyd, N. Parikh, E. Chu, B. Peleato and J. Eckstein, "Distributed optimization and statistical
learning via the alternating direction method of multipliers," Found. Trends Mach. Learn. 3 (2011) 1-122.

Each iteration is one Gauss-Seidel sweep over the blocks, each block minimizing the augmented Lagrangian

    L(x; y, mu) = f(x) + y^T phi(x) + 1/2 sum_i mu_i phi_i(x)^2,

followed by the dual step y <- y + mu * phi. The penalty mu is fixed. The iteration stops when the
primal residual max|phi| and the dual residual are both small. For two blocks the dual residual
mu A^T B (z^{k+1} - z^k) is the stationarity violation of the earlier blocks after the later ones
moved, which is what the blocks' KKT residuals measure, so the max-norm KKT residual over all blocks is
used. This is ALBCD with a single sweep per multiplier update, no penalty updates and a stopping test
that also requires stationarity. ADMM has no convergence guarantee for nonconvex problems.
"""

import time
from typing import Optional, Sequence, Union
import numpy as np


class ADMM():
    """Canonical ADMM with a fixed penalty mu, on the same subproblems as :class:`~albcd.ALBCD`.

    Parameters
    ----------
    subproblems : list of Subproblem
        One subproblem per block, solved in list order during each sweep.
    x0 : array_like
        Initial design vector.
    mu : float or array_like
        The fixed penalty parameter. An array gives one value per coupling constraint; a float needs
        ``n_con``.
    n_con : int, optional
        Number of coupling constraints, when ``mu`` is a float.
    data0 : dict, optional
        Initial values of any quantities the subproblems share.
    feas_tol : float
        Tolerance on the primal residual ``max|phi|``.
    opt_tol : float
        Tolerance on the dual residual (the max-norm KKT residual over all blocks).
    max_iter : int
        Maximum number of iterations (sweeps).
    verbose : bool
        Print per-iteration diagnostics.

    Attributes
    ----------
    x, y, mu, phi : ndarray
        Design vector, multipliers, (fixed) penalty parameters and coupling constraints.
    success : bool
        True if both residuals reached their tolerances.
    x_history : list of ndarray
        ``x0`` followed by ``x`` after every subproblem solve.
    feas_history, opt_history : list of float
        Primal and dual residuals after every sweep.
    tf : float
        Wall-clock time of :meth:`solve` in seconds.
    """

    def __init__(self,
                 subproblems: Sequence,
                 x0: np.ndarray,
                 mu: Union[float, np.ndarray],
                 n_con: Optional[int] = None,
                 data0: Optional[dict] = None,
                 feas_tol: float = 1e-3,
                 opt_tol: float = 1e-3,
                 max_iter: int = 500,
                 verbose: bool = True,
                 ):

        self.subproblems = subproblems
        self.x = np.array(x0, dtype=float)
        self.data = {} if data0 is None else dict(data0)
        if np.ndim(mu) == 0:
            if n_con is None:
                raise ValueError('a scalar mu needs n_con, the number of coupling constraints')
            mu = np.full(n_con, float(mu))
        self.mu = np.array(mu, dtype=float).copy()
        if self.mu.ndim != 1 or self.mu.size == 0:
            raise ValueError('mu needs one entry per coupling constraint')
        self.y = np.zeros_like(self.mu)
        self.phi = None
        self.feas_tol = feas_tol
        self.opt_tol = opt_tol
        self.max_iter = max_iter
        self.verbose = verbose

        self.x_history = [self.x.copy()]
        self.feas_history = []
        self.opt_history = []
        self.success = False
        self.tf = None

    def solve(self) -> "ADMM":
        t0 = time.perf_counter()
        log = print if self.verbose else lambda *args: None
        self.success = False

        for k in range(1, self.max_iter + 1):
            # one Gauss-Seidel sweep over the blocks
            for sub in self.subproblems:
                outputs = {}
                sub.solve(self.x, self.y, self.mu, self.data, outputs)
                self.x = outputs.pop("x")
                self.phi = np.array(outputs.pop("phi"), dtype=float).reshape(self.mu.shape)
                self.data.update(outputs)
                self.x_history.append(self.x.copy())

            primal = float(np.max(np.abs(self.phi)))
            dual = float(max(sub.residual(self.x, self.y, self.mu, self.data) for sub in self.subproblems))
            self.feas_history.append(primal)
            self.opt_history.append(dual)
            log(f'iter {k:4d} | primal {primal:.3e} | dual {dual:.3e}')

            if primal <= self.feas_tol and dual <= self.opt_tol:
                self.success = True
                break

            self.y = self.y + self.mu * self.phi  # dual ascent with the fixed penalty

        self.tf = time.perf_counter() - t0
        log(f"{'Converged' if self.success else 'Did not converge'} after {k} iterations ({self.tf:.2f} s)")
        return self
