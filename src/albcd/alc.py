# Distribution Statement A. Approved for public release: distribution is unlimited. Approved AFRL-2026-1671 28-09-2026.

"""Augmented Lagrangian coordination (ALC) of Tosserams, Etman and Rooda (2008), for comparison
with ALBCD.

S. Tosserams, L. F. P. Etman and J. E. Rooda, "Augmented Lagrangian coordination for
distributed optimal design in MDO," Int. J. Numer. Meth. Engng 73 (2008) 1885-1910.

ALC relaxes the coupling constraints with the penalty v^T q + ||w o q||^2. With y = v and
mu_i = 2 w_i^2 this is exactly ALBCD's y^T phi + 1/2 sum mu_i phi_i^2, and ALC's multiplier
update v <- v + 2 w o w o q is ALBCD's y <- y + mu o phi. The subproblems are therefore the
same :class:`~albcd.Subproblem` objects, and only the coordination differs:

- the inner loop stops when the relative change of the merit function between two sweeps,
  |F^k - F^{k-1}| / (1 + |F^k|), is below ``inner_tol`` (Eq. 28), not on a KKT residual;
- each weight grows as w_i <- beta w_i, i.e. mu_i <- beta^2 mu_i, whenever
  |q_i| > gamma |q_i^prev| (Eq. 25), with no feasibility threshold and no cap;
- the outer loop stops when max|q| < eps and max|q - q_prev| < eps (Eqs. 26-27).
"""

import time
from typing import Callable, Optional, Sequence
import numpy as np


class ALC():
    """Augmented Lagrangian coordination (Tosserams et al., 2008).

    Uses the same subproblems as :class:`~albcd.ALBCD`. Penalties are passed and stored in
    ALBCD's convention, ``mu = 2 w^2``; ``beta`` and ``gamma`` are ALC's weight-growth constants,
    so ``mu`` grows by ``beta**2``.

    Parameters
    ----------
    subproblems : list of Subproblem
        One subproblem per block, solved in list order during each sweep.
    x0 : array_like
        Initial design vector.
    mu0 : array_like
        Initial penalty parameters, one per coupling constraint (``mu = 2 w^2``). See
        :func:`initial_penalty` for ALC's own rule (Eq. 29).
    objective : callable(x, data) -> float
        The objective f without the penalty terms. ALC's inner stopping test needs the merit
        function ``F = f + y^T phi + 1/2 sum mu phi^2``, which the subproblems do not return.
    phi0 : array_like, optional
        The coupling constraints at ``x0``. Without it, the first inner loop cannot stop after
        its first sweep, since there is no previous merit value to compare with.
    data0 : dict, optional
        Initial values of any quantities the subproblems share.
    beta : float
        Weight growth factor (Eq. 25); 2.2 for an exact inner loop, 2.0 inexact.
    gamma : float
        Required reduction of each constraint per outer iteration (Eq. 25); 0.4 exact, 0.5 inexact.
    feas_tol : float
        The tolerance epsilon of the outer stopping test (Eqs. 26-27).
    inner_tol : float, optional
        Relative merit-change tolerance of the inner loop (Eq. 28); ``feas_tol / 100`` by default.
    max_outer_iter, max_inner_iter : int
        Iteration limits (the paper gives none).
    record_residual : bool
        Evaluate every subproblem's KKT residual after each sweep for ``opt_history``. It is
        diagnostic only: ALC never uses it to stop.
    verbose : bool
        Print per-iteration diagnostics.

    Attributes
    ----------
    x, y, mu, phi : ndarray
        Design vector, multipliers, penalty parameters and coupling constraints.
    success : bool
        True if the outer stopping test (Eqs. 26-27) was met. Note that ALC never checks
        stationarity, so this is ALC's own claim of convergence.
    x_history : list of ndarray
        ``x0`` followed by ``x`` after every subproblem solve.
    feas_history, opt_history, merit_history : list of float
        ``max|phi|``, the max-norm KKT residual over all blocks (NaN unless
        ``record_residual``) and the merit function F, after every sweep.
    outer_history : list of int
        The outer iteration of every sweep.
    tf : float
        Wall-clock time of :meth:`solve` in seconds.
    """

    def __init__(self,
                 subproblems: Sequence,
                 x0: np.ndarray,
                 mu0: np.ndarray,
                 objective: Callable,
                 phi0: Optional[np.ndarray] = None,
                 data0: Optional[dict] = None,
                 beta: float = 2.2,
                 gamma: float = 0.4,
                 feas_tol: float = 1e-3,
                 inner_tol: Optional[float] = None,
                 max_outer_iter: int = 100,
                 max_inner_iter: int = 200,
                 record_residual: bool = True,
                 verbose: bool = True,
                 ):

        self.subproblems = subproblems
        self.x = np.array(x0, dtype=float)
        self.data = {} if data0 is None else dict(data0)
        self.mu = np.array(mu0, dtype=float).copy()  # copy, so the caller's mu0 is never grown in place
        if self.mu.ndim != 1 or self.mu.size == 0:
            raise ValueError('mu0 needs one entry per coupling constraint')
        self.y = np.zeros_like(self.mu)
        self.phi = None if phi0 is None else np.array(phi0, dtype=float).reshape(self.mu.shape)
        self.objective = objective
        self.beta = beta
        self.gamma = gamma
        self.feas_tol = feas_tol
        self.inner_tol = feas_tol / 100 if inner_tol is None else inner_tol
        self.max_outer_iter = max_outer_iter
        self.max_inner_iter = max_inner_iter
        self.record_residual = record_residual
        self.verbose = verbose

        self.x_history = [self.x.copy()]
        self.feas_history = []
        self.opt_history = []
        self.merit_history = []
        self.outer_history = []
        self.success = False
        self.tf = None

    def merit(self) -> float:
        """The relaxed objective F = f + y^T phi + 1/2 sum mu phi^2 at the current x."""
        return float(self.objective(self.x, self.data) + self.y @ self.phi + 0.5 * self.mu @ self.phi**2)

    def _update_mu(self, q_new, q_old) -> None:
        """Eq. 25: grow the weight of every constraint that did not shrink by gamma, satisfied or not."""
        grow = np.abs(q_new) > self.gamma * np.abs(q_old)
        self.mu[grow] *= self.beta**2

    def solve(self) -> "ALC":
        t0 = time.perf_counter()
        log = print if self.verbose else lambda *args: None
        self.success = False
        q_old = np.full_like(self.mu, np.inf)  # no weight grows on the first outer iteration

        for k in range(1, self.max_outer_iter + 1):
            F_old = None if self.phi is None else self.merit()

            # block coordinate descent inner loop, stopped on the relative change of the merit function
            for j in range(1, self.max_inner_iter + 1):
                for sub in self.subproblems:
                    outputs = {}
                    sub.solve(self.x, self.y, self.mu, self.data, outputs)
                    self.x = outputs.pop("x")
                    self.phi = np.array(outputs.pop("phi"), dtype=float).reshape(self.mu.shape)
                    self.data.update(outputs)
                    self.x_history.append(self.x.copy())

                F = self.merit()
                res = (max(sub.residual(self.x, self.y, self.mu, self.data) for sub in self.subproblems)
                       if self.record_residual else np.nan)
                self.feas_history.append(float(np.max(np.abs(self.phi))))
                self.opt_history.append(float(res))
                self.merit_history.append(F)
                self.outer_history.append(k)

                change = np.inf if F_old is None else abs(F - F_old) / (1 + abs(F))
                log(f'outer {k:3d} | sweep {j:3d} | merit change {change:.3e} | opt_res {res:.3e}')
                if change < self.inner_tol:
                    break
                F_old = F

            feas = float(np.max(np.abs(self.phi)))
            change_q = float(np.max(np.abs(self.phi - q_old)))
            log(f'outer {k:3d} | feas {feas:.3e} | change {change_q:.3e} | max mu {np.max(self.mu):.3e}')

            # Eqs. 26-27: feasible, and the constraints no longer change
            if feas < self.feas_tol and change_q < self.feas_tol:
                self.success = True
                break

            # method of multipliers (Eq. 24) and weight update (Eq. 25)
            self.y = self.y + self.mu * self.phi
            self._update_mu(self.phi, q_old)
            q_old = self.phi.copy()

        self.tf = time.perf_counter() - t0
        log(f"{'Converged' if self.success else 'Did not converge'} after {k} outer iterations ({self.tf:.2f} s)")
        return self


def initial_penalty(subproblems: Sequence,
                    x0: np.ndarray,
                    n_con: int,
                    objective: Callable,
                    phi0: Optional[np.ndarray] = None,
                    data0: Optional[dict] = None,
                    f_hat: Optional[float] = None,
                    alpha: float = 0.1,
                    mu_small: float = 1e-3,
                    inner_tol: float = 1e-6,
                    max_inner_iter: int = 200) -> float:
    """ALC's initial weight (Tosserams et al., 2008, Eq. 29), returned as a penalty mu0 = 2 w^2.

    One ALC inner loop is solved from ``x0`` with the small penalty ``mu_small`` and zero
    multipliers, which gives estimates q_hat of the coupling constraints. The weight then makes
    the penalty term a fraction ``alpha`` of a typical objective value ``f_hat`` (by default the
    objective at ``x0``): w = sqrt(alpha |f_hat| / q_hat^T q_hat). The subproblems are solved in
    place, so pass ones that can be reused.

    When q_hat is close to zero (for example an inequality that the small-weight solve already
    satisfies), w becomes very large; ALC gives no safeguard for that case.
    """
    est = ALC(subproblems, x0, np.full(n_con, mu_small), objective, phi0=phi0, data0=data0,
              beta=1.0, inner_tol=inner_tol, max_outer_iter=1, max_inner_iter=max_inner_iter,
              record_residual=False, verbose=False)
    est.solve()
    q_hat = est.phi
    if f_hat is None:
        f_hat = objective(np.array(x0, dtype=float), {} if data0 is None else dict(data0))
    w = np.sqrt(alpha * abs(f_hat) / max(float(q_hat @ q_hat), 1e-300))
    return 2 * w**2
