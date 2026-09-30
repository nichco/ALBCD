# Distribution Statement A. Approved for public release: distribution is unlimited. Approved AFRL-2026-1671 28-09-2026.

import time
from typing import Optional, Sequence, Union
import numpy as np


class ALBCD():
    """Augmented Lagrangian block coordinate descent (ALBCD).

    Solves ``min f(x)  s.t.  phi(x) = 0``, where ``x`` is partitioned into
    blocks, each owned by a :class:`~albcd.Subproblem`, and ``phi`` are the
    coupling constraints between blocks. Constraints local to one block are
    handled inside that block's subproblem; coupling inequalities can be
    written as equalities with slack variables.

    Each outer iteration approximately minimizes the augmented Lagrangian

        L(x; y, mu) = f(x) + y^T phi(x) + 1/2 sum_i mu_i phi_i(x)^2

    by sweeping over the subproblems (block coordinate descent) until every
    subproblem residual is at most ``opt_tol``. It then updates the multipliers,
    ``y <- y + mu * phi``, and multiplies ``mu_i`` by ``rho`` wherever
    ``|phi_i|`` did not drop below ``tau`` times its previous value. The solve
    stops once ``max|phi| <= feas_tol`` in the final phase.

    With ``unconstrained=True`` there are no coupling constraints: ``y``,
    ``mu`` and ``phi`` are empty, the augmented Lagrangian is just ``f``, and
    the solve reduces to block coordinate descent. Each ``opt_tol`` phase is
    one outer iteration of at most ``max_inner_iter`` sweeps, until every
    subproblem residual is at most that phase's tolerance. Constraints local
    to one block are still allowed.

    Parameters
    ----------
    subproblems : list of Subproblem
        One subproblem per block, solved in list order during each sweep.
    x0 : array_like
        Initial design vector.
    mu0 : array_like, optional
        Initial penalty parameters, one per coupling constraint; sets the
        length of ``y`` and ``phi``. Required unless ``unconstrained=True``,
        in which case it must be omitted.
    data0 : dict, optional
        Initial values of any quantities the subproblems share, keyed by name.
    max_mu : float
        Upper bound on each penalty parameter.
    rho : float
        Penalty growth factor.
    tau : float
        Required reduction factor of each constraint violation per outer iteration.
    feas_tol : float
        Feasibility tolerance on ``max|phi|``.
    opt_tol : float or sequence of float
        Inner loop optimality tolerance for each phase; its length sets the
        number of phases (e.g. a loose tolerance first, then a tight one).
    max_outer_iter : int
        Maximum number of outer iterations, counted across all phases.
    max_inner_iter : int
        Maximum number of sweeps per outer iteration.
    max_y : float
        Bound on the magnitude of each Lagrange multiplier.
    verbose : bool
        Print per-iteration diagnostics.
    unconstrained : bool
        The problem has no coupling constraints, so ``mu0`` is omitted and no
        subproblem sets the ``"phi"`` output. Only block coordinate
        descent runs, once per ``opt_tol`` phase (``max_outer_iter`` still
        bounds the number of phases); ``max_mu``, ``rho``, ``tau``,
        ``feas_tol`` and ``max_y`` are unused.

    Attributes
    ----------
    x, y, mu, phi : ndarray
        Design vector, multipliers, penalty parameters and coupling
        constraints; the solution once :meth:`solve` returns. ``y``, ``mu``
        and ``phi`` are empty when unconstrained.
    success : bool
        True if the solve ended feasible (``max|phi| <= feas_tol``) and
        optimal (the last inner loop reached the final ``opt_tol``); when
        unconstrained, only optimal.
    data : dict
        Quantities shared between subproblems, passed to each solve() and residual().
    x_history : list of ndarray
        ``x0`` followed by ``x`` after every subproblem solve.
    feas_history : list of float
        ``max|phi|`` after every sweep (0 when unconstrained).
    opt_history : list of float
        Max-norm KKT stationarity residual over all blocks after every sweep,
        the value the inner loop compares with ``opt_tol``. Aligned with
        ``feas_history``: entry ``i`` of both describes the ``i``-th sweep.
    tf : float
        Wall-clock time of :meth:`solve` in seconds.
    """

    def __init__(self,
                 subproblems: Sequence,
                 x0: np.ndarray,
                 mu0: Optional[np.ndarray] = None,
                 data0: Optional[dict] = None,
                 max_mu: float = 1e3,
                 rho: float = 1.2,
                 tau: float = 0.5,
                 feas_tol: float = 1e-3,
                 opt_tol: Union[float, Sequence[float]] = (1e-2, 1e-5),
                 max_outer_iter: int = 100,
                 max_inner_iter: int = 10,
                 max_y: float = 1e6,
                 verbose: bool = True,
                 unconstrained: bool = False,
                 ):

        self.subproblems = subproblems
        self.x = np.array(x0, dtype=float) # copy, so the caller's x0 is never modified
        self.data = {} if data0 is None else dict(data0) # other quantities shared between subproblems; never holds x, y, mu or phi
        self.tf = None
        self.unconstrained = unconstrained

        # mu0 has one entry per coupling constraint, so it's needed exactly when there are some
        if unconstrained and mu0 is not None:
            raise ValueError('mu0 must be omitted when unconstrained=True')
        if not unconstrained and (mu0 is None or np.size(mu0) == 0):
            raise ValueError('mu0 needs one entry per coupling constraint; set unconstrained=True if there are none')

        # force float dtype; an integer-dtype mu would otherwise truncate penalty growth.
        # Unconstrained, mu (and with it y and phi) is empty
        self.mu = np.zeros(0) if unconstrained else np.asarray(mu0, dtype=float) # penalty parameter(s)

        self.max_mu = max_mu
        self.rho = rho
        self.tau = tau
        self.feas_tol = feas_tol

        # one optimality tolerance per phase; a scalar gives a single phase
        self.opt_tol = np.atleast_1d(np.asarray(opt_tol, dtype=float))
        if self.opt_tol.ndim != 1 or self.opt_tol.size == 0:
            raise ValueError('opt_tol must be a float or a 1D list/array of floats')

        self.max_outer_iter = max_outer_iter
        self.max_inner_iter = max_inner_iter
        self.max_y = max_y
        self.verbose = verbose
        self.x_history = [self.x.copy()]
        self.feas_history = [] # feasibility (max constraint violation) after each sweep
        self.opt_history = []  # optimality (max-norm KKT residual) after each sweep
        self.y = np.zeros_like(self.mu) # Lagrange multipliers, one per coupling constraint
        self.phi = np.zeros(0) if unconstrained else None # coupling constraints at the current x, from the latest subproblem solve
        self.success = False # set by solve()


    def _update_mu(self, c_new, c_old) -> None:
        """Increase the penalty parameter of each scalar constraint if
         it is infeasible and its feasibility is not decreasing fast enough."""
        # vectorized over all constraints instead of a per-index Python loop
        f_new = np.abs(c_new)
        f_old = np.abs(c_old)
        grow = (f_new > self.tau * f_old) & (f_new > self.feas_tol)
        self.mu[grow] = np.minimum(self.rho * self.mu[grow], self.max_mu)

        return None


    def solve(self) -> "ALBCD":

        t0 = time.perf_counter()
        log = print if self.verbose else lambda *args: None
        self.success = False
        phi_old = np.full_like(self.mu, np.inf)  # no penalty parameter grows on the first outer iteration
        k = 0

        for phase, opt_tol in enumerate(self.opt_tol, start=1):  # tolerance phases

            # augmented Lagrangian outer loop
            while k < self.max_outer_iter:
                k += 1

                # BCD inner loop
                for j in range(1, self.max_inner_iter + 1):

                    for sub in self.subproblems:

                        outputs = {}
                        sub.solve(self.x, self.y, self.mu, self.data, outputs)
                        self.x = outputs.pop("x")
                        # phi at the new x; none when unconstrained. A missing or wrong-sized phi fails the reshape
                        self.phi = np.array(outputs.pop("phi", []), dtype=float).reshape(self.mu.shape)
                        self.data.update(outputs)  # any other output is shared through data

                        self.x_history.append(self.x.copy())

                    self.feas_history.append(float(np.max(np.abs(self.phi), initial=0.0)))

                    res = max(sub.residual(self.x, self.y, self.mu, self.data) for sub in self.subproblems)
                    self.opt_history.append(res)

                    log(f'outer {k:3d} | sweep {j:3d} | opt_res {res:.3e}')

                    if res <= opt_tol: break


                feas = np.max(np.abs(self.phi), initial=0.0)  # 0 when unconstrained

                log(f'outer {k:3d} | feas {feas:.3e} | max mu {np.max(self.mu, initial=0.0):.3e} | |y| {np.linalg.norm(self.y):.3e}')

                # if feasible in the final phase: stop
                # success also requires the last inner loop to have reached opt_tol
                if feas <= self.feas_tol and phase == len(self.opt_tol):
                    self.success = res <= opt_tol
                    break

                # update the multipliers and clip to the bounds
                self.y = np.clip(self.y + self.mu * self.phi, -self.max_y, self.max_y)
                self._update_mu(self.phi, phi_old)
                phi_old = self.phi

                if feas <= self.feas_tol:
                    log(f'Phase {phase} complete, starting phase {phase + 1}')
                    break  # next phase

        self.tf = time.perf_counter() - t0
        iters = f'{j} sweeps' if self.unconstrained else f'{k} outer iterations'
        log(f"{'Converged' if self.success else 'Did not converge'} after {iters} ({self.tf:.2f} s)")
        return self
