# Distribution Statement A. Approved for public release: distribution is unlimited. Approved AFRL-2026-1671 28-09-2026.

"""The ALBCD solver."""

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
    the solve reduces to the BCD inner loop, i.e. at most ``max_inner_iter``
    sweeps until every subproblem residual is at most the final ``opt_tol``.
    Constraints local to one block are still allowed.

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
        Initial values of any extra subproblem inputs/outputs, keyed by name.
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
        With ``unconstrained=True`` only the final tolerance is used.
    max_outer_iter : int
        Maximum number of outer iterations, counted across all phases.
    max_inner_iter : int
        Maximum number of sweeps per outer iteration.
    max_y : float
        Bound on the magnitude of each Lagrange multiplier.
    save : bool
        Record ``history``, ``feas_history`` and ``opt_history``. Recording the
        optimality costs one extra ``residual()`` call per block per subproblem
        solve; with ``save=False`` it is only computed once per sweep, which is
        all the convergence test needs.
    verbose : bool
        Print per-iteration diagnostics.
    unconstrained : bool
        The problem has no coupling constraints, so ``mu0`` is omitted and no
        subproblem declares the ``"phi"`` output. Only the BCD inner loop runs;
        ``max_outer_iter``, ``max_mu``, ``rho``, ``tau``, ``feas_tol`` and
        ``max_y`` are unused.

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
        Extra inputs/outputs shared between subproblems.
    history : list of ndarray
        ``x0`` followed by ``x`` after every subproblem solve.
    feas_history : list of float
        ``max|phi|`` after every subproblem solve (0 when unconstrained).
    opt_history : list of float
        Max-norm KKT stationarity residual over all blocks after every
        subproblem solve; ``nan`` until every block has solved once, since
        ``residual()`` may need data a block caches when it solves. Aligned with
        ``feas_history``, and with ``history`` offset by the leading ``x0``:
        entry ``i`` of both describes ``history[i + 1]``.
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
                 save: bool = True,
                 verbose: bool = True,
                 unconstrained: bool = False,
                 ):
        """Set up the solver. The parameters are described above."""

        self.subproblems = subproblems
        self.x = np.array(x0, dtype=float) # copy, so the caller's x0 is never modified
        self.data = {} if data0 is None else dict(data0) # extra inputs/outputs shared between subproblems; never holds x, y, mu or phi
        self.tf = None
        self.unconstrained = unconstrained

        # mu0 has one entry per coupling constraint, so it's needed exactly when there are some
        if unconstrained and mu0 is not None:
            raise ValueError('mu0 must be omitted when unconstrained=True')
        if not unconstrained and (mu0 is None or np.size(mu0) == 0):
            raise ValueError('mu0 needs one entry per coupling constraint; set unconstrained=True if there are none')

        # every subproblem reports the coupling constraints ("phi") at the x it returns,
        # so the algorithm needs no separate constraint function. An unconstrained
        # problem has none, so its subproblems must not declare "phi"
        required = {"x"} if unconstrained else {"x", "phi"}
        for sub in self.subproblems:
            missing = required - sub.outputs.keys()
            if missing:
                raise ValueError(f'{type(sub).__name__} must declare outputs {sorted(missing)} in setup()')
            if unconstrained and "phi" in sub.outputs:
                raise ValueError(f'{type(sub).__name__} declares output "phi", but unconstrained=True')

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
            raise ValueError('opt_tol must be a float or a non-empty 1D list/array of floats')

        # phases tighten opt_tol as feasibility improves; unconstrained, there is nothing
        # to make feasible, so a single phase with the final tolerance is enough
        if unconstrained:
            self.opt_tol = self.opt_tol[-1:]

        # at least one sweep per outer iteration, so every outer iteration ends with a fresh phi
        if max_inner_iter < 1:
            raise ValueError('max_inner_iter must be at least 1')

        self.max_outer_iter = max_outer_iter
        self.max_inner_iter = max_inner_iter
        self.max_y = max_y
        self.save = save
        self.verbose = verbose
        self.history = [self.x.copy()] if self.save else []
        self.feas_log = [] # feasibility (max constraint violation) after each subproblem solve
        self.opt_log = []  # optimality (max-norm KKT residual) after each subproblem solve
        self.y = np.zeros_like(self.mu) # Lagrange multipliers, one per coupling constraint
        self.phi = np.zeros(0) if unconstrained else None # coupling constraints at the current x, from the latest subproblem solve
        self.success = False # set by solve()


    def _update_mu(self, c_new, c_old) -> None:
        """Grow the penalty of each constraint that is infeasible and not decreasing fast enough."""
        # vectorized over all constraints instead of a per-index Python loop
        f_new = np.abs(c_new)
        f_old = np.abs(c_old)
        grow = (f_new > self.tau * f_old) & (f_new > self.feas_tol)
        self.mu[grow] = np.minimum(self.rho * self.mu[grow], self.max_mu)

        return None


    def _set_inputs(self, subproblem) -> None:
        """Pass the current x, y, mu and any declared data entries to a subproblem."""
        subproblem.inputs["x"] = self.x
        subproblem.inputs["y"] = self.y
        subproblem.inputs["mu"] = self.mu

        # any other declared input comes from the data dictionary
        for name in subproblem.inputs.keys() - {"x", "y", "mu"}:
            subproblem.inputs[name] = self.data[name]

        return None


    def _get_outputs(self, subproblem) -> None:
        """Read x, phi and any extra outputs back from a solved subproblem."""
        self.x = subproblem.outputs["x"]

        # phi: the coupling constraints evaluated at the x this subproblem just returned.
        # np.array copies it, so a subproblem that reuses its output array can't alter c_old.
        # Unconstrained subproblems have no phi, so it stays empty
        if not self.unconstrained:
            if subproblem.outputs["phi"] is None:
                raise RuntimeError(f'{type(subproblem).__name__}.solve() did not set outputs["phi"]')
            phi = np.array(subproblem.outputs["phi"], dtype=float)
            if phi.shape != self.mu.shape:
                raise ValueError(f'{type(subproblem).__name__}: outputs["phi"] has shape {phi.shape}, '
                                 f'expected {self.mu.shape} (one entry per penalty parameter in mu0)')
            self.phi = phi

        # any other declared output goes into the data dictionary
        for name in subproblem.outputs.keys() - {"x", "phi"}:
            self.data[name] = subproblem.outputs[name]

        return None


    def _optimality(self) -> float:
        """Max-norm KKT stationarity residual over all subproblem blocks at the current x."""
        # refresh every subproblem's inputs to the current state before checking
        # optimality -- a subproblem's own inputs may be stale relative to blocks
        # that moved after it
        for sub in self.subproblems:
            self._set_inputs(sub)

        return max(sub.residual(sub.inputs) for sub in self.subproblems)


    def solve(self) -> "ALBCD":

        t0 = time.perf_counter()
        log = print if self.verbose else lambda *args: None
        self.success = False
        phi_old = np.full_like(self.mu, np.inf)  # no penalty grows on the first outer iteration

        # a single outer iteration counter (counted from 1) shared by all phases, so each phase
        # resumes where the previous one stopped and max_outer_iter bounds the whole solve
        # outer_iters = iter(range(1, self.max_outer_iter + 1))
        k = j = 0 # outer iterations completed, sweeps in the latest outer iteration

        for phase, opt_tol in enumerate(self.opt_tol, start=1):  # tolerance phases

            # augmented Lagrangian outer loop
            # for k in outer_iters:
            while k < self.max_outer_iter:
                k += 1

                # BCD inner loop
                for j in range(1, self.max_inner_iter + 1):

                    for sub in self.subproblems:

                        self._set_inputs(sub)
                        sub.solve(sub.inputs, sub.outputs)
                        self._get_outputs(sub)

                        self.history.append(self.x.copy())

                        # ready = all_blocks_solved or i == len(self.subproblems) - 1
                        # self.opt_history.append(self._optimality() if ready else np.nan)
                        # self.opt_history.append(self._optimality())

                        # if self.save:
                        #     self.history.append(self.x.copy())
                        #     # initial=0 gives an empty (unconstrained) phi a feasibility of 0
                        #     self.feas_history.append(float(np.max(np.abs(self.phi), initial=0.0)))
                        #     # residual() may need data a block caches when it solves, so the
                        #     # optimality is undefined until every block has solved once
                        #     ready = all_blocks_solved or i == len(self.subproblems) - 1
                        #     self.opt_history.append(self._optimality() if ready else np.nan)

                        self.feas_log.append(float(np.max(np.abs(self.phi), initial=0.0)))

                    res = self._optimality()  # get the current optimality residual
                    self.opt_log.append(res)

                    log(f'outer {k:3d} | sweep {j:3d} | opt_res {res:.3e}')

                    if res <= opt_tol:
                        break


                feas = np.max(np.abs(self.phi), initial=0.0)  # 0 when unconstrained

                log(f'outer {k:3d} | feas {feas:.3e} | max mu {np.max(self.mu):.3e} | |y| {np.linalg.norm(self.y):.3e}')

                # feasible in the final phase: stop, and skip the multiplier update.
                # success also requires the last inner loop to have reached opt_tol
                if feas <= self.feas_tol and phase == len(self.opt_tol):
                    self.success = res <= opt_tol
                    break  # converged

                # update the multipliers (elementwise), then clip to the magnitude bound
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
