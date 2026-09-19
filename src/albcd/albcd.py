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

    Parameters
    ----------
    subproblems : list of Subproblem
        One subproblem per block, solved in list order during each sweep.
    x0 : array_like
        Initial design vector.
    mu0 : array_like
        Initial penalty parameters, one per coupling constraint; sets the
        length of ``y`` and ``phi``.
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
    max_outer_iter : int
        Maximum number of outer iterations, counted across all phases.
    max_inner_iter : int
        Maximum number of sweeps per outer iteration.
    max_y : float
        Bound on the magnitude of each Lagrange multiplier.
    save : bool
        Record ``history`` and ``feas_history``.
    verbose : bool
        Print per-iteration diagnostics.

    Attributes
    ----------
    x, y, mu, phi : ndarray
        Design vector, multipliers, penalty parameters and coupling
        constraints; the solution once :meth:`solve` returns.
    success : bool
        True if the solve ended feasible (``max|phi| <= feas_tol``) and
        optimal (the last inner loop reached the final ``opt_tol``).
    data : dict
        Extra inputs/outputs shared between subproblems.
    history : list of ndarray
        ``x0`` followed by ``x`` after every subproblem solve.
    feas_history : list of float
        ``max|phi|`` at the end of each outer iteration.
    tf : float
        Wall-clock time of :meth:`solve` in seconds.
    """

    def __init__(self,
                 subproblems: Sequence,
                 x0: np.ndarray,
                 mu0: np.ndarray,
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
                 ):
        """Set up the solver. The parameters are described above."""

        self.subproblems = subproblems
        self.x = np.array(x0, dtype=float) # copy, so the caller's x0 is never modified
        self.data = {} if data0 is None else dict(data0) # extra inputs/outputs shared between subproblems; never holds x, y, mu or phi
        self.tf = None

        # every subproblem reports the coupling constraints ("phi") at the x it returns,
        # so the algorithm needs no separate constraint function
        for subproblem in self.subproblems:
            missing = {"x", "phi"} - subproblem.outputs.keys()
            if missing:
                raise ValueError(f'{type(subproblem).__name__} must declare outputs {sorted(missing)} in setup()')

        # force float dtype; an integer-dtype mu would otherwise truncate penalty growth
        self.mu = np.asarray(mu0, dtype=float) # penalty parameter(s)

        self.max_mu = max_mu
        self.rho = rho
        self.tau = tau
        self.feas_tol = feas_tol

        # one optimality tolerance per phase; a scalar gives a single phase
        self.opt_tol = np.atleast_1d(np.asarray(opt_tol, dtype=float))
        if self.opt_tol.ndim != 1 or self.opt_tol.size == 0:
            raise ValueError('opt_tol must be a float or a non-empty 1D list/array of floats')

        # at least one sweep per outer iteration, so every outer iteration ends with a fresh phi
        if max_inner_iter < 1:
            raise ValueError('max_inner_iter must be at least 1')

        self.max_outer_iter = max_outer_iter
        self.max_inner_iter = max_inner_iter
        self.max_y = max_y
        self.save = save
        self.verbose = verbose
        self.history = [self.x.copy()] if self.save else []
        self.feas_history = [] # feasibility (max constraint violation) at each outer iteration
        self.y = np.zeros_like(self.mu) # Lagrange multipliers, one per coupling constraint
        self.phi = None # coupling constraints at the current x, from the latest subproblem solve
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
        # np.array copies it, so a subproblem that reuses its output array can't alter c_old
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


    def solve(self) -> None:
        """Run ALBCD; the results are stored in the attributes ``x``, ``y``, ``mu``, ``phi``, ``data`` and ``success``."""

        t0 = time.perf_counter()
        num_phases = len(self.opt_tol)
        self.success = False

        # no constraint values exist before the first sweep; an infinite previous violation
        # means no constraint fails the progress test in _update_mu on the first outer
        # iteration, so mu is left unchanged there. Carried forward via c_old = c_new below
        c_old = np.full_like(self.mu, np.inf)

        # a single outer iteration counter (counted from 1) shared by all phases, so each phase
        # resumes where the previous one stopped and max_outer_iter bounds the whole solve
        outer_iters = iter(range(1, self.max_outer_iter + 1))
        k = 0 # outer iterations completed

        # tolerance phases: each phase runs outer iterations with its own inner loop
        # optimality tolerance until feasibility reaches feas_tol, then moves on to the next
        for phase, opt_tol in enumerate(self.opt_tol, start=1):
            final_phase = phase == num_phases

            # augmented Lagrangian outer loop
            for k in outer_iters:

                # BCD inner loop (sweeps counted from 1)
                for j in range(1, self.max_inner_iter + 1):

                    for subproblem in self.subproblems:

                        self._set_inputs(subproblem)
                        subproblem.outputs["phi"] = None # cleared so a solve() that doesn't set phi is caught
                        subproblem.solve(subproblem.inputs, subproblem.outputs)
                        self._get_outputs(subproblem)

                        if self.save:
                            self.history.append(self.x.copy())

                    # refresh every subproblem's inputs to the post-sweep state before
                    # checking optimality -- a subproblem's own inputs may be stale
                    # relative to blocks that moved later in this sweep
                    for subproblem in self.subproblems:
                        self._set_inputs(subproblem)

                    # max-norm KKT stationarity residual over all subproblem blocks
                    opt_res = max(subproblem.residual(subproblem.inputs) for subproblem in self.subproblems)

                    if self.verbose:
                        print(f'outer {k:3d} | sweep {j:3d} | opt_res {opt_res:.3e}')

                    if opt_res <= opt_tol:
                        break


                # the last subproblem solved in the sweep evaluated phi at the current x
                c_new = self.phi
                feas = np.max(np.abs(c_new))
                if self.save:
                    self.feas_history.append(feas)

                if self.verbose:
                    print(f'outer {k:3d} | feas {feas:.3e} | max mu {np.max(self.mu):.3e} | |y| {np.linalg.norm(self.y):.3e}')

                # feasible in the final phase: stop, and skip the multiplier update.
                # success also requires the last inner loop to have reached opt_tol
                if feas <= self.feas_tol and final_phase:
                    self.success = opt_res <= opt_tol
                    break

                # update the multipliers (elementwise), then clip to the magnitude bound
                self.y += self.mu * c_new
                np.clip(self.y, -self.max_y, self.max_y, out=self.y)

                # update mu on a per-scalar-constraint basis
                self._update_mu(c_new, c_old)
                c_old = c_new

                # feasible in an intermediate phase: move on to the next opt_tol
                if feas <= self.feas_tol:
                    if self.verbose:
                        print(f'Phase {phase} complete, starting phase {phase + 1}')
                    break

        self.tf = time.perf_counter() - t0
        if self.verbose:
            print(f"{'Converged' if self.success else 'Did not converge'} after {k} outer iterations ({self.tf:.2f} s)")
        return None
