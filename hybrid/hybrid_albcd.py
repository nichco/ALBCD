"""Hybrid distributed-monolithic ALBCD (prototype).

Phase 1 is ALBCD as in src/albcd/albcd.py: block sweeps on the augmented Lagrangian, with the
first-order multiplier update and the penalty update between inexact inner loops. Phase 2 solves the
full (monolithic) problem with modOpt's OpenSQP, started from the phase-1 iterate.

ALBCD makes fast early progress with cheap, independent block solves, but its tail is linear and floors
at the accuracy of the block solves. An SQP method on the full problem is the opposite: slow from a poor
start, where it has no curvature information and a wrong active set, and fast near the solution. The
hybrid uses each where it is strong. When to hand over is decided by a switching criterion, a function of
the solver that is called after every sweep (see the criteria at the end of this file).

The solver needs both descriptions of the problem:

    subproblems   the blocks, exactly as for ALBCD (Subproblem.solve and Subproblem.gradient)
    full_model    a function (x, y, data) -> modOpt problem, or -> (modOpt problem, init_multipliers)

``full_model`` receives the phase-1 iterate (the design vector x, the multipliers y of the coupling
constraints and the shared data) and returns the full problem as a modOpt problem whose x0 is the point
that x maps to. The full problem can use its own variables, e.g. without the copies that the
decomposition introduced. ``init_multipliers``, if returned, is OpenSQP's warm start for the multipliers:
a dict of arrays 'cl', 'cu', 'xl', 'xu' in the scaled problem.
"""

import time
import numpy as np
import modopt as mo
from albcd import ALBCD


class RecordingOpenSQP(mo.OpenSQP):
    """OpenSQP that keeps (wall time, x, optimality, feasibility, objective) at every major iteration;
    iteration 0 is the starting point."""

    def solve(self):
        self.history = []
        self.t_start = time.perf_counter()
        return super().solve()

    def update_outputs(self, **kwargs):
        self.history.append((time.perf_counter() - self.t_start, np.array(kwargs["x"]), kwargs["opt"],
                             kwargs["feas"], kwargs["obj"]))
        super().update_outputs(**kwargs)


def init_multipliers(problem, lam):
    """OpenSQP's ``init_multipliers`` from an estimate ``lam`` of the constraint multipliers at problem.x0.

    ``lam`` is in the scaled problem, with the sign of grad f = J^T lam. OpenSQP keeps separate, nonnegative
    multipliers for the lower and upper sides of the constraints and of the variable bounds. The bound
    multipliers are estimated from the stationarity residual grad f - J^T lam at the variables on a bound.
    """
    x = problem.x0
    lam = np.asarray(lam, dtype=float)
    equality = problem.c_lower == problem.c_upper
    r = problem._compute_objective_gradient(x) - problem._compute_constraint_jacobian(x).T @ lam
    tol = 1e-8 * (1 + np.abs(x))
    return {"cl": np.where(equality, lam, np.maximum(lam, 0.0)), "cu": np.maximum(-lam, 0.0),
            "xl": np.where(x <= problem.x_lower + tol, np.maximum(r, 0.0), 0.0),
            "xu": np.where(x >= problem.x_upper - tol, np.maximum(-r, 0.0), 0.0)}


class HybridALBCD(ALBCD):

    def __init__(self, subproblems, full_model, x0, mu0=None, *,
                 switch=None,
                 sqp_options: dict = None,
                 **albcd_options):
        """
        switch : callable(solver) -> bool, called after every sweep; phase 1 ends when it returns True.
            Phase 1 also ends when ALBCD meets its own feas_tol and opt_tol, or runs out of outer
            iterations, so with switch=None the full problem is solved from the ALBCD result.
        sqp_options : options of OpenSQP for phase 2, e.g. dict(maxiter=200, opt_tol=1e-6, feas_tol=1e-6).
        albcd_options : the remaining arguments of ALBCD (max_mu, rho, tau, feas_tol, opt_tol, ...).
        """
        super().__init__(subproblems, x0, mu0, **albcd_options)
        self.full_model = full_model
        self.switch = switch
        self.sqp_options = dict(sqp_options or {})
        self.switched = False     # True if the switching criterion ended phase 1
        self.z = None             # solution of the full problem, in its own (unscaled) variables
        self.sqp = None           # the OpenSQP optimizer of phase 2
        self.t_albcd = self.t_sqp = None

    # ------------------------------------------------------------------ phase 1: ALBCD

    def _sweep(self):
        """One sweep over the blocks, as in ALBCD._block_coordinate_descent."""
        for subproblem in self.subproblems:
            outputs = {}
            subproblem.solve(self.x, self.y, self.mu, self.data, outputs)
            self.x = outputs.pop("x")
            self.phi = np.array(outputs.pop("phi", []), dtype=float).reshape(self.mu.shape)
            self.data.update(outputs)
            self.history["x"].append(self.x.copy())

        optimality = max(np.linalg.norm(sp.gradient(self.x, self.y, self.mu, self.data), ord=np.inf)
                         for sp in self.subproblems)
        feas = np.max(np.abs(self.phi), initial=0.0)
        h = self.history
        h["optimality"].append(optimality)
        h["feasibility"].append(feas)
        # the state after every sweep: what the switching criteria see, and enough to restart phase 2 from
        h["time"].append(time.perf_counter() - self._t0)
        h["x_sweep"].append(self.x.copy())
        h["y"].append(self.y.copy())
        h["mu"].append(self.mu.copy())
        h["data"].append(dict(self.data))
        h["outer"].append(self._outer)
        return optimality, feas

    def _albcd(self):
        """ALBCD.solve, with the switching criterion checked after every sweep."""
        phi_old = np.full_like(self.mu, np.inf)
        for self._outer in range(1, self.max_outer_iter + 1):

            for i in range(self.max_inner_iter):
                optimality, feas = self._sweep()
                if self.switch is not None and self.switch(self):
                    self.switched = True
                    return
                # inexact inner loop: loose while feasibility is large, opt_tol near feas_tol
                if optimality <= self.opt_tol * max(1.0, self.kappa * feas / self.feas_tol):
                    break

            if feas <= self.feas_tol and optimality <= self.opt_tol:
                self.success = True
                return

            self.y = self._update_y(self.y, self.mu, self.phi)
            self.mu = self._update_mu(self.mu, self.phi, phi_old)
            phi_old = self.phi

    # ------------------------------------------------------------------ phase 2: OpenSQP on the full problem

    def _monolithic(self):
        model = self.full_model(self.x, self.y, self.data)
        problem, init_multipliers = model if isinstance(model, tuple) else (model, None)
        self.sqp = RecordingOpenSQP(problem, init_multipliers=init_multipliers, turn_off_outputs=True,
                                    **self.sqp_options)
        self.sqp.solve()
        self.z = self.sqp.results["x"] / problem.x_scaler
        self.success = bool(self.sqp.results["success"])

    def solve(self) -> "HybridALBCD":
        self._t0 = time.perf_counter()
        self._albcd()
        self.t_albcd = time.perf_counter() - self._t0
        sweeps = len(self.history["optimality"])
        reason = "switching criterion" if self.switched else ("ALBCD converged" if self.success else "ALBCD stopped")
        self._log(f"phase 1 (ALBCD): {sweeps} sweeps, {self.t_albcd:.2f} s, optimality "
                  f"{self.history['optimality'][-1]:.2e}, feasibility {self.history['feasibility'][-1]:.2e} ({reason})")

        self._monolithic()
        self.tf = time.perf_counter() - self._t0
        self.t_sqp = self.tf - self.t_albcd
        r = self.sqp.results
        self._log(f"phase 2 (OpenSQP): {r['niter']} iterations, {self.t_sqp:.2f} s, optimality "
                  f"{r['optimality']:.2e}, feasibility {r['feasibility']:.2e}")
        self._log(f"{'Hybrid ALBCD converged' if self.success else 'Hybrid ALBCD failed'} ({self.tf:.2f} s)")
        return self


# ---------------------------------------------------------------------- switching criteria
# A criterion is a function of the solver, called after every sweep. It reads solver.history, which has
# one entry per sweep of "optimality", "feasibility", "time", "x_sweep", "y", "mu" and "outer" (the outer
# iteration the sweep belongs to), and the tolerances solver.opt_tol and solver.feas_tol.

def merit(solver):
    """max(optimality / opt_tol, feasibility / feas_tol) after every sweep: ALBCD has converged at 1."""
    h = solver.history
    return np.maximum(np.array(h["optimality"]) / solver.opt_tol, np.array(h["feasibility"]) / solver.feas_tol)


def after_sweeps(n):
    """Switch after a fixed number of sweeps."""
    return lambda solver: len(solver.history["optimality"]) >= n


def feasibility_below(factor):
    """Switch once the coupling constraints are within factor * feas_tol."""
    return lambda solver: solver.history["feasibility"][-1] <= factor * solver.feas_tol


def merit_below(factor):
    """Switch once ALBCD is within a factor of its own tolerances, in both optimality and feasibility."""
    return lambda solver: merit(solver)[-1] <= factor


def stagnation(rate=0.7, window=3):
    """Switch when ALBCD slows down: the best merit so far has fallen by less than a factor ``rate`` per
    sweep, on average, over the last ``window`` sweeps."""
    def criterion(solver):
        best = np.minimum.accumulate(merit(solver))
        return len(best) > window and best[-1] > rate**window * best[-1 - window]
    return criterion


def multipliers_settled(tol=0.1):
    """Switch when a multiplier update changes y by less than tol, relative to y: the multipliers of the
    coupling constraints, and with them the blocks' active sets, are then roughly identified. Checked on
    the first sweep after each update."""
    def criterion(solver):
        y = solver.history["y"]  # the multipliers each sweep was solved with
        if len(y) < 2 or solver.history["outer"][-1] == solver.history["outer"][-2]:
            return False
        return np.linalg.norm(y[-1] - y[-2], np.inf) <= tol * max(np.linalg.norm(y[-1], np.inf), 1e-12)
    return criterion


def step_below(tol, x_scale=1.0):
    """Switch when a sweep moves the (scaled) design by less than tol, relative to its size."""
    def criterion(solver):
        xs = solver.history["x_sweep"]
        if len(xs) < 2:
            return False
        return np.linalg.norm((xs[-1] - xs[-2]) * x_scale, np.inf) <= tol * np.linalg.norm(xs[-1] * x_scale, np.inf)
    return criterion
