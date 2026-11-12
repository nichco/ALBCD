import time
from typing import Optional, Sequence
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import BFGS


class ALBCD3():
    """ALBCD with Bertsekas' second-order multiplier update.

    Solves ``min f(x)  s.t.  phi(x) = 0`` like ALBCD2: each outer iteration minimizes the
    augmented Lagrangian

        L_mu(x, y) = f(x) + y^T phi(x) + 1/2 sum_i mu_i phi_i(x)^2

    by block coordinate descent, then updates the multipliers. The dual function
    q(y) = min_x L_mu(x, y) has gradient phi and Hessian -D, where

        D = N Z (Z^T H Z)^-1 Z^T N^T,      H = B + N^T diag(mu) N,

    N = dphi/dx, B is the Hessian of the Lagrangian f + v^T phi (plus the blocks' local
    constraint terms) and the columns of Z span the null space of the blocks' active local
    constraints. The second-order update is Newton's method on q:

        y <- y + D^-1 phi          (first order: y <- y + mu * phi)

    Hessians are not available, so B is a damped-BFGS approximation. It gets one secant
    pair per sweep: the change in x since the previous sweep and the change in the
    Lagrangian gradient between those two points, both gradients taken with the current
    multiplier estimate v = y + mu * phi. B does not depend on mu, so it carries over
    between outer iterations unchanged. There are no safeguards on the update.

    On top of solve(), each subproblem must provide

        gradient(x, y, mu, data) -> ndarray
            Gradient of the block's augmented Lagrangian with respect to its own variables
            at x, including the terms of its local constraints with their current
            multipliers (so it is zero at a block solution with no active bounds). It must
            be valid at any x, not only where solve() ended.
        residual(x, y, mu, data) -> float
            Max-norm KKT residual of the block at x, for the inner stopping test.
        active(x, y, mu, data) -> ndarray, shape (k, block size)        [optional]
            Jacobian of the block's active local constraints at x, with one unit row per
            active bound. Blocks without it are treated as unconstrained.

    Parameters are those of ALBCD2, plus ``multiplier_update``: 'second_order' or
    'first_order' (the same driver with the usual update, for comparison).
    """

    def __init__(self,
                 subproblems: Sequence,
                 x0: np.ndarray,
                 mu0: np.ndarray,
                 data0: Optional[dict] = None,
                 multiplier_update: str = 'second_order',
                 max_mu: float = 1e3,
                 rho: float = 1.2,
                 tau: float = 0.5,
                 feas_tol: float = 1e-3,
                 opt_tol: float = 1e-4,
                 max_outer_iter: int = 100,
                 max_inner_iter: int = 10,
                 max_y: float = 1e6,
                 verbose: bool = True,
                 ):

        if multiplier_update not in ('second_order', 'first_order'):
            raise ValueError("multiplier_update must be 'second_order' or 'first_order'")

        self.subproblems = subproblems
        self.x = np.array(x0, dtype=float) # copy, so the caller's x0 is never modified
        self.data = {} if data0 is None else dict(data0) # other quantities shared between subproblems
        self.multiplier_update = multiplier_update

        self.mu = np.array(mu0, dtype=float) # penalty parameters (copy, float)
        self.max_mu = max_mu
        self.rho = rho
        self.tau = tau
        self.feas_tol = feas_tol
        self.opt_tol = opt_tol
        self.max_outer_iter = max_outer_iter
        self.max_inner_iter = max_inner_iter
        self.max_y = max_y
        self.verbose = verbose

        self.y = np.zeros_like(self.mu) # Lagrange multipliers, one per coupling constraint
        self.phi = None                 # coupling constraints at the current x
        self.success = False
        self.tf = None
        self.outer_iterations = 0
        self.x_history = [self.x.copy()]
        self.feas_history = [] # max|phi| after each sweep
        self.opt_history = []  # max block residual after each sweep

        # damped-BFGS approximation of the Lagrangian Hessian with respect to all variables
        self.B = BFGS(exception_strategy="damp_update", init_scale="auto")
        self.B.initialize(self.x.size, "hess")
        self._x_old = None    # x and data at the end of the previous sweep
        self._data_old = None

    def _log(self, *args) -> None:
        if self.verbose:
            print(*args)

    def _lagrangian_gradient(self, x, v, data) -> np.ndarray:
        # gradient of f + v^T phi (and the blocks' local constraint terms) with respect to all variables
        zero = np.zeros_like(self.mu)
        g = np.zeros_like(x)
        for sp in self.subproblems:
            g[sp.index] = sp.gradient(x, v, zero, data)
        return g

    def _coupling_jacobian(self, x, data) -> np.ndarray:
        # N = dphi/dx. With mu = 0 a block's gradient is affine in y with slope (dphi/dv)^T,
        # so row j is gradient(y = e_j) - gradient(y = 0); every other term cancels.
        m = self.mu.size
        zero = np.zeros(m)
        N = np.zeros((m, x.size))
        for sp in self.subproblems:
            g0 = sp.gradient(x, zero, zero, data)
            for j in range(m):
                e = np.zeros(m)
                e[j] = 1.0
                N[j, sp.index] = sp.gradient(x, e, zero, data) - g0
        return N

    def _active_null_space(self, x, y, mu, data) -> np.ndarray:
        # basis Z of the directions that keep every block's active local constraints fixed
        n = x.size
        rows = []
        for sp in self.subproblems:
            if hasattr(sp, "active"):
                A_block = np.atleast_2d(sp.active(x, y, mu, data))
                A = np.zeros((A_block.shape[0], n))
                A[:, sp.index] = A_block
                rows.append(A)
        A = np.vstack(rows) if rows else np.zeros((0, n))
        return null_space(A) if A.shape[0] else np.eye(n)

    def _update_hessian(self) -> None:
        # one secant pair of the Lagrangian Hessian from the step since the previous sweep
        if self._x_old is not None:
            v = self.y + self.mu * self.phi
            g_new = self._lagrangian_gradient(self.x, v, self.data)
            g_old = self._lagrangian_gradient(self._x_old, v, self._data_old)
            self.B.update(self.x - self._x_old, g_new - g_old)
        self._x_old = self.x.copy()
        self._data_old = dict(self.data)

    def _update_mu(self, mu, phi_new, phi_old) -> np.ndarray:
        # increase the penalty parameter of each scalar constraint if
        # it is infeasible and its feasibility is not decreasing fast enough.
        grow = np.abs(phi_new) > np.maximum(self.tau * np.abs(phi_old), self.feas_tol)
        return np.minimum(np.where(grow, self.rho * mu, mu), self.max_mu)

    def _update_y(self, y, mu, phi) -> np.ndarray:

        if self.multiplier_update == 'first_order':
            dy = mu * phi
        else:
            N = self._coupling_jacobian(self.x, self.data)
            Z = self._active_null_space(self.x, y, mu, self.data)
            NZ = N @ Z
            H = Z.T @ self.B.get_matrix() @ Z + NZ.T @ (mu[:, None] * NZ) # reduced Hessian of L_mu
            D = NZ @ np.linalg.solve(H, NZ.T)                             # -(Hessian of the dual function)
            # D is singular where the multipliers are not unique (e.g. two consensus copies on the
            # same active local constraint), so take the minimum-norm solution of D dy = phi
            dy = np.linalg.lstsq(D, phi, rcond=None)[0]

        return np.clip(y + dy, -self.max_y, self.max_y)

    def _block_coordinate_descent(self, k) -> float:

        for j in range(1, self.max_inner_iter + 1):

            for sp in self.subproblems:
                outputs = {}
                sp.solve(self.x, self.y, self.mu, self.data, outputs)
                self.x = outputs.pop("x")
                self.phi = np.array(outputs.pop("phi"), dtype=float).reshape(self.mu.shape)
                self.data.update(outputs)  # any other output is shared through data
                self.x_history.append(self.x.copy())

            if self.multiplier_update == 'second_order':
                self._update_hessian()

            opt = max(sp.residual(self.x, self.y, self.mu, self.data) for sp in self.subproblems)
            self.opt_history.append(opt)
            self.feas_history.append(float(np.max(np.abs(self.phi))))
            self._log(f'outer {k:3d} | sweep {j:3d} | opt_res {opt:.3e}')

            if opt <= self.opt_tol:
                break

        return opt

    def solve(self) -> "ALBCD3":

        t0 = time.perf_counter()
        phi_old = np.full_like(self.mu, np.inf)
        self.success = False

        k = 0
        while k < self.max_outer_iter:
            k += 1

            opt = self._block_coordinate_descent(k)
            feas = np.max(np.abs(self.phi))
            self._log(f'outer {k:3d} | feas {feas:.3e} | max mu {np.max(self.mu):.3e} | |y| {np.linalg.norm(self.y):.3e}')

            if feas <= self.feas_tol:
                self.success = opt <= self.opt_tol
                break

            self.y = self._update_y(self.y, self.mu, self.phi)
            self.mu = self._update_mu(self.mu, self.phi, phi_old)
            phi_old = self.phi

        self.outer_iterations = k
        self.tf = time.perf_counter() - t0
        self._log(f"{'Converged' if self.success else 'Did not converge'} after {k} outer iterations, "
                  f"{len(self.opt_history)} sweeps ({self.tf:.2f} s)")

        return self
