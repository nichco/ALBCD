import time
from typing import Optional, Sequence
import numpy as np
from scipy.linalg import null_space, qr
from scipy.optimize import BFGS


class ALBCD3():
    """ALBCD with a second-order (Newton) multiplier update that tolerates inexact block coordinate descent.

    Solves ``min f(x)  s.t.  phi(x) = 0``. Each outer iteration runs sweeps of block coordinate
    descent on the augmented Lagrangian

        L_mu(x, y) = f(x) + y^T phi(x) + 1/2 sum_i mu_i phi_i(x)^2

    and then updates the multipliers. The first-order update is v = y + mu * phi. The
    second-order update is one Newton step on the optimality conditions

        grad_x L(x, y) = 0,   phi(x) = 0          (L = f + y^T phi, plus the blocks' local constraints)

    taken from (x, v):

        [ B   N^T ] [ d     ]     [ g   ]
        [ N   0   ] [ delta ] = - [ phi ],        y <- v + delta,    x <- x + d  (optional)

    restricted to the directions d that keep the blocks' active local constraints fixed. Here
    N = dphi/dx, B is the Hessian of L and g = grad_x L_mu(x, y) = grad_x L(x, v) is whatever
    gradient the sweeps left behind. With g = 0 the multiplier part is Bertsekas' second-order
    update y + D^-1 phi, D = N (B + N^T mu N)^-1 N^T; with g != 0 it is his update for inexact
    minimization, y + D^-1 (phi - N (B + N^T mu N)^-1 g). Written as above it needs neither mu
    nor a positive definite B + N^T mu N, only curvature on the tangent space of all constraints.

    Hessian. B is built block by block, which is exact whenever f and phi are sums of terms
    that each involve one block (every formulation that couples the blocks through copies).
    Each block of B is a symmetric rank-one (SR1) approximation that starts at zero and is fed
    by the iterates of that block's own minimizations, so it costs no extra gradients. Unlike
    BFGS, SR1 can represent the zero and negative curvature that copies of coupling variables
    have; a positive definite approximation invents curvature there, which the update then
    divides by the (small) coupling scale squared. If the objective or a coupling constraint
    multiplies variables of different blocks, set cross_terms=True to estimate the
    off-diagonal blocks as well.

    Safeguards. (1) Without curvature information in some tangent direction the update is
    first order. (2) A Newton step should remove the constraint violation; if the sweeps that
    follow at least double it instead, the step is undone (x, y and data return to where it
    was taken, with the first-order multipliers) and the next attempts are postponed. After a
    smaller increase the next update is first order. (3) x + d is used only if phi there,
    which the linear model predicts to be zero, is at most 10% of the current violation.

    On top of solve(), each subproblem provides

        gradient(x, y, mu, data) -> ndarray
            Gradient of the block's augmented Lagrangian with respect to its own variables
            at x, including the terms of its local constraints with their current
            multipliers. It must be valid at any x, not only where solve() ended.
        residual(x, y, mu, data) -> float
            Max-norm KKT residual of the block at x, for the inner stopping test.
        active(x, y, mu, data) -> ndarray, shape (k, block size)                 [optional]
            Jacobian of the block's active local constraints at x, with one unit row per
            active bound. Blocks without it are treated as unconstrained.
        coupling_jacobian(x, data) -> ndarray, shape (m, block size)             [optional]
            dphi/d(block variables) at x. Without it, it is built from m + 1 calls of gradient().
        trace -> list of (v, gradient)                                           [optional]
            The block variables at every iterate of the latest solve() and what gradient()
            would have returned there. Without it the block contributes one secant pair per
            solve (its start and end points), at the cost of two calls of gradient().
        evaluate(x, data, outputs) -> None                                       [optional]
            Like solve() without the minimization: set outputs["phi"] and whatever the block
            shares through data, at the given x. The x part of the Newton step is taken only
            if every block has it.
        hessian(x, v, data) -> ndarray                                           [optional]
            Hessian of the block's Lagrangian (mu = 0, multipliers v). Used by hessian='exact'.

    Parameters are those of ALBCD2, plus

        multiplier_update   'second_order' or 'first_order'
        primal_step         also take the x part of the Newton step (needs evaluate())
        cross_terms         estimate the off-diagonal blocks of B too (two more calls of every
                            block's gradient() per block solve)
        hessian             'sr1'; 'exact' (the blocks' hessian(), for comparison); or 'bfgs'
                            (one damped-BFGS matrix for all variables, fed by one secant pair
                            per sweep; it diverges on problems with copies, for comparison)
        inexact             False drops g from the Newton step (assumes exact minimization)
        safeguard           False disables safeguard (2)

    Unlike ALBCD2, a feasible point whose inner loop has not converged is not a failure: the
    outer loop keeps sweeping until both tolerances are met or the iterations run out.
    """

    def __init__(self,
                 subproblems: Sequence,
                 x0: np.ndarray,
                 mu0: np.ndarray,
                 data0: Optional[dict] = None,
                 multiplier_update: str = 'second_order',
                 hessian: str = 'sr1',
                 inexact: bool = True,
                 cross_terms: bool = False,
                 safeguard: bool = True,
                 primal_step: bool = True,
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
        if hessian not in ('sr1', 'exact', 'bfgs'):
            raise ValueError("hessian must be 'sr1', 'exact' or 'bfgs'")

        self.subproblems = subproblems
        self.x = np.array(x0, dtype=float) # copy, so the caller's x0 is never modified
        self.data = {} if data0 is None else dict(data0) # other quantities shared between subproblems
        self.multiplier_update = multiplier_update
        self.hessian = hessian
        self.inexact = inexact
        self.cross_terms = cross_terms
        self.safeguard = safeguard
        self.primal_step = primal_step

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
        self.update_history = [] # which multiplier update each outer iteration took

        n = self.x.size
        self.B = np.zeros((n, n))   # approximation of the Lagrangian Hessian
        self._pairs = []            # (block, s, change of the AL gradient, mu) not yet in B
        self._cross = []            # (block moved, s, change of the Lagrangian gradient of all blocks)
        self._second_order_ok = True
        self._undo = None           # state to return to if the latest second-order update fails
        self._wait = 0              # outer iterations left before the second-order update is tried again
        self._backoff = 0
        self.rejected = 0           # second-order updates that were undone

        if hessian == 'bfgs':
            self._bfgs = BFGS(exception_strategy="damp_update", init_scale="auto")
            self._bfgs.initialize(n, "hess")
            self._x_old = None
            self._data_old = None

    def _log(self, *args) -> None:
        if self.verbose:
            print(*args, flush=True)

    # ---------------------------------------------------------------- first-order information

    def _gradient(self, x, y, mu, data) -> np.ndarray:
        g = np.zeros_like(x)
        for sp in self.subproblems:
            g[sp.index] = sp.gradient(x, y, mu, data)
        return g

    def _coupling_jacobian(self, x, data) -> np.ndarray:
        m = self.mu.size
        zero = np.zeros(m)
        N = np.zeros((m, x.size))
        for sp in self.subproblems:
            if hasattr(sp, "coupling_jacobian"):
                N[:, sp.index] = sp.coupling_jacobian(x, data)
            else:
                # with mu = 0 a block's gradient is affine in y with slope (dphi/dv)^T,
                # so row j is gradient(y = e_j) - gradient(y = 0); every other term cancels
                g0 = sp.gradient(x, zero, zero, data)
                for j in range(m):
                    e = np.zeros(m)
                    e[j] = 1.0
                    N[j, sp.index] = sp.gradient(x, e, zero, data) - g0
        return N

    def _active_null_space(self, x, y, mu, data) -> np.ndarray:
        # basis Z of the directions that keep every block's active local constraints fixed
        n = x.size
        columns = []
        for sp in self.subproblems:
            E = np.eye(n)[:, sp.index]
            if hasattr(sp, "active"):
                A = np.atleast_2d(sp.active(x, y, mu, data))
                if A.shape[0]:
                    E = E @ null_space(A)
            columns.append(E)
        return np.hstack(columns)

    # ---------------------------------------------------------------- Hessian of the Lagrangian

    @staticmethod
    def _sr1(B, s, y) -> None:
        # symmetric rank-one update of B (in place) with the secant pair B s = y
        r = y - B @ s
        d = r @ s
        if abs(d) > 1e-8 * np.linalg.norm(s) * np.linalg.norm(r):
            B += np.outer(r, r) / d

    def _collect_pairs(self, i, sp, x_before, x_after) -> None:
        # secant pairs of block i's augmented Lagrangian from the solve that just ended
        if hasattr(sp, "trace"):
            trace = sp.trace
        else:
            trace = [(x_before[sp.index], sp.gradient(sp.recompose(x_after, x_before[sp.index]), self.y, self.mu, self.data)),
                     (x_after[sp.index], sp.gradient(x_after, self.y, self.mu, self.data))]
        for (va, ga), (vb, gb) in zip(trace[:-1], trace[1:]):
            if np.any(vb != va):
                self._pairs.append((i, vb - va, gb - ga, self.mu.copy()))

    def _update_hessian(self, N, v) -> None:
        if self.hessian == 'exact':
            for sp in self.subproblems:
                self.B[sp.index, sp.index] = sp.hessian(self.x, v, self.data)
            self._update_cross_terms(N)
        elif self.hessian == 'bfgs':
            self.B = self._bfgs.get_matrix()
        else:
            for i, s, dg, mu in self._pairs:
                idx = np.arange(self.x.size)[self.subproblems[i].index]
                Ni, Bi = N[:, idx], self.B[np.ix_(idx, idx)]
                # the pair is of L_mu; without the penalty curvature N^T mu N it is a pair of L
                self._sr1(Bi, s, dg - Ni.T @ (mu * (Ni @ s)))
                self.B[np.ix_(idx, idx)] = Bi
            self._update_cross_terms(N)
        self._pairs = []

    def _update_cross_terms(self, N) -> None:
        # off-diagonal blocks of B: when block j moved by s, the Lagrangian gradient of block i
        # changed by B_ij s. Broyden's rank-one update of B_ij with that secant pair, in the
        # variables scaled by the diagonal of the augmented Lagrangian's Hessian.
        n = self.x.size
        for j, s, dg, mu in self._cross:
            w = np.abs(np.diag(self.B)) + np.sum(mu[:, None] * N**2, axis=0)
            w = np.where(w > 0, w, 1.0)
            jdx = np.arange(n)[self.subproblems[j].index]
            dg = dg - N.T @ (mu * (N[:, jdx] @ s))   # remove the penalty curvature
            for i, sp in enumerate(self.subproblems):
                if i != j:
                    idx = np.arange(n)[sp.index]
                    Bij = self.B[np.ix_(idx, jdx)]
                    Bij = Bij + np.outer(dg[idx] - Bij @ s, w[jdx] * s) / (s @ (w[jdx] * s))
                    self.B[np.ix_(idx, jdx)] = Bij
                    self.B[np.ix_(jdx, idx)] = Bij.T
        self._cross = []

    def _update_bfgs(self) -> None:
        # hessian='bfgs': one secant pair of the Lagrangian Hessian per sweep, for all variables at once
        if self._x_old is not None:
            v = self.y + self.mu * self.phi
            zero = np.zeros_like(self.mu)
            g_new = self._gradient(self.x, v, zero, self.data)
            g_old = self._gradient(self._x_old, v, zero, self._data_old)
            self._bfgs.update(self.x - self._x_old, g_new - g_old)
        self._x_old = self.x.copy()
        self._data_old = dict(self.data)

    # ---------------------------------------------------------------- multiplier update

    @staticmethod
    def _newton_step(B, N, Z, g, phi):
        """Solve [B N^T; N 0] [d; delta] = -[g; phi] over d in span(Z) by the null-space method.

        Returns (delta, d), or None if the system is singular in a way that matters.
        """
        # work in variables scaled by the diagonal of B + N^T N, so that the orthogonal bases
        # below do not depend on the units of the variables (which can differ by 1e7)
        c = np.sqrt(np.abs(np.diag(B)) + np.sum(N**2, axis=0))
        c = np.where(c > 0, c, 1.0)
        S, _ = np.linalg.qr(Z * c[:, None])        # orthonormal basis of the allowed directions, scaled variables
        S = S / c[:, None]                         # the same basis in x
        NZ, Br, gr = N @ S, S.T @ B @ S, S.T @ g

        Q, R, _ = qr(NZ.T, pivoting=True)
        diag = np.abs(np.diag(R))
        rank = int(np.sum(diag > 1e-10 * diag[0])) if diag.size and diag[0] > 0 else 0
        Y, W = Q[:, :rank], Q[:, rank:]            # range of NZ^T, null space of NZ

        NY = NZ @ Y
        p = Y @ np.linalg.lstsq(NY, -phi, rcond=None)[0]       # restores phi = 0 to first order
        if W.shape[1]:
            # tangential step: stationary point of the model on the null space of NZ
            Rw = W.T @ Br @ W
            eig = np.linalg.eigvalsh(0.5 * (Rw + Rw.T))
            if np.min(np.abs(eig)) <= 1e-10 * np.max(np.abs(eig)):
                return None                        # no curvature information in some tangent direction
            p = p - W @ np.linalg.solve(Rw, W.T @ (gr + Br @ p))
        delta = np.linalg.lstsq(NY.T, -Y.T @ (gr + Br @ p), rcond=None)[0]   # minimum norm if N is rank deficient
        return delta, S @ p

    def _update_y(self, y, mu, phi):
        # returns the new multipliers and, after a second-order update, the x part of the Newton step

        v = y + mu * phi   # first-order update
        update, d = 'first_order', None

        if self.multiplier_update == 'second_order':
            N = self._coupling_jacobian(self.x, self.data)
            self._update_hessian(N, v)
            if self.safeguard and self._wait > 0:
                self._wait -= 1
            elif self._second_order_ok or not self.safeguard:
                Z = self._active_null_space(self.x, y, mu, self.data)
                g = self._gradient(self.x, y, mu, self.data) if self.inexact else np.zeros_like(self.x)
                step = self._newton_step(self.B, N, Z, g, phi)
                if step is not None and np.all(np.isfinite(step[0])):
                    if self.safeguard:
                        self._undo = dict(x=self.x.copy(), data=dict(self.data), phi=phi.copy(),
                                          y=np.clip(v, -self.max_y, self.max_y))
                    v = v + step[0]
                    update, d = 'second_order', step[1]

        self.update_history.append(update)
        return np.clip(v, -self.max_y, self.max_y), d

    def _take_primal_step(self, d) -> None:
        # x <- x + d if the linearization behind d holds there. Needs every block's evaluate(),
        # which recomputes the block's outputs (phi and whatever it shares through data) at a given x.
        if d is None or not all(hasattr(sp, "evaluate") for sp in self.subproblems):
            return
        x, data = self.x + d, dict(self.data)
        for sp in self.subproblems:
            outputs = {}
            sp.evaluate(x, data, outputs)
            phi = np.array(outputs.pop("phi"), dtype=float).reshape(self.mu.shape)
            data.update(outputs)
        # the Newton step predicts phi = 0 at x + d; accept it if at least 90% of that reduction is real
        if np.all(np.isfinite(phi)) and np.max(np.abs(phi)) <= 0.1 * np.max(np.abs(self.phi)):
            self.x, self.data = x, data
            self.x_history.append(self.x.copy())
            self.update_history[-1] = 'newton'

    # ---------------------------------------------------------------- main loops

    def _block_coordinate_descent(self, k) -> float:

        for j in range(1, self.max_inner_iter + 1):

            second_order = self.multiplier_update == 'second_order'
            # a block's gradient() may need its first solve, so no cross terms from the very first sweep
            cross = second_order and self.cross_terms and len(self.opt_history) > 0
            for i, sp in enumerate(self.subproblems):
                x_before = self.x
                if cross:
                    g_before = self._gradient(self.x, self.y, self.mu, self.data)
                outputs = {}
                sp.solve(self.x, self.y, self.mu, self.data, outputs)
                self.x = outputs.pop("x")
                self.phi = np.array(outputs.pop("phi"), dtype=float).reshape(self.mu.shape)
                self.data.update(outputs)  # any other output is shared through data
                self.x_history.append(self.x.copy())
                if second_order and self.hessian == 'sr1':
                    self._collect_pairs(i, sp, x_before, self.x)
                if cross and np.any(self.x != x_before):
                    g_after = self._gradient(self.x, self.y, self.mu, self.data)
                    self._cross.append((i, (self.x - x_before)[sp.index], g_after - g_before, self.mu.copy()))

            if second_order and self.hessian == 'bfgs':
                self._update_bfgs()

            opt = max(sp.residual(self.x, self.y, self.mu, self.data) for sp in self.subproblems)
            self.opt_history.append(opt)
            self.feas_history.append(float(np.max(np.abs(self.phi))))
            self._log(f'outer {k:3d} | sweep {j:3d} | opt_res {opt:.3e} | feas {self.feas_history[-1]:.3e}')

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
            self._log(f'outer {k:3d} | feas {feas:.3e} | max mu {np.max(self.mu):.3e} | |y| {np.linalg.norm(self.y):.3e}'
                      + (f' | {self.update_history[-1]}' if self.update_history else ''))

            if feas <= self.feas_tol:
                self.success = opt <= self.opt_tol
                if self.success:
                    break   # feasible but not yet stationary: keep sweeping

            # A Newton step is supposed to remove the constraint violation. If the violation
            # doubled instead, undo the step: return to the point it was taken from, with the
            # first-order multipliers, and wait before trying the second-order update again.
            if self._undo is not None:
                if not np.isfinite(feas) or feas > max(2 * np.max(np.abs(self._undo["phi"])), self.feas_tol):
                    self.x, self.data, self.phi, self.y = (self._undo[key] for key in ("x", "data", "phi", "y"))
                    self._backoff = max(1, 2 * self._backoff)
                    self._wait = self._backoff
                    self.rejected += 1
                    self.update_history[-1] = 'rejected'
                    self._undo = None
                    self._log(f'outer {k:3d} | second-order update undone (feas {feas:.3e}), next attempt in {self._wait} iterations')
                    continue
                self._undo, self._backoff = None, 0

            # after any increase of the constraint violation the next update is first order
            self._second_order_ok = np.max(np.abs(self.phi)) <= max(np.max(np.abs(phi_old)), self.feas_tol) \
                if np.all(np.isfinite(phi_old)) else True

            self.y, d = self._update_y(self.y, self.mu, self.phi)
            self.mu = self._update_mu(self.mu, self.phi, phi_old)
            phi_old = self.phi
            if self.primal_step:
                self._take_primal_step(d)

        self.outer_iterations = k
        self.tf = time.perf_counter() - t0
        self._log(f"{'Converged' if self.success else 'Did not converge'} after {k} outer iterations, "
                  f"{len(self.opt_history)} sweeps ({self.tf:.2f} s)")

        return self

    def _update_mu(self, mu, phi_new, phi_old) -> np.ndarray:
        # increase the penalty parameter of each scalar constraint if
        # it is infeasible and its feasibility is not decreasing fast enough.
        grow = np.abs(phi_new) > np.maximum(self.tau * np.abs(phi_old), self.feas_tol)
        return np.minimum(np.where(grow, self.rho * mu, mu), self.max_mu)
