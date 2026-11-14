"""Experimental accelerated inner loop for ALBCD2 (benchmark only).

Options (all off by default -> identical to ALBCD2):

  accel = None | 'anderson' | 'aitken' | 'sor'
      anderson : type-II Anderson acceleration (= IQN-ILS of FSI, = GMRES on linear problems)
                 of the sweep map u -> G(u), u = (x, numeric data).
      aitken   : Irons-Tuck dynamic relaxation of the sweep map (OpenMDAO NLBGS / FSI style).
      sor      : fixed over-relaxation omega of each block right after its solve.
  m            : Anderson memory
  omega        : SOR factor
  warm         : None | 'secant'  -- multi-secant predictor of x(y, mu) across outer iterations
  inner_rule   : 'kappa' (ALBCD2) | 'relative' (Eckstein-Silva style: tol = sigma*||mu*phi||)
  skip         : skip a block whose own residual is already below skip_frac*tol
"""

import time
import numpy as np
from collections import defaultdict
from albcd.albcd2 import ALBCD2


class ALBCD2X(ALBCD2):

    def __init__(self, *args, accel=None, m=5, omega=1.5, warm=None, warm_m=3,
                 inner_rule='kappa', sigma=0.5, skip=False, skip_frac=0.1,
                 scale=None, safeguard=True, reuse=False, block_eta=None, xonly=False, omega_max=1.8, **kw):
        super().__init__(*args, **kw)
        self.accel = accel
        self.m = m
        self.omega = omega
        self.warm = warm
        self.warm_m = warm_m
        self.inner_rule = inner_rule
        self.sigma = sigma
        self.skip = skip
        self.skip_frac = skip_frac
        self.scale = scale
        self.safeguard = safeguard
        self.reuse = reuse
        self.omega_max = omega_max
        self.xonly = xonly
        self.block_eta = block_eta
        self._cols = []  # persistent (dF, dG) columns
        self.n_solves = 0
        self.n_grads = 0
        self.n_sweeps = 0
        self.n_reject = 0

    # --- state vector u = (x, numeric data) -------------------------------------------
    def _keys(self):
        if self.xonly: return []
        return sorted(k for k, v in self.data.items()
                      if not k.startswith('_') and isinstance(v, (float, int, np.floating, np.ndarray)) and np.size(v) > 0)

    def _pack(self):
        parts = [self.x] + [np.ravel(np.asarray(self.data[k], float)) for k in self._dkeys]
        return np.concatenate(parts)

    def _unpack(self, u):
        n = self.x.size
        self.x = u[:n].copy()
        off = n
        for k in self._dkeys:
            v = self.data[k]
            sz = np.size(v)
            new = u[off:off + sz]
            self.data[k] = float(new[0]) if np.ndim(v) == 0 else new.reshape(np.shape(v)).copy()
            off += sz

    def _grad_all(self):
        self.n_grads += len(self.subproblems)
        return max(np.linalg.norm(sp.gradient(self.x, self.y, self.mu, self.data), ord=np.inf)
                   for sp in self.subproblems)

    def _sweep(self, tol):
        for j, sp in enumerate(self.subproblems):
            if self.skip and j > 0 and self.n_sweeps > 0:
                # Gauss-Southwell-like laziness: skip a block that is already stationary
                self.n_grads += 1
                if np.linalg.norm(sp.gradient(self.x, self.y, self.mu, self.data), ord=np.inf) <= self.skip_frac * tol:
                    continue
            x_old = self.x
            d_old = {k: self.data.get(k) for k in self._dkeys}
            if self.block_eta is not None:
                self.data['_tol'] = self.block_eta * tol
            outputs = {}
            sp.solve(self.x, self.y, self.mu, self.data, outputs)
            self.n_solves += 1
            x_new = outputs.pop("x")
            self.phi = np.array(outputs.pop("phi", []), dtype=float).reshape(self.mu.shape)
            if self.accel in ('sor', 'sor_adaptive'):
                w = self.omega
                x_new = x_old + w * (x_new - x_old)
                lo = getattr(sp, 'xl', getattr(sp, 'XL', None)); hi = getattr(sp, 'xu', getattr(sp, 'XU', None))
                if lo is not None:
                    v = x_new[sp.index]; x_new[sp.index] = np.clip(v, lo, hi)
                for k, v in outputs.items():
                    if k in d_old and d_old[k] is not None:
                        outputs[k] = np.asarray(d_old[k]) + w * (np.asarray(v) - np.asarray(d_old[k]))
                        if np.ndim(v) == 0: outputs[k] = float(outputs[k])
            self.x = x_new
            self.data.update(outputs)
            self.history["x"].append(self.x.copy())
        self.n_sweeps += 1

    def _adapt_omega(self, u_in, u_out, i):
        # Young/Hageman: for a consistently ordered splitting (any 2-block partition) the SOR
        # eigenvalue q and the Jacobi eigenvalue b satisfy (q + w - 1)^2 = q w^2 b^2, and the
        # optimal w is 2 / (1 + sqrt(1 - b^2)). Estimate q by the power method on the sweep
        # displacements and back out b^2 (= Gauss-Seidel rate).
        if getattr(self, '_s', None) is None or self._s.size != u_out.size:
            self._s = 1.0 / np.maximum(np.abs(u_out), 1e-8 + 1e-3 * np.max(np.abs(u_out)))
        d = np.linalg.norm((u_out - u_in) * self._s)
        prev = self._dprev if i > 0 else None
        self._dprev = d
        if prev is None or prev == 0:
            return
        q = d / prev
        w = self.omega
        if q >= 1.0:
            # not contracting (nonlinearity, or w above optimum): back off toward Gauss-Seidel
            self.omega = max(1.0, 0.5 * (1.0 + w))
        elif q > w - 1.0:
            b2 = min((q + w - 1.0) ** 2 / (q * w * w), 0.995)
            w_opt = 2.0 / (1.0 + np.sqrt(1.0 - b2))
            # damp the change; never above w_max
            self.omega = float(min(self.omega_max, 0.5 * (w + w_opt)))
        self.history["omega"].append(self.omega)

    def _block_coordinate_descent(self, opt_tol):
        self._dkeys = self._keys()
        F, Gs = [], []         # residual and image history for Anderson
        r_prev = u_prev = None
        w_aitken = 1.0
        best = np.inf
        tol = getattr(self, '_last_tol', 1e-2)
        if not self.reuse: self._cols = []
        self._prev = None
        for i in range(self.max_inner_iter):
            u_in = self._pack()
            self._sweep(tol)
            u_out = self._pack()          # G(u_in): a consistent point (all solves done)

            optimality = self._grad_all()
            feas = np.max(np.abs(self.phi), initial=0.0)
            self.history["optimality"].append(optimality)
            self.history["feasibility"].append(feas)

            if self.inner_rule == 'relative':
                tol = max(opt_tol, self.sigma * np.max(np.abs(self.mu * self.phi), initial=0.0))
            else:
                tol = opt_tol * max(1.0, self.kappa * feas / self.feas_tol)
            self._last_tol = tol
            if optimality <= tol:
                self._log(f"inner converged opt {optimality:.2e} tol {tol:.2e} in {i+1}")
                break

            if self.accel == 'sor_adaptive':
                self._adapt_omega(u_in, u_out, i)
                continue
            # ---- acceleration of the fixed-point map (never touches the convergence test) ----
            if self.accel not in ('anderson', 'aitken'):
                continue
            if self.scale is None:
                if getattr(self, '_s', None) is None or self._s.size != u_out.size:
                    self._s = 1.0 / np.maximum(np.abs(u_out), 1e-8 + 1e-3 * np.max(np.abs(u_out)))
                s = self._s
            else:
                s = self.scale_vec if hasattr(self, 'scale_vec') else 1.0
            r = (u_out - u_in) * s

            if self.safeguard and optimality > 10 * best:
                # the extrapolated point made things worse: restart from the consistent point
                F, Gs = [], []
                self._cols = []; self._prev = None
                self.n_reject += 1
                w_aitken = 1.0
                best = optimality
                continue
            best = min(best, optimality)

            if self.accel == 'aitken':
                if r_prev is not None:
                    dr = r - r_prev
                    den = dr @ dr
                    if den > 0:
                        w_aitken = float(np.clip(-w_aitken * (r_prev @ dr) / den, 0.1, 5.0))
                r_prev = r
                self._unpack(u_in + w_aitken * (u_out - u_in))
                continue

            # anderson (type II) on persistent difference columns: differences are only formed
            # between sweeps of the same outer iteration, but columns from earlier outer
            # iterations are kept (IQN-ILS with reuse) since the sweep map changes slowly with y, mu
            if self._prev is not None:
                self._cols.append((r - self._prev[0], u_out - self._prev[1]))
                self._cols = self._cols[-self.m:]
            self._prev = (r, u_out)
            if self._cols:
                dF = np.array([c[0] for c in self._cols]).T
                dG = np.array([c[1] for c in self._cols]).T
                lam = 1e-10 * np.linalg.norm(dF) ** 2
                A = dF.T @ dF + lam * np.eye(dF.shape[1])
                gamma = np.linalg.solve(A, dF.T @ r)
                self._unpack(u_out - dG @ gamma)
        else:
            self._log(f"inner failed opt {optimality:.2e} in {i+1}")
        # the iterate handed to the outer loop is the consistent point G(u_in)
        if self.accel in ('anderson', 'aitken') and not np.array_equal(self._pack(), u_out):
            self._unpack(u_out)
        return optimality

    def solve(self):
        t0 = time.perf_counter()
        phi_old = np.full_like(self.mu, np.inf)
        success = False
        X, Y = [], []   # outer history for the secant predictor
        k = 0
        while k < self.max_outer_iter:
            k += 1
            optimality = self._block_coordinate_descent(self.opt_tol)
            feas = np.max(np.abs(self.phi))
            if feas <= self.feas_tol and optimality <= self.opt_tol:
                success = True
                break
            y_old = self.y.copy()
            self.y = self._update_y(self.y, self.mu, self.phi)
            self.mu = self._update_mu(self.mu, self.phi, phi_old)
            phi_old = self.phi

            if self.warm == 'secant':
                # multi-secant (least-change) predictor of the AL minimizer as a function of
                # the multipliers: u(y_new) ~ u(y_old) + dU dY^+ (y_new - y_old)
                self._dkeys = self._keys()
                u = self._pack()
                X.append(u); Y.append(np.concatenate([y_old, np.log(self.mu)]))
                X, Y = X[-(self.warm_m + 1):], Y[-(self.warm_m + 1):]
                if len(X) >= 2:
                    dU = np.diff(np.array(X), axis=0).T
                    dY = np.diff(np.array(Y), axis=0).T
                    step = np.concatenate([self.y, np.log(self.mu)]) - Y[-1]
                    coef, *_ = np.linalg.lstsq(dY, step, rcond=1e-8)
                    du = dU @ coef
                    # trust region: never move more than the last outer displacement
                    lim = np.linalg.norm(dU[:, -1])
                    if np.linalg.norm(du) > lim and np.linalg.norm(du) > 0:
                        du *= lim / np.linalg.norm(du)
                    self._unpack(u + du)

        self.tf = time.perf_counter() - t0
        self.success = success
        self.outer = k
        self._log(f"{'ALBCD converged' if success else 'ALBCD failed'} ({self.tf:.2f} s)")
        return self
