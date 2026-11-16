"""ALBCD followed by a quasi-Newton polish: Anderson-accelerated sweeps.

Once ALBCD stops, the penalty mu is frozen, and one sweep followed by the first-order multiplier update
is treated as a fixed-point map s <- G(s) of the state s = (x, shared data, y). A fixed point is a KKT
point: phi = 0 leaves y unchanged, and an unchanged x means every block is stationary at that y. G is
evaluated only through the block solves. Anderson acceleration (Walker & Ni, 2011) extrapolates it from
the last `memory` residuals f = G(s) - s,

    gamma = argmin || f_k - dF gamma ||,    s_{k+1} = G(s_k) - dG gamma,

where the columns of dF and dG are differences of successive residuals and values of G. This is a
multisecant quasi-Newton method on s - G(s) = 0 (IQN-ILS in partitioned fluid-structure interaction),
with the Jacobian of the coupled problem estimated from sweeps instead of block derivatives.
memory = 0 gives the plain fixed-point iteration.

Inexact block solves make G noisy, and a step extrapolated from noise can undo several sweeps of
progress, so the polish returns the best iterate it has seen, measured by
max(optimality / polish_opt_tol, feasibility / polish_feas_tol). The blocks' own caches, such as SLSQP
multipliers, stay those of their latest solve.
"""

import numpy as np
from .albcd import ALBCD


class PolishedALBCD(ALBCD):

    def __init__(self, *args,
                 polish_opt_tol: float = 1e-6,
                 polish_feas_tol: float = 1e-6,
                 max_polish_iter: int = 50,
                 memory: int = 10,
                 x_scale=1.0,
                 data_scale=None,
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.polish_opt_tol = polish_opt_tol
        self.polish_feas_tol = polish_feas_tol
        self.max_polish_iter = max_polish_iter
        self.memory = memory
        self.x_scale = np.broadcast_to(np.asarray(x_scale, dtype=float), self.x.shape)
        self.data_scale = {} if data_scale is None else dict(data_scale)  # data entries in s, and their scales

    def _pack(self):
        data = [np.ravel(self.data[k]) * w for k, w in self.data_scale.items()]
        return np.concatenate([self.x * self.x_scale, *data, self.y])

    def _unpack(self, s):
        i = self.x.size
        self.x = s[:i] / self.x_scale
        for k, w in self.data_scale.items():
            n = np.size(self.data[k])
            self.data[k] = (s[i:i + n] / w).reshape(np.shape(self.data[k]))
            i += n
        self.y = s[i:].copy()

    def _sweep(self):
        """One sweep over the blocks; returns the optimality and feasibility at its result."""
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
        self.history["optimality"].append(optimality)
        self.history["feasibility"].append(feas)
        return optimality, feas

    def polish(self) -> "PolishedALBCD":

        self.history["polish_start"] = len(self.history["optimality"])
        self.success = False
        s, F, G = self._pack(), [], []
        best = (np.inf,)

        for k in range(self.max_polish_iter):
            self._unpack(s)
            optimality, feas = self._sweep()
            self._log(f"polish {k + 1}: optimality {optimality:.3e}, feasibility {feas:.3e}")
            merit = max(optimality / self.polish_opt_tol, feas / self.polish_feas_tol)
            if merit < best[0]:
                best = (merit, self.x.copy(), dict(self.data), self.y.copy(), self.phi.copy(), k + 1)
            if merit <= 1.0:
                self.success = True
                break

            self.y = self._update_y(self.y, self.mu, self.phi)
            g = self._pack()
            F, G = (F + [g - s])[-(self.memory + 1):], (G + [g])[-(self.memory + 1):]
            s = g
            if len(F) > 1:
                gamma = np.linalg.lstsq(np.diff(F, axis=0).T, F[-1], rcond=None)[0]
                s = g - np.diff(G, axis=0).T @ gamma

        _, self.x, self.data, self.y, self.phi, self.history["polish_best"] = best
        self._log(f"polish {'converged' if self.success else 'failed'}, best iterate from polish sweep {best[-1]}")
        return self

    def solve(self) -> "PolishedALBCD":
        super().solve()
        return self.polish()
