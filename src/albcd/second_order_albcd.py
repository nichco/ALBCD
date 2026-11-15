"""ALBCD with a second-order multiplier update that needs nothing from the blocks but their solves.

ALBCD updates the multipliers with the first-order rule v = y + mu * phi. If x minimizes the
augmented Lagrangian L_mu(., y), then x also solves the perturbed problem

    min f(x)  s.t.  phi(x) = u,        u = phi(x),

with multipliers v, so v = -grad p(u), where p(u) is the optimal value of that problem. Every
outer iteration therefore gives one sample (phi, v) of a fixed function v(phi) that does not
depend on mu, and the multipliers of the original problem are y* = v(0). The first-order update
uses the newest sample as its estimate of v(0). The second-order update extrapolates to
phi = 0 with the secant model through the last few samples,

    y  <-  v + delta,     delta = -dV gamma,     gamma = argmin || phi - dPhi gamma ||,

where the columns of dPhi and dV are the differences of successive samples. For one constraint
this is the secant method on phi(y) = 0; in general it is Newton's method on the dual with the
dual Hessian replaced by a multisecant estimate (Bertsekas' second-order multiplier iteration,
with nothing to differentiate). Directions of phi the samples have not explored get delta = 0,
the first-order update.

The relation v = v(phi) holds only if x minimizes L_mu(., y). ALBCD stops its sweeps at a loose
tolerance while the iterate is infeasible, and phi is then no dual gradient. So:

    gate   A sample is recorded, and a second-order step taken, only if the inner loop ended
           with optimality <= opt_tol * max(1, gate * feas / feas_tol). The default 0 asks for
           the final tolerance opt_tol; kappa (ALBCD's own inner tolerance) is gate's upper end.
    cap    The step is taken only if ||delta|| <= cap * ||mu * phi||. A much larger correction
           means the samples say the first-order update is barely converging, which on the
           examples was noise far more often than fact.
    undo   A second-order step should reduce the violation. If the next inner loop ends with a
           larger one, x, data, phi, y and mu return to what the first-order update would have
           left, and the next attempts are postponed (1, 2, 4, ... outer iterations).

Everything else, including the constructor arguments, is ALBCD's.
"""

import numpy as np
from .albcd import ALBCD


class SecondOrderALBCD(ALBCD):

    def __init__(self, *args,
                 memory: int = 5,
                 gate: float = 0.0,
                 cap: float = 2.0,
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.memory = memory    # secant pairs kept
        self.gate = gate
        self.cap = cap

        self._samples = []      # (phi, v) of the latest outer iterations that passed the gate
        self._saved = None      # state to return to if the latest second-order step fails
        self._undone = False
        self._wait = 0          # outer iterations left before the next second-order step
        self._backoff = 0

    def _update_mu(self, mu, phi_new, phi_old):
        if self._undone:        # back at an iteration whose penalty update has been made
            self._undone = False
            return mu
        return super()._update_mu(mu, phi_new, phi_old)

    def _update_y(self, y, mu, phi):

        v = y + mu * phi        # first-order update
        feas = np.max(np.abs(phi))

        # did the latest second-order step reduce the violation?
        if self._saved is not None:
            saved, self._saved = self._saved, None
            if not feas <= max(saved["feas"], self.feas_tol):
                self.x, self.data, self.phi = saved["x"], saved["data"], saved["phi"]
                self._backoff = max(1, 2 * self._backoff)
                self._wait = self._backoff
                self._undone = True
                self.history["update"].append("undone")
                return saved["y"]
            self._backoff = 0

        delta = np.zeros_like(v)
        optimality = self.history["optimality"][-1]
        if optimality <= self.opt_tol * max(1.0, self.gate * feas / self.feas_tol):
            self._samples = (self._samples + [(phi.copy(), v.copy())])[-(self.memory + 1):]
            if self._wait > 0:
                self._wait -= 1
            elif len(self._samples) > 1:
                dPhi = np.diff([p for p, _ in self._samples], axis=0).T
                dV = np.diff([w for _, w in self._samples], axis=0).T
                delta = -dV @ np.linalg.lstsq(dPhi, phi, rcond=None)[0]
                if not (np.all(np.isfinite(delta)) and np.linalg.norm(delta) <= self.cap * np.linalg.norm(mu * phi)):
                    delta = np.zeros_like(v)

        y_first = np.clip(v, -self.max_y, self.max_y)
        second_order = bool(np.any(delta != 0))
        if second_order:
            self._saved = dict(x=self.x.copy(), data=dict(self.data), phi=phi.copy(), y=y_first, feas=feas)
        self.history["update"].append("second_order" if second_order else "first_order")

        return np.clip(v + delta, -self.max_y, self.max_y)
