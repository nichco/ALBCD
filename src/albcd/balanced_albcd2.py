"""ALBCD2 with an ADMM-like residual-balancing penalty update.

Feasibility ||phi|| and the stationarity residual r (the optimality measure of ALBCD2, the
gradient of the augmented Lagrangian, i.e. the KKT residual at the multiplier estimate
y + mu * phi) play the roles of ADMM's primal and dual residuals. Each is scaled by its own
stopping tolerance, so that 1 means done:

    p = ||phi||_inf / feas_tol,    d = r / opt_tol

At every multiplier update:

    d > beta * p   stationarity lags, so the penalty is likely too large:
                   no increases (V), and mu <- max(mu / rho, mu_floor) for all constraints (D)
    otherwise      the increase rule of ALBCD2, per constraint

The balanced inner-loop tolerance (T), r <= opt_tol * max(1, kappa * ||phi||_inf / feas_tol),
is already part of ALBCD2; kappa = 0 turns it off.

Safeguards: mu_floor defaults to mu0, so decreases only undo earlier increases; a decrease
needs `hold` outer iterations without a penalty change and is followed by `hold` outer
iterations without one; at most `max_decrease` decreases per run.

beta = inf recovers ALBCD2 exactly.
"""

import numpy as np
from .albcd2 import ALBCD2


class BalancedALBCD2(ALBCD2):

    def __init__(self, *args,
                 beta: float = 10.0,
                 decrease: bool = True,
                 mu_floor=None,
                 hold: int = 2,
                 max_decrease: int = 10,
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.beta = beta
        self.decrease = decrease
        self.mu_floor = self.mu.copy() if mu_floor is None else np.broadcast_to(np.asarray(mu_floor, dtype=float), self.mu.shape)
        self.hold = hold
        self.max_decrease = max_decrease

        self.k = 0 # multiplier updates so far
        self.last_change = -np.inf # update at which mu last changed
        self.last_decrease = -np.inf # update at which mu was last decreased
        self.n_increase = 0
        self.n_decrease = 0

    def _update_mu(self, mu, phi_new, phi_old):

        self.k += 1
        p = np.max(np.abs(phi_new), initial=0.0) / self.feas_tol
        d = self.history["optimality"][-1] / self.opt_tol
        self.history["p_hat"].append(p)
        self.history["d_hat"].append(d)

        if d > self.beta * p: # stationarity lags: veto increases (V) and possibly decrease (D)
            mu_new = mu
            if (self.decrease and self.n_decrease < self.max_decrease
                    and self.k - self.last_change > self.hold):
                mu_new = np.maximum(mu / self.rho, self.mu_floor)
                if np.any(mu_new < mu):
                    self.n_decrease += 1
                    self.last_decrease = self.k
                    self._log(f"penalty decreased (p_hat {p:.2e}, d_hat {d:.2e})")

        elif self.k - self.last_decrease > self.hold: # feasibility lags or balanced: rule of ALBCD2
            mu_new = super()._update_mu(mu, phi_new, phi_old)
            self.n_increase += int(np.any(mu_new > mu))

        else: mu_new = mu # hold after a decrease

        if np.any(mu_new != mu): self.last_change = self.k
        self.history["mu"].append(np.array(mu_new, dtype=float))

        return mu_new
