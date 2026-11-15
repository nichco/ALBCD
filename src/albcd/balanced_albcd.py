"""ALBCD with an ADMM-like residual-balancing penalty update.

Feasibility ||phi|| and the stationarity residual r (the optimality measure of ALBCD, the
gradient of the augmented Lagrangian, i.e. the KKT residual at the multiplier estimate
y + mu * phi) play the roles of ADMM's primal and dual residuals. Each is scaled by its own
stopping tolerance, so that 1 means done:

    p = ||phi||_inf / feas_tol,    d = r / opt_tol

At every multiplier update:

    d > beta * p   stationarity lags, so the penalty is likely too large:
                   mu <- max(mu / rho, mu_floor) for all constraints, and no increases
    otherwise      the increase rule of ALBCD, per constraint

mu_floor defaults to mu0, so decreases only undo earlier increases. The balanced inner-loop
tolerance, r <= opt_tol * max(1, kappa * ||phi||_inf / feas_tol), is part of ALBCD.
beta = inf recovers ALBCD exactly.
"""

import numpy as np
from .albcd import ALBCD


class BalancedALBCD(ALBCD):

    def __init__(self, *args, beta: float = 10.0, mu_floor=None, **kwargs):

        super().__init__(*args, **kwargs)
        self.beta = beta
        self.mu_floor = self.mu.copy() if mu_floor is None else np.broadcast_to(np.asarray(mu_floor, dtype=float), self.mu.shape)

    def _update_mu(self, mu, phi_new, phi_old):

        p = np.max(np.abs(phi_new), initial=0.0) / self.feas_tol
        d = self.history["optimality"][-1] / self.opt_tol
        self.history["p_hat"].append(p)
        self.history["d_hat"].append(d)

        if d > self.beta * p: # stationarity lags: the penalty is likely too large
            return np.maximum(mu / self.rho, self.mu_floor)

        return super()._update_mu(mu, phi_new, phi_old)
