"""Solve the Cessna ALBCD problem in cessna.py for a sweep of proximal coefficients tau.

Every subproblem adds the proximal term tau/2 ||x_i - x_i^{k-1}||^2 of the ALBCD paper (Eq. 25) to
its augmented Lagrangian, measured in the scaled variables SLSQP sees (see solve_slsqp in cessna.py).
tau = 0 is the solve without it. The optimality residual is still that of the augmented Lagrangian
without the proximal term. Writes cessna_prox_sweep.npz, for fig_prox_sweep.py.
"""

import os
import numpy as np

from cessna import HERE, make_albcd, design_error

taus = np.array([0.0, 1e-3, 1e-2, 1e-1, 1.0])

# the histories have different lengths for each tau, so they are saved as object arrays
opt_history, feas_history, error, success = [], [], [], []
for tau in taus:
    print(f"\n=== tau = {tau:g} ===")
    opt = make_albcd(tau=tau)
    opt.solve()
    opt_history.append(np.array(opt.opt_history))
    feas_history.append(np.array(opt.feas_history))
    error.append(design_error(opt.x_history))
    success.append(opt.success)
    print(f"tau = {tau:g}: {len(opt.feas_history)} sweeps, {opt.tf:.1f} s, success {opt.success}, "
          f"relative error {error[-1][-1]:.2e}")


def ragged(arrays):
    out = np.empty(len(arrays), dtype=object)
    out[:] = arrays
    return out


# opt_history and feas_history have one entry per sweep; error has one per subproblem solve,
# and a leading entry for x0
np.savez(os.path.join(HERE, "cessna_prox_sweep.npz"), tau=taus, opt_history=ragged(opt_history),
         feas_history=ragged(feas_history), error=ragged(error), success=np.array(success))
