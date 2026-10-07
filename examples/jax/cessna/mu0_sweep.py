"""Solve the Cessna ALBCD problem in cessna.py for a sweep of initial penalty parameters mu0.

Writes the convergence history of each solve to cessna_mu0_sweep.npz, for fig_mu0_sweep.py.
"""

import os
import numpy as np

from cessna import HERE, make_albcd, design_error

mu0s = np.geomspace(1.0, 100.0, 6)

# the histories have different lengths for each mu0, so they are saved as object arrays
opt_history, feas_history, error, success = [], [], [], []
for mu0 in mu0s:
    print(f"\n=== mu0 = {mu0:.3g} ===")
    opt = make_albcd(mu0=mu0)
    opt.solve()
    opt_history.append(np.array(opt.opt_history))
    feas_history.append(np.array(opt.feas_history))
    error.append(design_error(opt.x_history))
    success.append(opt.success)
    print(f"mu0 = {mu0:.3g}: {len(opt.feas_history)} sweeps, success {opt.success}, "
          f"relative error {error[-1][-1]:.2e}")


def ragged(arrays):
    out = np.empty(len(arrays), dtype=object)
    out[:] = arrays
    return out


# opt_history and feas_history have one entry per sweep; error has one per subproblem solve,
# and a leading entry for x0
np.savez(os.path.join(HERE, "cessna_mu0_sweep.npz"), mu0=mu0s, opt_history=ragged(opt_history),
         feas_history=ragged(feas_history), error=ragged(error), success=np.array(success))
