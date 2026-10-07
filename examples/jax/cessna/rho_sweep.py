"""Solve the Cessna ALBCD problem in cessna.py for a sweep of penalty growth factors rho.

Writes the convergence history of each solve to cessna_rho_sweep.npz, for fig_rho_sweep.py.
"""

import os
import numpy as np

from cessna import HERE, make_albcd, design_error

rhos = np.linspace(1.05, 2.0, 6)

# the histories have different lengths for each rho, so they are saved as object arrays
opt_history, feas_history, error = [], [], []
for rho in rhos:
    print(f"\n=== rho = {rho:.3f} ===")
    opt = make_albcd(rho)
    opt.solve()
    opt_history.append(np.array(opt.opt_history))
    feas_history.append(np.array(opt.feas_history))
    error.append(design_error(opt.x_history))
    print(f"rho = {rho:.3f}: {len(opt.feas_history)} sweeps, success {opt.success}, "
          f"relative error {error[-1][-1]:.2e}")


def ragged(arrays):
    out = np.empty(len(arrays), dtype=object)
    out[:] = arrays
    return out


# opt_history and feas_history have one entry per sweep; error has one per subproblem solve,
# and a leading entry for x0
np.savez(os.path.join(HERE, "cessna_rho_sweep.npz"), rho=rhos, opt_history=ragged(opt_history),
         feas_history=ragged(feas_history), error=ragged(error))
