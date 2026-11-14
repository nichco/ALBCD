import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
import toys
from albcd2x import ALBCD2X
for pname in sys.argv[1].split(","):
    prob = toys.PROBLEMS[pname]()
    opt = ALBCD2X(prob.subproblems(), prob.x0, mu0=np.ones(prob.m), max_mu=1e4, rho=1.5, tau=0.5, feas_tol=1e-6,
                  opt_tol=1e-6, max_outer_iter=200, max_inner_iter=100, verbose=False)
    per, mus, feas = [], [], []
    orig = opt._block_coordinate_descent
    def wrap(tol):
        s0 = opt.n_sweeps; r = orig(tol); per.append(opt.n_sweeps - s0); mus.append(opt.mu.max()); feas.append(np.abs(opt.phi).max()); return r
    opt._block_coordinate_descent = wrap
    opt.solve()
    print(pname, "sweeps/outer:", per)
    print("   max mu:", np.round(mus, 1).tolist())
    print("   feas:", ["%.0e" % f for f in feas])
    # contraction factor of the inner iteration
    o = np.array(opt.history["optimality"]); print("   opt ratios (last 15):", np.round(o[1:]/o[:-1], 2)[-15:].tolist())
