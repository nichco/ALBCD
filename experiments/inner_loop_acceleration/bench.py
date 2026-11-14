import sys, time, json
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import toys
from albcd2x import ALBCD2X

CONFIGS = {
    "baseline":      dict(),
    "sor1.3":        dict(accel='sor', omega=1.3),
    "sor1.6":        dict(accel='sor', omega=1.6),
    "aitken":        dict(accel='aitken'),
    "anderson3":     dict(accel='anderson', m=3),
    "anderson8":     dict(accel='anderson', m=8),
    "warm":          dict(warm='secant'),
    "relative":      dict(inner_rule='relative', sigma=0.5),
    "AA5+warm":      dict(accel='anderson', m=5, warm='secant'),
    "AA5+warm+rel":  dict(accel='anderson', m=5, warm='secant', inner_rule='relative'),
    "AAreuse5":      dict(accel='anderson', m=5, reuse=True),
    "AAreuse10":     dict(accel='anderson', m=10, reuse=True),
    "eta0.1":        dict(block_eta=0.1),
    "AAreuse10+eta": dict(accel='anderson', m=10, reuse=True, block_eta=0.1),
    "sor1.3+eta":    dict(accel='sor', omega=1.3, block_eta=0.1),
    "sorA":          dict(accel='sor_adaptive', omega=1.0),
    "sorA+eta":      dict(accel='sor_adaptive', omega=1.0, block_eta=0.1),
}

def run(pname, cname, solver_kw):
    prob = toys.PROBLEMS[pname]()
    toys.COUNTER["nfev"] = 0
    kw = dict(mu0=np.ones(prob.m), max_mu=1e4, rho=1.5, tau=0.5, feas_tol=1e-6, opt_tol=1e-6,
              max_outer_iter=200, max_inner_iter=100, verbose=False)
    solver_kw = dict(solver_kw); sc = solver_kw.pop("mu0_scale", 1.0); kw["mu0"] = kw["mu0"] * sc
    kw.update(solver_kw)
    opt = ALBCD2X(prob.subproblems(), prob.x0, **kw, **CONFIGS[cname])
    t = time.perf_counter(); opt.solve(); t = time.perf_counter() - t
    import jax.numpy as jnp
    fval = float(prob.f(jnp.asarray(opt.x)))
    return dict(problem=pname, config=cname, ok=opt.success, outer=opt.outer, sweeps=opt.n_sweeps,
                solves=opt.n_solves, nfev=toys.COUNTER["nfev"], grads=opt.n_grads, rej=opt.n_reject,
                f=fval, feas=float(np.max(np.abs(opt.phi))), time=t)

if __name__ == "__main__":
    probs = sys.argv[1].split(",") if len(sys.argv) > 1 else list(toys.PROBLEMS)
    confs = sys.argv[2].split(",") if len(sys.argv) > 2 else list(CONFIGS)
    extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    rows = []
    for p in probs:
        base = None
        for c in confs:
            r = run(p, c, extra)
            if base is None: base = r
            rows.append(r)
            print(f"{p:16s} {c:14s} ok={int(r['ok'])} outer={r['outer']:3d} sweeps={r['sweeps']:4d} "
                  f"solves={r['solves']:4d} nfev={r['nfev']:6d} ({r['nfev']/base['nfev']:.2f}x) "
                  f"rej={r['rej']:2d} f={r['f']:.8f} feas={r['feas']:.1e} t={r['time']:.1f}s", flush=True)
