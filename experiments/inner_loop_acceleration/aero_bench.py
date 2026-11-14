"""ALBCD2 inner-loop accelerations on the real aerostructural example (JAX VLM + beam, SLSQP blocks)."""
import os, sys, time, json
import numpy as np
AERO = r"C:\Users\Nicholas\Desktop\code\ALBCD\examples\jax\aerostruct"
sys.path.insert(0, AERO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(AERO)
import warnings; warnings.filterwarnings("ignore")
import jax.numpy as jnp
import aerostruct_albcd3 as A
from albcd2x import ALBCD2X

COUNT = {"nfev": 0}
_orig = A.solve_slsqp
def counting_solve(fns, args, v0, x_scaler, c_scaler, ftol, xl=None, xu=None, _data=[None]):
    obj = fns.obj
    grad = fns.grad
    def cobj(*a):
        COUNT["nfev"] += 1
        return obj(*a)
    def cgrad(*a):
        COUNT["ngev"] = COUNT.get("ngev", 0) + 1
        return grad(*a)
    return _orig(fns._replace(obj=cobj, grad=cgrad), args, v0, x_scaler, c_scaler, ftol, xl, xu)
A.solve_slsqp = counting_solve


def eff_ftol(sp, data):
    # adaptive block accuracy: SLSQP's ftol is a tolerance on the objective change, ~ g^2 / H,
    # so map the requested gradient tolerance t to ftol ~ c * t^2, never tighter than the default
    t = data.get("_tol")
    if t is None:
        return sp.FTOL
    # never looser than the default once the inner tolerance reaches opt_tol (t = 0.1 * 1e-3)
    return float(np.clip(sp.FTOL * max(1.0, t / 1e-4) ** 2, sp.FTOL, FTOL_CAP))


class Aero2(A.AeroSubproblem):
    def solve(self, x, y, mu, data, outputs):
        f0 = self.FTOL; self.FTOL = eff_ftol(self, data)
        try: super().solve(x, y, mu, data, outputs)
        finally: self.FTOL = f0
    def gradient(self, x, y, mu, data):
        v, other, yy, mm, w = (jnp.asarray(a) for a in (self.decompose(x), self.other(x), y, mu, data["weight"]))
        return np.array(self._fns.grad(v, other, yy, mm, w)) - self._jac_con.T @ self._multipliers


class Struct2(A.StructSubproblem):
    def solve(self, x, y, mu, data, outputs):
        f0 = self.FTOL; self.FTOL = eff_ftol(self, data)
        try: super().solve(x, y, mu, data, outputs)
        finally: self.FTOL = f0
    def gradient(self, x, y, mu, data):
        v, other, yy, mm, al = (jnp.asarray(a) for a in (self.decompose(x), self.other(x), y, mu, data["aero_loads"]))
        resid = np.array(self._fns.grad(v, other, yy, mm, al)) - self._jac_con.T @ self._multipliers
        v = np.array(v)
        viol = resid.copy()
        viol = np.where(v <= self.XL + 1e-6, np.minimum(0.0, resid), viol)
        viol = np.where(v >= self.XU - 1e-6, np.maximum(0.0, resid), viol)
        return viol


ETA_F = 1.0
FTOL_CAP = float(os.environ.get("FTOL_CAP", "1e-7"))
CONFIGS = {
    "baseline": dict(),
    "AAreuse10": dict(accel='anderson', m=10, reuse=True),
    "sor1.3": dict(accel='sor', omega=1.3),
    "aitken": dict(accel='aitken'),
    "eta": dict(block_eta=0.1),
    "AAreuse10+eta": dict(accel='anderson', m=10, reuse=True, block_eta=0.1),
    "sor1.3+eta": dict(accel='sor', omega=1.3, block_eta=0.1),
    "relative": dict(inner_rule='relative', sigma=0.5),
    "AA5": dict(accel='anderson', m=5),
    "AAreuse3": dict(accel='anderson', m=3, reuse=True),
    "AAreuse10_xonly": dict(accel='anderson', m=10, reuse=True, xonly=True),
}

if __name__ == "__main__":
    confs = sys.argv[1].split(",")
    _, aero_loads_init, _ = A.aero_model(A.twist_cp0)
    _, _, weight_init = A.structures_model(aero_loads_init, A.thickness_cp0)
    x0 = np.concatenate([A.twist_cp0, [float(weight_init)], A.thickness_cp0, np.array(aero_loads_init)])
    sol = np.load('monolithic_solution.npz')
    x_star = np.concatenate([sol['twist_cp'], sol['thickness_cp']])
    for c in confs:
        COUNT["nfev"] = 0; COUNT["ngev"] = 0
        subs = [Aero2(A.AERO_INDEX), Struct2(A.STRUCT_INDEX)]
        # warm up JIT so timings compare solver work, not compilation
        subs[0]._fns.obj(jnp.asarray(x0[A.AERO_INDEX]), jnp.asarray(x0[A.STRUCT_INDEX]), jnp.zeros(A.N_CON), jnp.ones(A.N_CON), jnp.asarray(float(weight_init)))
        opt = ALBCD2X(subs, x0, mu0=np.ones(A.N_CON), data0={"weight": float(weight_init)},
                      max_mu=1e6, rho=1.2, tau=0.5, feas_tol=3e-5, opt_tol=1e-3,
                      max_outer_iter=60, max_inner_iter=10, verbose=False, **CONFIGS[c])
        t = time.perf_counter(); opt.solve(); t = time.perf_counter() - t
        d = np.concatenate([opt.x[A.TWIST_SLICE], opt.x[A.THICKNESS_SLICE]])
        err = np.linalg.norm(d - x_star) / np.linalg.norm(x_star)
        print(f"aerostruct {c:14s} ok={int(opt.success)} outer={opt.outer:3d} sweeps={opt.n_sweeps:4d} solves={opt.n_solves:4d} "
              f"nfev={COUNT['nfev']:5d} ngev={COUNT['ngev']:5d} rej={opt.n_reject} CD={opt.data['CD']:.7f} relerr={err:.1e} "
              f"feas={np.abs(opt.phi).max():.1e} t={t:.1f}s", flush=True)
