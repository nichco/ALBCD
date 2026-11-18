"""Shared harness of experiments_airliner.py and experiments_ranger.py (Task 6).

run_config() solves one named configuration and returns a flat dict of arrays: the event history (§2.4) and the
distance ||x - x*|| after each event. For plain ALBCD runs, which keep no theta, a hook on the last block's solve
stores the iterate, multipliers and local multipliers after every sweep, and theta is evaluated after the timed
solve from those snapshots, with the same measure (qn_core.kkt_measure at lam = y + mu * phi).

Every configuration is stopped once its wall time exceeds ``time_limit`` (3x the albcd run): a hook on the block
solves raises TimeLimit, and the history up to that point is kept.
"""

import time
import numpy as np
from albcd import ALBCD

import qn_core as qc
from qn_albcd import QuasiNewtonALBCD


class TimeLimit(Exception):
    pass


def instrument(opt, time_limit, record_sweeps):
    """Wrap every block's solve: time it, enforce the time limit and, with record_sweeps, snapshot the state after
    the last block of each sweep. Returns the recorder dict."""
    rec = dict(t0=None, solve_times=[], snapshots=[])
    nb = len(opt.subproblems)
    for i, sp in enumerate(opt.subproblems):
        orig = sp.solve

        def solve(x, y, mu, data, outputs, _orig=orig, _i=i):
            if rec["t0"] is not None and time.perf_counter() - rec["t0"] > time_limit:
                raise TimeLimit()
            t = time.perf_counter()
            _orig(x, y, mu, data, outputs)
            rec["solve_times"].append(time.perf_counter() - t)
            if record_sweeps and _i == nb - 1:
                merged = dict(data)
                merged.update({k: v for k, v in outputs.items() if k not in ("x", "phi")})
                rec["snapshots"].append(dict(
                    t=time.perf_counter() - rec["t0"], x=np.array(outputs["x"], copy=True),
                    phi=np.array(outputs["phi"], dtype=float).reshape(np.shape(mu)), y=np.array(y, copy=True),
                    mu=np.array(mu, copy=True), data=merged,
                    kappas=[np.asarray(s.local_multipliers(), dtype=float).copy() for s in opt.subproblems],
                    t_par=float(np.sum(rec["solve_times"]))))
        sp.solve = solve
    return rec


def theta_at(opt, x, y, mu, phi, kappas, data):
    """theta (§1.6) at a post-sweep point with lam = y + mu * phi, the blocks' local multipliers and active sets."""
    lam = y + mu * phi
    lins, frees = [], []
    for sp, k in zip(opt.subproblems, kappas):
        lin = dict(sp.linearize(x, data))
        lin["v"] = np.array(x[sp.index], dtype=float)
        if len(k) != len(lin["h"]):
            k = np.zeros(len(lin["h"]))
        r = qc.stationarity(lin, lam, k)
        free, _, _ = qc.active_sets(lin, k, r, 1e-8, 1e-8, getattr(sp, "tight_tol", None))
        lins.append(lin)
        frees.append(free)
    kappas = [k if len(k) == len(l["h"]) else np.zeros(len(l["h"])) for k, l in zip(kappas, lins)]
    return qc.kkt_measure(lins, lam, kappas, frees, phi)


def run_config(name, build, compile_models, dist, time_limit=np.inf, log=print):
    """Build (build() -> solver), compile, solve and post-process one configuration."""
    opt = build()
    compile_models(opt)
    is_qn = isinstance(opt, QuasiNewtonALBCD)
    rec = instrument(opt, time_limit, record_sweeps=not is_qn)
    rec["t0"] = time.perf_counter()
    stopped = ""
    try:
        opt.solve()
    except TimeLimit:
        stopped = "time_limit"
        opt.tf = time.perf_counter() - rec["t0"]
    total = time.perf_counter() - rec["t0"]

    out = dict(name=name, solver="QN" if is_qn else "ALBCD", stopped=stopped, total_time=total)
    if is_qn:
        h = opt.history
        out.update(event=np.array(h["event"], dtype=str), theta=np.array(h["theta"], dtype=float),
                   time=np.array(h["time"]), t_parallel=np.array(h["t_parallel"]),
                   n_solves=np.array(h["n_solves"]), n_linearize=np.array(h["n_linearize"]),
                   n_pairs=np.array(h["n_pairs"]), cond_K=np.array(h["cond_K"]),
                   reason=np.array(h["reason"], dtype=str),
                   dist=np.array([dist(x) for x in h["x_event"]]),
                   mu_max=np.array([np.max(m) for m in h["mu_event"]]),
                   success=bool(opt.success), status=opt.status or stopped)
        theta_final = opt.theta if np.isfinite(opt.theta) else np.nan
    else:
        snaps = rec["snapshots"]
        thetas = np.array([theta_at(opt, s["x"], s["y"], s["mu"], s["phi"], s["kappas"], s["data"]) for s in snaps])
        nb = len(opt.subproblems)
        out.update(event=np.array(["sweep"] * len(snaps), dtype=str), theta=thetas,
                   time=np.array([s["t"] for s in snaps]), t_parallel=np.array([s["t_par"] for s in snaps]),
                   n_solves=nb * np.arange(1, len(snaps) + 1), n_linearize=np.zeros(len(snaps), int),
                   n_pairs=np.zeros(len(snaps), int), cond_K=np.full(len(snaps), np.nan),
                   reason=np.array([""] * len(snaps), dtype=str),
                   dist=np.array([dist(s["x"]) for s in snaps]),
                   mu_max=np.array([np.max(s["mu"]) for s in snaps]),
                   success=bool(opt.success), status=("converged" if opt.success else "failed") + (f" ({stopped})" if stopped else ""))
        theta_final = thetas[-1] if len(thetas) else np.nan
        out["albcd_time"] = opt.tf
    out["theta_final"] = theta_final
    out["dist_final"] = out["dist"][-1] if len(out["dist"]) else np.nan
    out["x_final"] = np.array(opt.x, copy=True)
    out["y_final"] = np.array(opt.y, copy=True)
    log(f"[{name}] {out['status']}: theta {theta_final:.3e}, ||x - x*|| {out['dist_final']:.3e}, "
        f"{np.sum(out['event'] == 'sweep')} sweeps, {total:.1f} s")
    return out


def time_to(out, level):
    """(wall time, critical-path time, sweeps) at the first event with theta <= level, or NaNs."""
    th = out["theta"]
    idx = np.flatnonzero(np.isfinite(th) & (th <= level))
    if not len(idx):
        return np.nan, np.nan, np.nan
    k = idx[0]
    return out["time"][k], out["t_parallel"][k], int(np.sum(out["event"][:k + 1] == "sweep"))


def summary_table(results, title):
    head = (f"{'configuration':14s} {'success':>7s} {'theta':>9s} {'|x-x*|':>9s} {'sweeps':>6s} {'solves':>6s} "
            f"{'acc':>4s} {'rej':>4s} {'t(1e-4)':>8s} {'tp(1e-4)':>8s} {'sw(1e-4)':>8s} {'t(1e-8)':>8s} "
            f"{'tp(1e-8)':>8s} {'sw(1e-8)':>8s} {'total':>7s}  status")
    lines = [title, head, "-" * len(head)]
    for r in results:
        e = r["event"]
        t4, tp4, s4 = time_to(r, 1e-4)
        t8, tp8, s8 = time_to(r, 1e-8)
        f = lambda v, fmt: (format("-", ">" + fmt.split(".")[0]) if not np.isfinite(v) else format(v, fmt))
        lines.append(f"{r['name']:14s} {str(r['success']):>7s} {r['theta_final']:9.2e} {r['dist_final']:9.2e} "
                     f"{int(np.sum(e == 'sweep')):6d} {int(r['n_solves'][-1]) if len(e) else 0:6d} "
                     f"{int(np.sum(e == 'qn_accept')):4d} {int(np.sum(e == 'qn_reject')):4d} "
                     f"{f(t4, '8.1f')} {f(tp4, '8.1f')} {f(s4, '8.0f')} {f(t8, '8.1f')} {f(tp8, '8.1f')} {f(s8, '8.0f')} "
                     f"{r['total_time']:7.1f}  {r['status']}")
    lines.append("")
    lines.append("theta: final KKT measure (plan sec. 1.6); |x-x*|: scaled distance to the tight reference; t/tp/sw(level): wall "
                 "time, critical-path time and sweeps to the first event with theta <= level; total: wall time (s).")
    return "\n".join(lines)


def save_results(results, path, **extra):
    arrays = dict(extra)
    arrays.update(flatten(results))
    np.savez(path, **arrays)


def flatten(results, prefix=""):
    """{prefix + "names": [...], prefix + "<configuration>__<field>": array} for np.savez."""
    arrays = {prefix + "names": np.array([r["name"] for r in results], dtype=str)}
    for r in results:
        for key, value in r.items():
            if key != "name":
                arrays[f"{prefix}{r['name']}__{key}"] = np.asarray(value)
    return arrays


QN_CONFIGS = [
    ("v1", dict(mode="dual")),
    ("v2", dict(mode="primal_dual")),
    ("v3", dict(mode="adaptive")),
    ("v3_mu_lo", dict(mode="adaptive", mu_factor=0.1)),
    ("v3_mu_hi", dict(mode="adaptive", mu_factor=10.0)),
    ("v3_outer", dict(mode="adaptive", pairs="outer")),
    ("v3_harvest", dict(mode="adaptive", pairs="harvest")),
    ("v3_norefine", dict(mode="adaptive", refine=0)),
]


def configurations(mu0, names=None):
    """(name, cls, settings) of Task 6's table; the QN runs use kkt_tol = 1e-8 and max_outer_iter = 100."""
    configs = [("albcd", ALBCD, dict(verbose=False)),
               ("albcd_tight", ALBCD, dict(opt_tol=1e-9, feas_tol=1e-9, max_outer_iter=200, verbose=False))]
    for name, s in QN_CONFIGS:
        s = dict(s)
        factor = s.pop("mu_factor", None)
        if factor is not None:
            s["mu0"] = np.asarray(mu0) * factor
        s.update(kkt_tol=1e-8, max_outer_iter=100, verbose=False)
        configs.append((name, QuasiNewtonALBCD, s))
    if names is not None:
        configs = [c for c in configs if c[0] in names]
    return configs
