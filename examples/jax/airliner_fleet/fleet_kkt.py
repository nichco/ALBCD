"""The fleet of airliner_fleet.py and monolithic.py, and a KKT residual of the monolithic problem
that measures the OpenSQP and the ALBCD iterates the same way (monolithic_opensqp.py and
albcd_timing.py).

The measure is the one of examples/jax/cessna/monolithic_opensqp.py. In the scaled variables
u = x_scaler * x, with the throttle and CL limits and the variable bounds written as g <= 0,
the flight-time equalities h_i = 0 and the budget b = 0, the multipliers are the least-squares
estimate that minimizes

    ||grad f + G^T lam_g + H^T mu + lam_b grad b||^2 + ||lam_g * g||^2   subject to lam_g >= 0,

the stationarity and complementarity residuals together. The optimality is the larger of the
max-norm stationarity and complementarity residuals at those multipliers, and the feasibility
is the max-norm constraint violation.

That least-squares problem has about 9,000 multipliers at N = 8, but every multiplier except the
budget's lam_b belongs to one flight. For a fixed lam_b it splits into one nonnegative least
squares (NNLS) problem per flight, and its optimal value is a convex, piecewise quadratic function
of lam_b, whose first and second derivatives follow from the flights' NNLS solutions. A
safeguarded Newton iteration on lam_b therefore finds the exact joint solution in a few steps.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import qr
from scipy.optimize import nnls
from scipy.stats import qmc

from models import nvar, cl, cu, xl, xu, x_scaler, setup_flight, flight_outputs

TAU = nvar - 1  # index of the block-time allocation in each flight's variables
EQ = cl == cu                       # flight time = tau
LO = np.isfinite(cl) & ~EQ          # throttle >= 0
UP = np.isfinite(cu) & ~EQ          # throttle <= 1, CL <= CL_limit


def make_fleet(N):
    """Ranges (m), takeoff masses (kg), flight data and block-time budget (ks) of the N flights,
    as in airliner_fleet.py and monolithic.py."""
    ranges, masses = qmc.scale(qmc.LatinHypercube(d=2, seed=0).random(N),
                               [2000e3, 22000.0], [5000e3, 28000.0]).T
    flights = [setup_flight(rf, m0) for rf, m0 in zip(ranges, masses)]
    T_budget = 0.9 * np.sum(0.276 + 5.468e-6 * ranges)
    return ranges, masses, flights, T_budget


def _flight_lsq(out, jac, v):
    """One flight's least-squares matrix M (stationarity rows over complementarity rows), its
    objective gradient, inequalities g <= 0 and flight-time residual, in the scaled variables."""
    c, J = out[:-1], jac / x_scaler  # derivatives with respect to the scaled variables
    Jc, grad_f = J[:-1], J[-1]
    u = x_scaler * v
    g = np.concatenate([cl[LO] - c[LO], c[UP] - cu[UP], x_scaler * xl - u, u - x_scaler * xu])
    G = np.vstack([-Jc[LO], Jc[UP], -np.eye(nvar), np.eye(nvar)])
    H = Jc[EQ]
    # columns: lam_g >= 0, then the flight-time multiplier mu split into mu+ >= 0 and mu- >= 0
    M = np.block([[G.T, H.T, -H.T],
                  [np.diag(g), np.zeros((len(g), 2))]])
    return M, grad_f, g, c[EQ]


def _solve_flights(mats, lam_b):
    """Each flight's NNLS at the budget multiplier lam_b, with the derivatives of the total
    squared residual V(lam_b): V' = 2 sum r_tau and V'' = 2 sum (1 - |P_F e_tau|^2), where r_tau
    is a flight's stationarity residual in its tau row and P_F projects onto the columns of its
    M that are free at its NNLS solution."""
    results, d1, d2 = [], 0.0, 0.0
    for M, grad_f, g, _ in mats:
        rhs = np.concatenate([-grad_f, np.zeros(len(g))])
        rhs[TAU] -= lam_b  # the budget's gradient is 1 in each tau (x_scaler = 1 for tau)
        lam, _ = nnls(M, rhs, maxiter=50 * M.shape[1])
        r = M @ lam - rhs
        free = lam > 0
        free[-2] |= not (free[-2] or free[-1])  # mu is a free variable, even where it is 0
        Q, R, _ = qr(M[:, free], mode="economic", pivoting=True)
        rank = int(np.sum(np.abs(np.diag(R)) > 1e-12 * max(1.0, abs(R[0, 0]))))
        d1 += 2 * r[TAU]
        d2 += 2 * (1.0 - np.sum(Q[TAU, :rank] ** 2))
        results.append((lam, r))
    return results, d1, d2


def kkt_point(outs, jacs, x, T_budget, lam_b0=0.0):
    """Optimality, feasibility and budget multiplier at the fleet design x, from each flight's
    outputs (local constraints, fuel) and their Jacobian."""
    N = len(outs)
    mats = [_flight_lsq(outs[i], jacs[i], x[i * nvar:(i + 1) * nvar]) for i in range(N)]

    # safeguarded Newton on lam_b: V is convex and quadratic between changes of the NNLS active
    # sets, so a Newton step from inside the final piece lands on the minimizer
    lam_b, lo, hi = lam_b0, -np.inf, np.inf
    for _ in range(60):
        results, d1, d2 = _solve_flights(mats, lam_b)
        if d1 == 0.0:
            break
        if d1 > 0:
            hi = lam_b
        else:
            lo = lam_b
        new = lam_b - d1 / d2 if d2 > 1e-12 else lam_b - np.sign(d1) * max(1.0, abs(lam_b))
        if not lo < new < hi:
            new = 0.5 * (lo + hi)
        if abs(new - lam_b) <= 1e-12 * (1 + abs(lam_b)):
            break
        lam_b = new

    n_g = len(mats[0][2])
    stationarity = max(np.max(np.abs(r[:nvar])) for _, r in results)
    complementarity = max(np.max(np.abs(lam[:n_g] * m[2])) for (lam, _), m in zip(results, mats))
    budget = np.sum(x[TAU::nvar]) - T_budget
    feasibility = max(abs(budget), max(np.max(np.abs(m[3])) for m in mats),
                      max(np.max(m[2]) for m in mats), 0.0)
    return max(stationarity, complementarity), feasibility, lam_b


def _kkt_chunk(args):
    """kkt_point() at each design in xs, warm-starting the budget multiplier from the previous one."""
    xs, N = args
    _, _, flights, T_budget = make_fleet(N)
    fleet = jax.tree.map(lambda *arrays: jnp.stack(arrays), *flights)
    outputs_fn = jax.jit(jax.vmap(flight_outputs))
    jac_fn = jax.jit(jax.vmap(jax.jacfwd(flight_outputs)))
    out, lam_b = [], 0.0
    for x in xs:
        v = jnp.asarray(np.reshape(x, (N, nvar)))
        res = kkt_point(np.asarray(outputs_fn(v, fleet)), np.asarray(jac_fn(v, fleet)), x, T_budget, lam_b)
        lam_b = res[2]
        out.append(res)
    return out


def load_previous_kkt(path):
    """The designs and KKT measure saved in path by an earlier run, or None. Load this before the
    new run overwrites path, and pass it to kkt_history_reusing()."""
    if not os.path.exists(path):
        return None
    old = np.load(path)
    if "optimality" not in old:
        return None
    return {k: np.array(old[k]) for k in ("x", "optimality", "feasibility", "budget_multiplier")}


def kkt_history_reusing(xs, N, previous):
    """kkt_history(), reusing the previous run's values for the leading designs it shares exactly
    with xs. OpenSQP and ALBCD are deterministic here, so a rerun to re-measure the wall time reuses
    every value, and an ALBCD run to a tighter tolerance reuses the iterates before it first
    differs from the looser run."""
    xs = np.asarray(xs)
    n = 0
    if previous is not None and previous["x"].shape[1:] == xs.shape[1:]:
        old_x = previous["x"]
        m = min(len(old_x), len(xs))
        same = np.all(old_x[:m] == xs[:m], axis=1)
        n = m if same.all() else int(np.argmin(same))
    print(f"KKT measure: reusing {n} of {len(xs)} designs from the previous run", flush=True)
    reused = [previous[k][:n] for k in ("optimality", "feasibility", "budget_multiplier")] if n else \
        [np.zeros(0)] * 3
    if n == len(xs):
        return tuple(reused)
    new = kkt_history(xs[n:], N)
    return tuple(np.concatenate([a, b]) for a, b in zip(reused, new))


def _lower_priority():
    """Run a worker below normal priority, so the post-processing yields to other work."""
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass


def kkt_history(xs, N, workers=None):
    """kkt_point() at every fleet design in xs, as three arrays (optimality, feasibility, budget
    multiplier). Contiguous chunks of xs run in parallel processes, below normal priority; this is
    post-processing, so it does not affect the recorded solve times."""
    import multiprocessing as mp
    xs = np.asarray(xs)
    workers = min(workers or max(1, (os.cpu_count() or 2) - 1), len(xs))
    chunks = [(c, N) for c in np.array_split(xs, workers)]
    with mp.get_context("spawn").Pool(workers, initializer=_lower_priority) as pool:
        out = [r for chunk in pool.map(_kkt_chunk, chunks) for r in chunk]
    return tuple(np.array(a) for a in zip(*out))
