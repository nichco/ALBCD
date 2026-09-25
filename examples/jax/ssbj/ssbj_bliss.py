"""Sobieski's supersonic business jet (SSBJ) problem with bi-level integrated system synthesis (BLISS).

BLISS (Sobieszczanski-Sobieski, Agte and Sandusky, NASA/TM-1998-208715), as described by
Martins and Lambe ("Multidisciplinary design optimization: a survey of architectures", AIAA
Journal, 2013). Each iteration linearizes the problem about the current design and takes a
step within move limits:

    1. The coupled analysis and its total derivatives, the solution of the global sensitivity
       equations, at the current design. models.analysis differentiates the converged
       couplings implicitly, which is the same linear solve.
    2. Each discipline solves a linear program over its local design variables x_i: minimize
       the linearized objective subject to its own linearized constraints, with the shared
       design variables x_0 fixed.
    3. The post-optimality sensitivities dx_i*/dx_0 of each discipline's solution, from the
       constraints and bounds active at it.
    4. The system solves a linear program over the shared design variables: minimize the
       linearized objective, including each discipline's response through dx_i*/dx_0,
       subject to every discipline's linearized constraints, including those responses.
    5. The step to the new design, x_0 + dx_0 and x_i* + dx_i*/dx_0 dx_0, is accepted if it
       reduces an L1 penalty merit function; otherwise the move limits are halved and steps
       2-4 repeat.

The linear programs are elastic: slack variables with a large penalty keep them feasible when
the linearized constraints can't all be met within the move limits. All the steps work in the
design variables normalized by their bound widths. Run this file to solve the problem and
compare it with the MDF solution in monolithic_solution.npz (regenerate it with ssbj.py).
"""

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import models
from models import analysis, design_constraints, report, cl, cu

HERE = os.path.dirname(os.path.abspath(__file__))

SHARED = np.arange(6)  # t/c, altitude, Mach number, aspect ratio, sweep, wing area
# each discipline's local design variables and its constraints (indices into z and into the constraints)
DISCIPLINES = {"structure": ([6, 7], np.arange(6)),   # taper ratio, wingbox area; stresses, twist
               "aerodynamics": ([8], [6]),              # skin friction; pressure gradient
               "propulsion": ([9], [7, 8, 9])}          # throttle; ESF, throttle / max, temperature

width = models.xu - models.xl
LP_PENALTY = 1e3      # on each unit of constraint violation in the elastic linear programs
MERIT_PENALTY = 1e2   # on the total constraint violation in the merit function
MOVE_LIMIT = 0.1      # initial move limit, as a fraction of each variable's bound width
MIN_MOVE_LIMIT = 1e-9


def functions(u):
    """[objective, constraints] at the normalized design u: the negative range in 1000 nm, then the
    design constraints, both at multidisciplinary feasibility."""
    out = analysis(models.xl + u * width)
    g = design_constraints(out["stress"], out["twist"], out["pressure_gradient"], out["esf"],
                           out["throttle_ratio"], out["temperature"])
    return jnp.concatenate([-out["range"][None] / 1e3, g])


functions_jit = jax.jit(functions)
total_derivatives = jax.jit(jax.jacfwd(functions))


def violation(g):
    return float(np.sum(np.maximum(g - cu, 0.0) + np.maximum(cl - g, 0.0)))


def elastic_lp(c, A, g, rows, lb, ub):
    """min c.d + LP_PENALTY * sum(slacks)  s.t.  cl <= g + A d <= cu elastically for the constraints
    in rows, and lb <= d <= ub.

    Returns d and, for each side of each constraint, (constraint, side, slack) with side +1 for
    the upper bound and -1 for the lower.
    """
    sides = [(k, 1) for k in rows if cu[k] < np.inf] + [(k, -1) for k in rows if cl[k] > -np.inf]
    n, m = len(c), len(sides)
    # side +1: A_k d - e <= cu_k - g_k;  side -1: -A_k d - e <= g_k - cl_k
    A_ub = np.array([np.concatenate([s * A[k], -np.eye(m)[j]]) for j, (k, s) in enumerate(sides)]).reshape(m, n + m)
    b_ub = np.array([cu[k] - g[k] if s > 0 else g[k] - cl[k] for k, s in sides])
    res = linprog(np.concatenate([c, np.full(m, LP_PENALTY)]), A_ub=A_ub if m else None, b_ub=b_ub if m else None,
                  bounds=list(zip(lb, ub)) + [(0, None)] * m, method="highs")
    d, e = res.x[:n], res.x[n:]
    return d, [(k, s, e[j]) for j, (k, s) in enumerate(sides)]


def post_optimality(d, active, A, A_shared, lb, ub, tol=1e-9):
    """dd*/dd_0 of a discipline's LP solution d with respect to the shared variables' step d_0.

    The constraints and bounds active at d stay active as d_0 changes: an active constraint k
    keeps A_k d + A_shared_k d_0 fixed, and a variable at its move limit or bound stays there.
    With fewer active rows than variables the LP's solution isn't unique, and the minimum-norm
    solution leaves the undetermined directions unchanged.
    """
    n, n0 = len(d), A_shared.shape[1]
    M = [A[k] for k in active] + [np.eye(n)[j] for j in range(n) if d[j] <= lb[j] + tol or d[j] >= ub[j] - tol]
    R = [-A_shared[k] for k in active] + [np.zeros(n0)] * (len(M) - len(active))
    if not M:
        return np.zeros((n, n0))
    return np.linalg.lstsq(np.array(M), np.array(R), rcond=None)[0]


def bliss_step(u, F, dF, r):
    """One BLISS step from the normalized design u, with values F and total derivatives dF, and
    move limit r. Returns the step."""
    g, dg = F[1:], dF[1:]
    df = dF[0]
    step = np.zeros_like(u)
    # the step's limits: the move limits and the bounds
    lb, ub = np.maximum(-u, -r), np.minimum(1 - u, r)

    # 2-3. discipline linear programs and their post-optimality sensitivities
    responses = {}
    for name, (local, rows) in DISCIPLINES.items():
        # this discipline's constraints only, as functions of its local variables
        A = np.zeros_like(dg[:, local]); A[rows] = dg[rows][:, local]
        d, sides = elastic_lp(df[local], A, g, rows, lb[local], ub[local])
        # the constraints active at d: tight, and met without slack
        active = [k for k, s, e in sides if e <= 1e-9 and abs(g[k] + A[k] @ d - (cu[k] if s > 0 else cl[k])) <= 1e-9]
        S = post_optimality(d, active, A, dg[:, SHARED], lb[local], ub[local])
        responses[name] = (local, d, S)

    # 4. system linear program over the shared variables, with every discipline's response
    c0 = df[SHARED].copy()
    A0 = dg[:, SHARED].copy()
    g0 = g.copy()
    for local, d, S in responses.values():
        c0 += df[local] @ S
        A0 += dg[:, local] @ S
        g0 += dg[:, local] @ d
    d0, _ = elastic_lp(c0, A0, g0, np.arange(len(g)), lb[SHARED], ub[SHARED])

    step[SHARED] = d0
    for local, d, S in responses.values():
        step[local] = np.clip(d + S @ d0, lb[local], ub[local])
    return step


u = (models.x0 - models.xl) / width
F = np.array(functions_jit(u)); dF = np.array(total_derivatives(u))
analyses = 1
r = MOVE_LIMIT
merit = F[0] + MERIT_PENALTY * violation(F[1:])
history = [u.copy()]  # the design after every accepted step
t_start = time.perf_counter()
iterations = 0
while r >= MIN_MOVE_LIMIT and iterations < 500:
    iterations += 1
    step = bliss_step(u, F, dF, r)
    if np.max(np.abs(step)) < 1e-12:
        break  # the linearized problem's solution is the current design
    u_new = np.clip(u + step, 0.0, 1.0)
    F_new = np.array(functions_jit(u_new))
    analyses += 1
    merit_new = F_new[0] + MERIT_PENALTY * violation(F_new[1:])
    if merit_new < merit - 1e-14:
        # accept, and relax the move limit if the step reached it
        u, F, merit = u_new, F_new, merit_new
        dF = np.array(total_derivatives(u))
        history.append(u.copy())
        if np.max(np.abs(step)) >= 0.99 * r:
            r = min(2 * r, MOVE_LIMIT)
    else:
        r /= 2
t_solve = time.perf_counter() - t_start

z = models.xl + u * width
g = F[1:]
report(z, -1e3 * F[0], g)

solution = np.load(os.path.join(HERE, "monolithic_solution.npz"))
error = [np.max(np.abs(models.xl + h * width - solution["z"]) / width) for h in history]
print(f"\nRange (MDF): {float(solution['range']):.2f} nm")
print(f"Max design error relative to the bound widths: {error[-1]:.2e}")
print(f"Max constraint violation: {max(np.max(g - cu), np.max(cl - g), 0.0):.2e}")
print(f"BLISS iterations: {iterations} ({len(history) - 1} accepted), coupled analyses: {analyses}, "
      f"final move limit: {r:.1e}, time: {t_solve:.1f} s")


# the design error and the constraint violation after every accepted step
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.25))
ax1.semilogy(error, color='tab:red', linewidth=2)
ax1.set_xlabel('Accepted step')
ax1.set_ylabel('Design error')
ax2.semilogy([violation(np.array(functions_jit(h))[1:]) + 1e-16 for h in history], color='tab:orange', linewidth=2)
ax2.set_xlabel('Accepted step')
ax2.set_ylabel('Constraint violation')
plt.tight_layout()
plt.show()
