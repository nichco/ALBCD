"""Lunar ascent trajectory optimization with ALBCD, with pure consensus coupling.

The same problem and arc decomposition as lunar_ascent.py (Braun, PhD thesis, Stanford,
1996, Section 6.2), with one difference in how the arcs are coupled. In lunar_ascent.py
the coupling constraints penalize X_end_i(v_i) - X_{i+1}: the end state of arc i is
computed from arc i's variables (a nonlinear function of its slopes and dt_i) and
compared with the start state owned by arc i + 1. Here each arc also owns a copy Xe_i
of its end state, tied to the computed end state by a local constraint, so that every
coupling constraint is a linear difference of two copies:

    ArcSubproblem i  owns [X_i, dt_i, Xdot_i, beta_i, Xe_i]
                     local constraints: midpoint defects, Xe_i = computed end state
                                        (+ initial state on arc 0, terminal state on arc N - 1)

    phi = [Xe_i - X_{i+1},  dt_i - dt_{i+1}]   for i = 0, ..., N - 2

The last arc also has an end-state copy, which keeps the block layout uniform; its
terminal constraints are written on that copy. Gradients come from JAX.

Run this file to solve the problem with ALBCD, compare the result with the monolithic
solution (the same variables and constraints solved with one SLSQP), and plot the
trajectory and the convergence. Pass the number of arcs as an argument, e.g.
`python lunar_ascent_consensus.py 4`; it must divide s = 24.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # small SLSQP matrices; must precede numpy
import sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # modopt works in float64
import jax.numpy as jnp
import modopt as mo
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from albcd import ALBCD, Subproblem
import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = os.path.splitext(os.path.basename(__file__))[0]


N = int(sys.argv[1]) if len(sys.argv) > 1 else 3  # number of arcs (blocks); Braun uses 3
s = 24                                            # total number of segments
assert s % N == 0, "the number of arcs must divide the number of segments"
m = s // N                                        # segments per arc

a = 3.0                                # thrust acceleration, lunar g's
yf = 1.129                             # final radius, 120 nm orbit
uf = np.sqrt(yf)                       # circular orbit speed
X_initial = jnp.array([0.0, 1.0, 0.0, 0.0])        # [x, y, u, v] on the surface
X_final = jnp.array([yf, uf, 0.0])                 # [y, u, v] in orbit (x(tf) is free)

# dimensional conversions, for plotting only
R_nm = 120.0 / (yf - 1.0)                          # lunar radius implied by the normalization, nm
T_s = np.sqrt(R_nm * 6076.12 / 5.3)                # time unit sqrt(R/g), s (g = 5.3 ft/s^2)

# global x = [v_0, v_1, ..., v_{N-1}], where v_i = [X_i (4), dt_i, Xdot_i (4m), beta_i (m), Xe_i (4)]
nv = 9 + 5 * m
nphi = 5 * (N - 1)

# bounds on one arc's variables: only the segment duration is bounded
xl = np.full(nv, -np.inf)
xu = np.full(nv, np.inf)
xl[4], xu[4] = 1e-4, 1.0


def eom(X, beta):
    """Normalized equations of motion; X is (..., 4) = [x, y, u, v]."""
    u, v = X[..., 2], X[..., 3]
    return jnp.stack([u, v, a * jnp.cos(beta), a * jnp.sin(beta) - 1.0], axis=-1)


def unpack(v):
    return v[:4], v[4], v[5:5 + 4 * m].reshape(m, 4), v[5 + 4 * m:5 + 5 * m], v[5 + 5 * m:]


def arc_profiles(v):
    """State (m + 1, 4) at the segment boundaries of one arc, and its midpoint thrust angles (m)."""
    X0, dt, Xdot, beta, _ = unpack(v)
    X = jnp.concatenate([X0[None, :], X0 + jnp.cumsum(Xdot * dt, axis=0)])
    return X, beta


def defects(v):
    """Midpoint collocation defects of one arc (Braun eqs. 6.12 - 6.15), all = 0."""
    _, dt, Xdot, beta, _ = unpack(v)
    X, _ = arc_profiles(v)
    X_mid = X[:-1] + 0.5 * dt * Xdot
    return (Xdot - eom(X_mid, beta)).ravel()


def local_constraints(v, i):
    """Constraints of arc i alone, all = 0: its defects, its end-state copy, and the boundary states it touches."""
    X, _ = arc_profiles(v)
    Xe = unpack(v)[4]
    con = [defects(v), Xe - X[-1]]
    if i == 0:
        con.append(X[0] - X_initial)
    if i == N - 1:
        con.append(Xe[1:] - X_final)
    return jnp.concatenate(con)


def coupling(x):
    """Coupling constraints phi: at each arc boundary, the end-state copy equals the next start state, and equal dt."""
    blocks = [x[j * nv:(j + 1) * nv] for j in range(N)]
    phi = []
    for j in range(N - 1):
        cur, nxt = blocks[j], blocks[j + 1]
        phi.append(jnp.concatenate([cur[5 + 5 * m:] - nxt[:4], (cur[4] - nxt[4])[None]]))
    return jnp.concatenate(phi)


def arc_time(v):
    """This arc's duration m dt_i; the arc durations sum to tf."""
    return m * v[4]


class ArcSubproblem(Subproblem):
    """Arc i: owns [X_i, dt_i, Xdot_i, beta_i, Xe_i] and satisfies the equations of motion along the arc."""

    def __init__(self, i):
        self.i = i
        super().__init__(slice(i * nv, (i + 1) * nv))

    def setup(self) -> None:
        # jit once; v, x, y and mu are arguments, so every solve reuses the compiled code
        self._obj = jax.jit(self.objective)
        self._grad = jax.jit(jax.grad(self.objective))
        self._con = jax.jit(self.local_constraints)
        self._jac = jax.jit(jax.jacfwd(self.local_constraints))

    def objective(self, v, x, y, mu):
        """Augmented Lagrangian as a function of this block's variables v."""
        c = coupling(x.at[self.index].set(v))
        # this arc's share of tf (the other arcs' share is constant in this block)
        return arc_time(v) + jnp.sum(y * c) + 0.5 * jnp.sum(mu * c**2)

    def local_constraints(self, v):
        return local_constraints(v, self.i)

    def solve(self, x, y, mu, data, outputs) -> None:
        args = tuple(jnp.asarray(a) for a in (x, y, mu))

        prob = mo.ProblemLite(x0=np.array(self.decompose(x)),
                              obj=lambda v: float(self._obj(jnp.asarray(v), *args)),
                              grad=lambda v: np.asarray(self._grad(jnp.asarray(v), *args)),
                              con=lambda v: np.asarray(self._con(jnp.asarray(v))),
                              jac=lambda v: np.asarray(self._jac(jnp.asarray(v))),
                              xl=xl, xu=xu, cl=0, cu=0)
        optimizer = mo.SLSQP(prob, solver_options={"maxiter": 1000, "ftol": 1e-12}, turn_off_outputs=True)
        optimizer.solve()
        v_new = optimizer.results["x"]

        # cache SLSQP's multipliers and the local constraints' Jacobian at the solution for residual()
        self._multipliers = np.asarray(optimizer.results["multipliers"])
        self._jac_con = np.asarray(self._jac(jnp.asarray(v_new)))

        outputs["x"] = self.recompose(x, v_new)
        outputs["phi"] = np.asarray(coupling(jnp.asarray(outputs["x"])))

    def residual(self, x, y, mu, data) -> float:
        args = tuple(jnp.asarray(a) for a in (x, y, mu))
        v = self.decompose(x)
        grad_f = np.asarray(self._grad(jnp.asarray(v), *args))

        # Lagrangian gradient projected onto the box bounds: zero iff v is a KKT point of this block
        grad_L = grad_f - self._jac_con.T @ self._multipliers
        return float(np.max(np.abs(v - np.clip(v - grad_L, xl, xu))))


# initial guess (the same as lunar_ascent.py): a linear thrust angle sweep from 70 to -50 deg
# over tf = 0.6, sampled at the segment midpoints, with the slopes from an explicit march of the
# scheme. Each arc's end-state copy starts at its computed end state, so the guess is continuous
# across the arc boundaries but satisfies neither the defects nor the terminal state
dt0 = 0.6 / s
beta_nodes = np.deg2rad(np.linspace(70.0, -50.0, s + 1))
X_nodes = np.zeros((s + 1, 4))
X_nodes[0] = X_initial
Xdot0 = np.zeros((s, 4))
for k in range(s):
    Xdot0[k] = np.asarray(eom(X_nodes[k], 0.5 * (beta_nodes[k] + beta_nodes[k + 1])))
    X_nodes[k + 1] = X_nodes[k] + Xdot0[k] * dt0
beta_mid0 = 0.5 * (beta_nodes[:-1] + beta_nodes[1:])
x0 = np.concatenate([np.concatenate([X_nodes[i * m], [dt0], Xdot0[i * m:(i + 1) * m].ravel(),
                                     beta_mid0[i * m:(i + 1) * m], X_nodes[(i + 1) * m]])
                     for i in range(N)])


# monolithic reference: the same variables and constraints solved with SLSQP
def monolithic_constraints(x):
    local = [local_constraints(x[i * nv:(i + 1) * nv], i) for i in range(N)]
    return jnp.concatenate(local + ([coupling(x)] if N > 1 else []))


nc = int(monolithic_constraints(jnp.asarray(x0)).size)
reference = mo.JaxProblem(x0=x0, jax_obj=lambda x: sum(arc_time(x[i * nv:(i + 1) * nv]) for i in range(N)),
                          jax_con=monolithic_constraints, cl=np.zeros(nc), cu=np.zeros(nc),
                          xl=np.tile(xl, N), xu=np.tile(xu, N))
reference_opt = mo.SLSQP(reference, solver_options={"maxiter": 1000, "ftol": 1e-12}, turn_off_outputs=True)
reference_opt.solve()
x_star = np.asarray(reference_opt.results["x"])
tf_star = s * x_star[4]
print(f"Monolithic: tf {tf_star:.6f} ({tf_star * T_s:.1f} s), {reference_opt.results['nit']} SLSQP iterations")


opt = ALBCD(subproblems=[ArcSubproblem(i) for i in range(N)],
            x0=x0,
            mu0=np.full(nphi, 10.0),
            max_mu=1e2,
            rho=1.5,
            tau=0.5,
            feas_tol=1e-6,
            opt_tol=[1e-2, 3e-4],
            max_y=1e6,
            max_outer_iter=500,
            max_inner_iter=100)

opt.solve()


blocks = opt.x.reshape(N, nv)
tf = sum(float(arc_time(v)) for v in blocks)
error = np.linalg.norm(np.array(opt.x_history) - x_star, axis=1) / np.linalg.norm(x_star)
print(f"ALBCD ({N} arcs, {m} segments each): tf {tf:.6f} ({tf * T_s:.1f} s), "
      f"segment durations {np.array2string(blocks[:, 4], precision=6)}")
print(f"Max coupling violation: {np.max(np.abs(opt.phi)):.2e}, "
      f"max local violation: {np.max(np.abs(np.asarray(monolithic_constraints(jnp.asarray(opt.x))))):.2e}")
print(f"tf error vs monolithic: {abs(tf - tf_star) / tf_star:.2e}, relative error in x: {error[-1]:.2e}")


def plot_trajectory(ax, x, labels=True, **kwargs):
    """Altitude vs. downrange and thrust angle vs. normalized time along each arc of x."""
    blocks = x.reshape(N, nv)
    tf = sum(float(arc_time(v)) for v in blocks)
    t_start = 0.0
    for i, v in enumerate(blocks):
        X, beta = (np.asarray(p) for p in arc_profiles(jnp.asarray(v)))
        t = t_start + v[4] * np.arange(m + 1)
        t_start = t[-1]
        ax[0].plot(X[:, 0] * R_nm, (X[:, 1] - 1) * R_nm, "o-", ms=4, label=f"Arc {i}" if labels else None, **kwargs)
        ax[1].plot(0.5 * (t[:-1] + t[1:]) / tf, np.rad2deg(beta), "o-", ms=4, **kwargs)
    ax[0].axhline(120, color="k", ls="--", lw=1)
    ax[0].set_xlabel("Downrange (nm)")
    ax[0].set_ylabel("Altitude (nm)")
    ax[0].legend(loc="lower right")
    ax[1].set_xlabel("time / $t_f$")
    ax[1].set_ylabel(r"Thrust angle $\beta$ (deg)")


fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
plot_trajectory(ax, opt.x)
fig.tight_layout()

fig, ax = plt.subplots(1, 2, figsize=(8, 2.5))
ax[0].semilogy(error, linewidth=2, color="tab:blue")
ax[0].set_xlabel("Subproblem solve")
ax[0].set_ylabel("Relative error")
ax[0].grid(color="lavender", alpha=0.5, axis="y")

ax[1].semilogy(opt.feas_history, linewidth=2, color="tab:orange")
ax[1].axhline(opt.feas_tol, color="gray", linewidth=1, linestyle="--")
ax[1].set_xlabel("Sweep")
ax[1].set_ylabel("Feasibility")
ax[1].grid(color="lavender", alpha=0.5, axis="y")

plt.tight_layout()

# convergence gif: the trajectory after each sweep (at most 200 frames) over the monolithic solution in gray
sweeps = np.array(opt.x_history)[N::N]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))


def draw(k):
    for a in ax:
        a.clear()
    plot_trajectory(ax, x_star, labels=False, color="0.8", lw=4)
    plot_trajectory(ax, sweeps[k])
    ax[0].set(xlim=(-10, 250), ylim=(-10, 150))
    ax[1].set(xlim=(0, 1), ylim=(-90, 90))
    fig.suptitle(f"Sweep {k + 1} of {len(sweeps)}, max|phi| = {opt.feas_history[k]:.1e}")
    fig.tight_layout()


frames = np.unique(np.linspace(0, len(sweeps) - 1, 200).astype(int))
FuncAnimation(fig, draw, frames=frames).save(os.path.join(HERE, f"{NAME}.gif"), writer=PillowWriter(fps=15))

plt.show()
