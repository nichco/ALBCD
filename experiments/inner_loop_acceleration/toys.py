"""Cheap benchmark problems for the ALBCD2 inner loop.

Each problem is min f(x) s.t. phi(x) = 0, xl <= x <= xu, split into blocks. Block solves use
scipy L-BFGS-B on the block's augmented Lagrangian with JAX gradients, to a tight tolerance,
so they behave like the "exact" block minimizations ALBCD assumes. Every block solve records
its function evaluations so the cost of a solve, not just the number of solves, is tracked.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize
from albcd import Subproblem

COUNTER = {"nfev": 0}
import os
SOLVER = os.environ.get("TOY_SOLVER", "lbfgsb")  # lbfgsb | bfgs | bfgs_warm


class Problem:
    def __init__(self, f, phi, n, blocks, xl=None, xu=None, x0=None, m=None):
        self.f, self.phi, self.n, self.blocks = f, phi, n, blocks
        self.xl = np.full(n, -np.inf) if xl is None else xl
        self.xu = np.full(n, np.inf) if xu is None else xu
        self.x0 = np.zeros(n) if x0 is None else x0
        self.m = m
        def al(x, y, mu):
            c = phi(x)
            return f(x) + jnp.sum(y * c) + 0.5 * jnp.sum(mu * c ** 2)
        self.al = jax.jit(al)
        self.al_grad = jax.jit(jax.grad(al))
        self.phi_j = jax.jit(phi)

    def subproblems(self, gtol=1e-11):
        return [Block(self, idx, gtol) for idx in self.blocks]


class Block(Subproblem):
    def __init__(self, prob, index, gtol):
        self.prob, self.gtol = prob, gtol
        self.xl, self.xu = prob.xl[index], prob.xu[index]
        super().__init__(index)

    def solve(self, x, y, mu, data, outputs):
        p = self.prob
        idx = self.index
        xj, yj, muj = jnp.asarray(x), jnp.asarray(y), jnp.asarray(mu)

        def fg(v):
            xx = xj.at[idx].set(jnp.asarray(v))
            return float(p.al(xx, yj, muj)), np.asarray(p.al_grad(xx, yj, muj))[idx]

        bounds = list(zip(p.xl[idx], p.xu[idx]))
        v0 = np.clip(x[idx], p.xl[idx], p.xu[idx])
        gt = float(np.clip(data.get('_tol', self.gtol), self.gtol, 1e-2))
        if SOLVER.startswith('bfgs'):
            opts = {'maxiter': 5000, 'gtol': gt}
            if SOLVER == 'bfgs_warm' and getattr(self, '_H', None) is not None:
                # recycle the inverse-Hessian estimate of this block's previous solve
                H = 0.5 * (self._H + self._H.T)
                if np.linalg.eigvalsh(H).min() > 1e-12 * np.abs(H).max():
                    opts['hess_inv0'] = H
            res = minimize(fg, v0, jac=True, method='BFGS', options=opts)
            self._H = res.hess_inv
            COUNTER["nfev"] += res.nfev
            x_new = self.recompose(x, res.x)
            outputs["x"] = x_new
            outputs["phi"] = np.asarray(p.phi_j(jnp.asarray(x_new)))
            return
        res = minimize(fg, v0, jac=True, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 5000, 'gtol': float(np.clip(data.get('_tol', self.gtol), self.gtol, 1e-2)), 'ftol': 1e-15, 'maxcor': 30})
        COUNTER["nfev"] += res.nfev
        x_new = self.recompose(x, res.x)
        outputs["x"] = x_new
        outputs["phi"] = np.asarray(p.phi_j(jnp.asarray(x_new)))

    def gradient(self, x, y, mu, data):
        p = self.prob
        idx = self.index
        g = np.asarray(p.al_grad(jnp.asarray(x), jnp.asarray(y), jnp.asarray(mu)))[idx]
        v = x[idx]
        return v - np.clip(v - g, p.xl[idx], p.xu[idx])


def _blocks(sizes):
    out, s = [], 0
    for k in sizes:
        out.append(np.arange(s, s + k)); s += k
    return out


def quad_copy(seed=0, na=15, nb=15, p=8, cond=1e2, couple=1.0):
    """Two disciplines coupled through copies: block A owns (a, Wc), block B owns (b, Lc).
    phi = [L(a) - Lc, W(b) - Wc], L and W linear. Quadratic objective in each block."""
    rng = np.random.default_rng(seed)
    def spd(k):
        Q, _ = np.linalg.qr(rng.standard_normal((k, k)))
        return Q @ np.diag(np.logspace(0, np.log10(cond), k)) @ Q.T
    nA, nB = na + 1, nb + p
    HA, HB = spd(nA), spd(nB)
    gA, gB = rng.standard_normal(nA), rng.standard_normal(nB)
    P = couple * rng.standard_normal((p, na)) / np.sqrt(na)
    qv = couple * rng.standard_normal(nb) / np.sqrt(nb)
    HA, HB, gA, gB, P, qv = map(jnp.asarray, (HA, HB, gA, gB, P, qv))

    def f(x):
        u, w = x[:nA], x[nA:]
        return 0.5 * u @ HA @ u + gA @ u + 0.5 * w @ HB @ w + gB @ w
    def phi(x):
        a, Wc, b, Lc = x[:na], x[na], x[nA:nA + nb], x[nA + nb:]
        return jnp.concatenate([P @ a - Lc, jnp.reshape(qv @ b - Wc, 1)])
    return Problem(f, phi, nA + nB, _blocks([nA, nB]), m=p + 1)


def nonlin_copy(seed=0, na=12, nb=12, p=8, bounds=True):
    """Nonlinear analogue of quad_copy (aerostructural flavour): loads saturate in a (tanh),
    weight is quadratic in b, the objective in each block is non-quadratic, B has bounds."""
    rng = np.random.default_rng(seed)
    nA, nB = na + 1, nb + p
    P = jnp.asarray(rng.standard_normal((p, na)) / np.sqrt(na))
    R = jnp.asarray(rng.standard_normal((nb, p)) / np.sqrt(p))
    cA = jnp.asarray(rng.standard_normal(na))
    cB = jnp.asarray(rng.uniform(0.5, 1.5, nb))

    def f(x):
        a, Wc, b, Lc = x[:na], x[na], x[nA:nA + nb], x[nA + nb:]
        drag = jnp.sum((a - cA) ** 2) + 0.1 * jnp.sum(a ** 4) + 0.5 * Wc ** 2 + jnp.sum(jnp.cos(a))
        # "structure": compliance-like term depends on loads and sizing
        comp = jnp.sum((R @ Lc) ** 2 / (0.2 + b ** 2)) + 0.5 * jnp.sum((b - cB) ** 2)
        return drag + comp
    def phi(x):
        a, Wc, b, Lc = x[:na], x[na], x[nA:nA + nb], x[nA + nb:]
        L = 2.0 * jnp.tanh(P @ a) + 0.3 * Wc
        W = jnp.sum(b ** 2) / nb
        return jnp.concatenate([L - Lc, jnp.reshape(W - Wc, 1)])
    xl = np.full(nA + nB, -np.inf); xu = np.full(nA + nB, np.inf)
    if bounds:
        xl[nA:nA + nb] = 0.3; xu[nA:nA + nb] = 2.0
    x0 = np.zeros(nA + nB); x0[nA:nA + nb] = 1.0
    return Problem(f, phi, nA + nB, _blocks([nA, nB]), xl, xu, x0, m=p + 1)


def rosen_consensus(n=6):
    """Chained Rosenbrock split into 2 blocks; the shared variables are duplicated (consensus)."""
    h = n // 2
    # block 1 owns x[0:h+1] (its copy of the interface var x_h), block 2 owns x[h:n] plus copy
    def f(z):
        x1 = z[:h + 1]; x2 = z[h + 1:]
        r1 = jnp.sum(100 * (x1[1:] - x1[:-1] ** 2) ** 2 + (1 - x1[:-1]) ** 2)
        r2 = jnp.sum(100 * (x2[1:] - x2[:-1] ** 2) ** 2 + (1 - x2[:-1]) ** 2)
        return r1 + r2
    def phi(z):
        return jnp.reshape(z[h] - z[h + 1], 1)
    x0 = np.full(n + 1, -1.2)
    return Problem(f, phi, n + 1, _blocks([h + 1, n - h]), x0=x0, m=1)


def shared3(seed=0, k=10, p=6, cond=1e2):
    """Three blocks coupled through a shared linear constraint sum_j A_j x_j = b (no copies)."""
    rng = np.random.default_rng(seed)
    Hs, gs, As = [], [], []
    for _ in range(3):
        Q, _ = np.linalg.qr(rng.standard_normal((k, k)))
        Hs.append(Q @ np.diag(np.logspace(0, np.log10(cond), k)) @ Q.T)
        gs.append(rng.standard_normal(k)); As.append(rng.standard_normal((p, k)))
    H = jnp.asarray(np.block([[Hs[i] if i == j else np.zeros((k, k)) for j in range(3)] for i in range(3)]))
    g = jnp.asarray(np.concatenate(gs)); A = jnp.asarray(np.hstack(As)); bb = jnp.asarray(rng.standard_normal(p))
    f = lambda x: 0.5 * x @ H @ x + g @ x + 0.05 * jnp.sum(x ** 4)
    phi = lambda x: A @ x - bb
    return Problem(f, phi, 3 * k, _blocks([k, k, k]), m=p)


PROBLEMS = {
    "quad_copy": quad_copy,
    "quad_copy_stiff": lambda: quad_copy(couple=4.0, cond=1e3),
    "nonlin_copy": nonlin_copy,
    "rosen_consensus": rosen_consensus,
    "shared3": shared3,
}
