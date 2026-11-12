"""First-order vs second-order (Bertsekas) multiplier update on small test problems.

Each problem is solved twice with ALBCD3, once per multiplier update, from the same start
and with the same settings. The problems are the ones in quadratic_global_circle.py,
quadratic_global_linear.py and rosenbrock_consensus.py, plus an equality-constrained
quadratic whose curvature is large compared with the penalty parameter.

    python examples/jax/albcd3_simple.py
"""

import numpy as np
import modopt as mo
import jax
jax.config.update("jax_enable_x64", True) # jax defaults to float32
import jax.numpy as jnp
from albcd import Subproblem
from albcd.albcd3 import ALBCD3
import warnings
warnings.filterwarnings("ignore")


class Block(Subproblem):
    """One block of min f(x) s.t. phi(x) = 0, with f and phi written in the global x.

    The block minimizes the augmented Lagrangian over its own entries of x with SLSQP,
    subject to its bounds xl <= v <= xu and, if given, local constraints con(v) >= 0.
    """

    def __init__(self, index, f, phi, xl=None, xu=None, con=None):
        self.f, self.phi, self.con = f, phi, con
        self.xl, self.xu = xl, xu
        super().__init__(index)

    def setup(self) -> None:
        def lagrangian(x, y, mu):
            c = self.phi(x)
            return self.f(x) + jnp.sum(y * c) + 0.5 * jnp.sum(mu * c**2)

        self._obj = jax.jit(lagrangian)
        self._grad = jax.jit(jax.grad(lagrangian))
        self._phi = jax.jit(self.phi)
        if self.con is not None:
            self._con = jax.jit(self.con)
            self._jac = jax.jit(jax.jacrev(self.con))
            self._multipliers = None # SLSQP's multipliers of con(v) >= 0, set by solve()

    def solve(self, x, y, mu, data, outputs) -> None:
        full = lambda v: self.recompose(x, v)
        kwargs = dict(x0=np.array(self.decompose(x)),
                      obj=lambda v: float(self._obj(full(v), y, mu)),
                      grad=lambda v: np.array(self._grad(full(v), y, mu))[self.index],
                      xl=self.xl, xu=self.xu)
        if self.con is not None:
            kwargs.update(con=lambda v: np.array(self._con(v)), jac=lambda v: np.array(self._jac(v)),
                          cl=0.0, cu=np.inf)

        optimizer = mo.SLSQP(mo.ProblemLite(**kwargs), solver_options={'maxiter': 200, 'ftol': 1e-14},
                             turn_off_outputs=True)
        optimizer.solve()
        if self.con is not None:
            self._multipliers = np.asarray(optimizer.results['multipliers'])

        x_new = full(optimizer.results['x'])
        outputs["x"] = x_new
        outputs["phi"] = np.array(self._phi(x_new))

    def gradient(self, x, y, mu, data) -> np.ndarray:
        g = np.array(self._grad(x, y, mu))[self.index]
        if self.con is not None and self._multipliers is not None:
            g = g - np.array(self._jac(self.decompose(x))).T @ self._multipliers
        return g

    def _at_bound(self, x, y, mu, data):
        # entries of v held at a bound by the gradient
        v, g = self.decompose(x), self.gradient(x, y, mu, data)
        lower = np.zeros(v.size, dtype=bool) if self.xl is None else (v <= self.xl + 1e-9) & (g > 0)
        upper = np.zeros(v.size, dtype=bool) if self.xu is None else (v >= self.xu - 1e-9) & (g < 0)
        return lower | upper

    def residual(self, x, y, mu, data) -> float:
        g = self.gradient(x, y, mu, data)
        g[self._at_bound(x, y, mu, data)] = 0.0
        return float(np.max(np.abs(g)))

    def active(self, x, y, mu, data) -> np.ndarray:
        # unit rows for the entries at a bound, then the Jacobian rows of the active local constraints
        v = self.decompose(x)
        rows = np.eye(v.size)[self._at_bound(x, y, mu, data)]
        if self.con is not None:
            jac = np.atleast_2d(np.array(self._jac(v)))
            rows = np.vstack([rows, jac[np.array(self._con(v)) <= 1e-8]])
        return rows


def run(name, make_blocks, x0, mu0, x_star=None, **options):
    print(f'\n{name}')
    results = {}
    for update in ('first_order', 'second_order'):
        opt = ALBCD3(make_blocks(), x0=x0, mu0=mu0, multiplier_update=update, verbose=False, **options)
        try:
            opt.solve()
        except np.linalg.LinAlgError as error: # the second-order update has no safeguards
            print(f'  {update:12s} | CRASHED ({error}) after {len(opt.opt_history)} sweeps | '
                  f'x = {np.round(opt.x, 3)}   y = {np.round(opt.y, 3)}')
            continue
        error = '' if x_star is None else f' | x error {np.max(np.abs(opt.x - x_star)):.1e}'
        print(f'  {update:12s} | {"converged" if opt.success else "FAILED   "} | outer {opt.outer_iterations:3d} | '
              f'sweeps {len(opt.opt_history):4d} | feas {opt.feas_history[-1]:.1e} | opt {opt.opt_history[-1]:.1e} | '
              f'max mu {opt.mu.max():8.2f}{error} | {opt.tf:5.1f} s')
        print(f'               x = {np.round(opt.x, 6)}   y = {np.round(opt.y, 6)}')
        results[update] = opt
    return results


if __name__ == "__main__":

    options = dict(max_mu=1e3, rho=1.2, tau=0.5, feas_tol=1e-6, opt_tol=1e-6, max_outer_iter=200, max_inner_iter=100)

    quadratic = lambda x1, x2: x1**2 + x2**2 - 1.5 * x1 * x2
    rosenbrock = lambda x1, x2: (1 - x1)**2 + 100 * (x2 - x1**2)**2
    s_lower, s_upper = np.array([-np.inf, 0.0]), np.array([np.inf, np.inf])

    # 1. quadratic objective, x1^2 + x2^2 >= 0.25. x = [x1, s, x2] with slack s >= 0
    f = lambda x: quadratic(x[0], x[2])
    phi = lambda x: jnp.array([-(x[0]**2 + x[2]**2) + 0.25 + x[1]])
    run('quadratic, circle constraint',
        lambda: [Block(slice(0, 2), f, phi, xl=s_lower, xu=s_upper), Block(slice(2, 3), f, phi)],
        x0=np.array([-0.5, 0.0, 1.0]), mu0=np.array([1.0]), **options)

    # 2. quadratic objective, x1 - 0.5 x2 >= 0.5. x = [x1, s, x2] with slack s >= 0
    #    (quadratic_global_linear.py has the right-hand side 0, where the constraint is weakly active)
    phi = lambda x: jnp.array([-(x[0] - 0.5 * x[2]) + 0.5 + x[1]])
    run('quadratic, linear constraint',
        lambda: [Block(slice(0, 2), f, phi, xl=s_lower, xu=s_upper), Block(slice(2, 3), f, phi)],
        x0=np.array([-0.5, 0.0, 1.0]), mu0=np.array([1.0]), **options)

    # 3. stiff quadratic with two linear equality constraints: curvature up to 1e3, mu0 = 1
    #    x = [a (3), b (3)], f = 1/2 a'Qa + 1/2 b'Rb - p'[a, b], constraints C [a, b] = d
    rng = np.random.default_rng(0)
    Q = jnp.asarray(np.diag([1.0, 30.0, 1e3]))
    R = jnp.asarray(np.diag([3.0, 100.0, 300.0]))
    p = jnp.asarray(rng.standard_normal(6))
    C = jnp.asarray(rng.standard_normal((2, 6)))
    d = jnp.asarray(rng.standard_normal(2))
    f = lambda x: 0.5 * x[:3] @ Q @ x[:3] + 0.5 * x[3:] @ R @ x[3:] - p @ x
    phi = lambda x: C @ x - d
    K = np.block([[np.diag(np.concatenate([np.diag(Q), np.diag(R)])), np.array(C).T], [np.array(C), np.zeros((2, 2))]])
    kkt = np.linalg.solve(K, np.concatenate([np.array(p), np.array(d)]))
    print(f'\nstiff quadratic reference: x = {np.round(kkt[:6], 6)}   y = {np.round(kkt[6:], 6)}')
    run('stiff quadratic, linear equality constraints',
        lambda: [Block(slice(0, 3), f, phi), Block(slice(3, 6), f, phi)],
        x0=np.zeros(6), mu0=np.ones(2), x_star=kkt[:6], **options)

    # 4. Rosenbrock in consensus form. x = [x1_1, x2_1, x1_2, x2_2]: each block owns a copy of
    #    (x1, x2), half of the objective and the local constraint x1^2 + x2^2 >= 0.25
    f = lambda x: 0.5 * rosenbrock(x[0], x[1]) + 0.5 * rosenbrock(x[2], x[3])
    phi = lambda x: x[:2] - x[2:]
    circle = lambda v: jnp.array([v[0]**2 + v[1]**2 - 0.25])
    #    The first start is the one in rosenbrock_consensus.py; the copies start equal and stay equal,
    #    so no multiplier update is needed. The other starts have the copies on opposite sides.
    blocks = lambda: [Block(slice(0, 2), f, phi, con=circle), Block(slice(2, 4), f, phi, con=circle)]
    for x0, mu0 in (([-0.5, 1.0, -0.5, 1.0], 10.0), ([-1.2, 1.0, 1.2, 1.0], 10.0), ([-1.2, 1.0, 1.2, 1.0], 1.0)):
        run(f'Rosenbrock consensus, local circle constraints, x0 = {x0}, mu0 = {mu0}',
            blocks, x0=np.array(x0), mu0=np.full(2, mu0), **options)
