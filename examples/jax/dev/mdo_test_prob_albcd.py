"""Distributed ALBCD version of ``mdo_test_prob.py``.

Each subproblem owns a copy of the n0 shared variables and its ni local
variables. The local sphere constraint stays inside the corresponding block;
spanning-tree consensus constraints enforce agreement between neighboring
copies of the shared variables.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import modopt as mo
from albcd import ALBCD, Subproblem


SETTINGS = dict(max_mu=1e3, rho=1.2, tau=0.5, feas_tol=1e-6,
                opt_tol=[1e-2, 1e-5], max_outer_iter=200,
                max_inner_iter=100, verbose=False)


def solve_case(N=4, n0=10, ni=20):
    """Solve the distributed MDO problem for arbitrary positive dimensions."""
    if N < 2 or n0 < 1 or ni < 1:
        raise ValueError("N must be at least 2, and n0 and ni must be positive")

    # Match the monolithic problem's driver assignment, including n0 > ni.
    drv = (np.arange(ni) * n0) // ni
    counts = np.bincount(drv, minlength=n0)
    scale = np.sqrt(counts * N)
    m = n0 + ni

    # x = [z_1, x_1, z_2, x_2, ..., z_N, x_N], where each z_i is a copy
    # of the monolithic global variable vector.
    block_xl = np.concatenate([np.full(n0, -1.5 * scale), np.full(ni, -1.5)])
    block_xu = -block_xl
    xl = np.tile(block_xl, N)
    xu = np.tile(block_xu, N)

    def local_objective(w):
        z = w[:n0]
        xs = w[n0:]
        u = z[drv] / scale[drv]
        terms = (1.0 - u) ** 2 + 100.0 * (xs - u ** 2) ** 2
        return jnp.mean(terms)

    def local_constraint(w):
        xs = w[n0:]
        return jnp.array([4.0 * jnp.mean(xs ** 2) - 1.0])

    def coupling(x):
        blocks = x.reshape(N, m)
        z = blocks[:, :n0]
        return (z[:-1] - z[1:]).ravel()

    class Block(Subproblem):

        def setup(self) -> None:
            self.obj_fn = jax.jit(self.merit)
            self.grad_fn = jax.jit(jax.grad(self.merit))
            self.con_fn = jax.jit(lambda v: local_constraint(v[:m]))
            self.jac_fn = jax.jit(jax.jacobian(lambda v: local_constraint(v[:m])))

        def merit(self, v, x, y, mu):
            x_with_v = x.at[self.index].set(v)
            c = coupling(x_with_v)
            return local_objective(v) + y @ c + 0.5 * mu @ c ** 2

        def solve(self, x, y, mu, data, outputs) -> None:
            x, y, mu = (jnp.asarray(value) for value in (x, y, mu))
            obj = lambda v: np.float64(self.obj_fn(v, x, y, mu))
            grad = lambda v: np.array(self.grad_fn(v, x, y, mu))
            con = lambda v: np.array(self.con_fn(v))
            jac = lambda v: np.array(self.jac_fn(v))

            problem = mo.ProblemLite(
                x0=np.array(self.decompose(x)), obj=obj, grad=grad,
                con=con, jac=jac, xl=xl[self.index], xu=xu[self.index],
                cl=-np.inf, cu=0.0)
            optimizer = mo.SLSQP(
                problem, solver_options={'maxiter': 300, 'ftol': 1e-14},
                turn_off_outputs=True)
            optimizer.solve()
            v = optimizer.results['x']

            # SLSQP's multiplier and local constraint Jacobian are used by
            # ALBCD's projected-gradient residual calculation.
            self._multiplier = np.asarray(optimizer.results['multipliers'])
            self._jac_con = jac(v)

            x_new = self.recompose(x, v)
            outputs['x'] = x_new
            outputs['phi'] = np.array(coupling(jnp.asarray(x_new)))

        def residual(self, x, y, mu, data) -> float:
            x, y, mu = (jnp.asarray(value) for value in (x, y, mu))
            v = jnp.asarray(self.decompose(x))
            grad = np.array(self.grad_fn(v, x, y, mu))
            grad += self._jac_con.T @ self._multiplier
            return float(np.max(np.abs(
                np.array(v) - np.clip(np.array(v) - grad, xl[self.index], xu[self.index]))))

    blocks = [Block(slice(i * m, (i + 1) * m)) for i in range(N)]
    opt = ALBCD(subproblems=blocks, x0=np.zeros(N * m),
                mu0=np.ones((N - 1) * n0), **SETTINGS)
    opt.solve()
    return opt, blocks, local_objective, local_constraint


if __name__ == '__main__':
    opt, blocks, local_objective, local_constraint = solve_case()
    w = opt.x.reshape(-1, blocks[0].index.stop - blocks[0].index.start)
    print(f'Objective: {sum(float(local_objective(wi)) for wi in w):.12f}')
    print(f'Local constraints: {[float(local_constraint(wi)[0]) for wi in w]} (<= 0)')
    print(f'Max consensus violation: {np.max(np.abs(opt.phi)):.2e}')
    print(f'Global copies: {w[:, :10]}')
