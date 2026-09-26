"""ALBCD version of the scalable test problem in scalable_test_problem.py.

Each of the N subproblems owns a local copy z_i of the n0 global variables and its own ni
local variables, so its sphere constraint involves only its own variables and is handled
inside the block. The coupling constraints phi are

    z_i - z_{i+1} = 0,  i = 1, ..., N-1       (spanning-tree consensus, chain)
    mean_i x_i[0] - 0.93 + s = 0              (global hyperplane constraint, slack s >= 0)

and are enforced through the augmented Lagrangian. The last block also owns the slack.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1") # speedup
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import modopt as mo
from albcd import ALBCD, Subproblem
import warnings
warnings.filterwarnings("ignore")

N = 4 # number of subproblems
n0 = 10 # number of global variables
ni = 20 # number of local variables
m = n0 + ni # variables per subproblem: [z_i, x_i]

# x = [z_1, x_1, z_2, x_2, ..., z_N, x_N, s]
xl = np.full(N * m + 1, -1.5)
xu = np.full(N * m + 1, 1.5)
xl[-1], xu[-1] = 0.0, np.inf # slack


def local_objective(w):
    """Rosenbrock function of one subproblem's variables w = [z_i, x_i]."""
    return jnp.sum(100 * (w[1:] - w[:-1] ** 2) ** 2 + (1.0 - w[:-1]) ** 2) / N


def local_constraint(w):
    """Sphere constraint of one subproblem, <= 0.8^2."""
    return jnp.sum(w ** 2, keepdims=True) / m


def coupling(x):
    """Consensus between neighboring copies of the global variables, then the global constraint."""
    w = x[:N * m].reshape(N, m)
    z, xs = w[:, :n0], w[:, n0:]
    plane = jnp.mean(xs[:, 0]) - 0.93 + x[-1]
    return jnp.concatenate([(z[:-1] - z[1:]).ravel(), plane[None]])


class Block(Subproblem):

    def setup(self) -> None:
        self.add_input("x")
        self.add_input("y")  # Lagrange multipliers
        self.add_input("mu") # penalty parameters
        self.add_output("x")
        self.add_output("phi") # coupling constraints at the new x

        # compiled once per block: x, y and mu are arguments rather than captured
        # constants, so new values between solves don't trigger a recompile
        con = lambda v: local_constraint(v[:m])
        self.obj = jax.jit(self.merit)
        self.grad = jax.jit(jax.grad(self.merit))
        self.con = jax.jit(con)
        self.jac = jax.jit(jax.jacobian(con))

    def merit(self, v, x, y, mu):
        """Augmented Lagrangian over this block's variables v, with the other blocks fixed."""
        c = coupling(x.at[self.index].set(v))
        return local_objective(v[:m]) + y @ c + 0.5 * mu @ c ** 2

    def solve(self, inputs, outputs) -> None:
        x, y, mu = (jnp.asarray(inputs[name]) for name in ("x", "y", "mu"))

        prob = mo.ProblemLite(x0=np.array(self.decompose(x)),
                              obj=lambda v: np.float64(self.obj(v, x, y, mu)),
                              grad=lambda v: np.array(self.grad(v, x, y, mu)),
                              con=lambda v: np.array(self.con(v)),
                              jac=lambda v: np.array(self.jac(v)),
                              xl=xl[self.index], xu=xu[self.index], cl=-np.inf, cu=0.8 ** 2)
        optimizer = mo.SLSQP(prob, solver_options={'maxiter': 300, 'ftol': 1e-10}, turn_off_outputs=True)
        optimizer.solve()
        v = optimizer.results['x']

        # SLSQP's multiplier of 0.8^2 - c >= 0, and the constraint gradient, for residual()
        self._multiplier = np.asarray(optimizer.results['multipliers'])
        self._jac_con = np.array(self.jac(v))

        x_new = self.recompose(x, v)
        outputs["x"] = x_new
        outputs["phi"] = np.array(coupling(jnp.asarray(x_new)))

    def residual(self, inputs) -> float:
        """Projected-gradient KKT residual, with the sphere constraint's multiplier and the bounds."""
        x = jnp.asarray(inputs["x"])
        v = np.array(self.decompose(x))
        grad = np.array(self.grad(jnp.asarray(v), x, inputs["y"], inputs["mu"]))
        grad += self._jac_con.T @ self._multiplier
        return float(np.max(np.abs(v - np.clip(v - grad, xl[self.index], xu[self.index]))))


blocks = [Block(slice(i * m, (i + 1) * m)) for i in range(N - 1)]
blocks.append(Block(slice((N - 1) * m, N * m + 1))) # the last block also owns the slack

opt = ALBCD(subproblems=blocks,
            x0=np.zeros(N * m + 1),
            mu0=np.ones((N - 1) * n0 + 1),
            max_mu=1e3,
            rho=1.2,
            tau=0.5,
            feas_tol=1e-6,
            opt_tol=[1e-2, 1e-5],
            max_outer_iter=200,
            max_inner_iter=100)

opt.solve()

w = opt.x[:N * m].reshape(N, m)
print(f"Objective: {sum(local_objective(wi) for wi in w):.6f}")
print(f"Sphere constraints: {[float(local_constraint(wi)[0]) for wi in w]} (<= {0.8 ** 2})")
print(f"Plane constraint: {np.mean(w[:, n0]):.6f} (<= 0.93)")
print(f"Max consensus violation: {np.max(np.abs(opt.phi[:-1])):.2e}")
print(f"Global variables: {w[0, :n0]}")
