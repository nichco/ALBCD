"""Scaling study of ALBCD on the scalable test problem in scalable_test_problem_albcd.py.

Solves the problem for a range of sizes (N subproblems, n0 global variables, ni local
variables per subproblem) with the same ALBCD settings and tolerances, and records for each:

- solves:       subproblem (block) solves
- analyses:     subproblem model evaluations
- derivatives:  subproblem model derivative evaluations
- wall_time:    ALBCD solve time in seconds, after the JAX functions are compiled
- compile_time: time to compile the JAX functions, in seconds

and whether ALBCD met its tolerances (success), with the objective, the largest coupling
constraint violation (feasibility) and the largest block optimality residual (optimality) at
the solution. Counts from cases that didn't succeed aren't comparable with the others.

A subproblem's model is its Rosenbrock term f_i and sphere constraint g_i, a function of its
variables [z_i, x_i]. An analysis is an evaluation of f_i and g_i, and a derivative evaluation
one of their gradients. Each subproblem remembers its last input, so a repeated request at that
input is free (SLSQP asks for the objective and the constraint separately at the same point,
for example), and a derivative evaluation at a new input also counts an analysis. Every
evaluation ALBCD makes counts, including its optimality checks. The coupling constraints are
linear in the variables, so they need no model evaluations.
"""

import os
# one BLAS thread, for reproducible counts: the thread count changes SLSQP's rounding, and with
# it the path to the solution (at the baseline, 464 block solves with one thread, 484 without)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import modopt as mo
from albcd import ALBCD, Subproblem
import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))

# (N, n0, ni): one-at-a-time sweeps around the baseline (4, 10, 20)
CASES = sorted({(N, 10, 20) for N in (2, 4, 8, 16, 32)}
               | {(4, n0, 20) for n0 in (5, 10, 20, 40)}
               | {(4, 10, ni) for ni in (10, 20, 40, 80)})

# the same settings and tolerances for every case
SETTINGS = dict(max_mu=1e3, rho=1.2, tau=0.5, feas_tol=1e-6, opt_tol=[1e-2, 1e-5],
                max_outer_iter=200, max_inner_iter=100, save=False, verbose=False)


def solve_case(N, n0, ni):
    """Solve the scalable test problem with ALBCD and return its costs and solution quality."""
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

    def merit(v, x, y, mu, start):
        """Augmented Lagrangian over the block starting at x[start], with the other blocks fixed."""
        c = coupling(jax.lax.dynamic_update_slice(x, v, (start,)))
        return local_objective(v[:m]) + y @ c + 0.5 * mu @ c ** 2

    # compiled once and shared by all blocks: a block's start in x is an argument, so only the
    # two block sizes (with and without the slack) are compiled, whatever N is
    con = lambda v: local_constraint(v[:m])
    obj_fn, grad_fn = jax.jit(merit), jax.jit(jax.grad(merit))
    con_fn, jac_fn = jax.jit(con), jax.jit(jax.jacobian(con))

    counts = dict(solves=0, analyses=0, derivatives=0)

    class Block(Subproblem):

        def setup(self) -> None:
            self.add_input("x")
            self.add_input("y")  # Lagrange multipliers
            self.add_input("mu") # penalty parameters
            self.add_output("x")
            self.add_output("phi") # coupling constraints at the new x
            self.analyzed = self.differentiated = None # last model inputs

        def count(self, v, derivative=False):
            """Count an evaluation of this subproblem's model at v[:m] (the slack isn't a model input)."""
            key = np.asarray(v[:m]).tobytes()
            if key != self.analyzed:
                counts["analyses"] += 1
                self.analyzed = key
            if derivative and key != self.differentiated:
                counts["derivatives"] += 1
                self.differentiated = key

        def obj(self, v, *args):
            self.count(v)
            return np.float64(obj_fn(v, *args, self.index.start))

        def grad(self, v, *args):
            self.count(v, derivative=True)
            return np.array(grad_fn(v, *args, self.index.start))

        def con(self, v):
            self.count(v)
            return np.array(con_fn(v))

        def jac(self, v):
            self.count(v, derivative=True)
            return np.array(jac_fn(v))

        def solve(self, inputs, outputs) -> None:
            args = tuple(jnp.asarray(inputs[name]) for name in ("x", "y", "mu"))

            prob = mo.ProblemLite(x0=np.array(self.decompose(inputs["x"])),
                                  obj=lambda v: self.obj(v, *args),
                                  grad=lambda v: self.grad(v, *args),
                                  con=self.con, jac=self.jac,
                                  xl=xl[self.index], xu=xu[self.index], cl=-np.inf, cu=0.8 ** 2)
            # ftol bounds the objective's change, so v is accurate to only ~sqrt(ftol): 1e-10 left
            # an error floor near 1e-5 that stalled ALBCD for larger N
            optimizer = mo.SLSQP(prob, solver_options={'maxiter': 300, 'ftol': 1e-14}, turn_off_outputs=True)
            optimizer.solve()
            v = optimizer.results['x']
            counts["solves"] += 1

            # SLSQP's multiplier of 0.8^2 - c >= 0, and the constraint gradient, for residual()
            self._multiplier = np.asarray(optimizer.results['multipliers'])
            self._jac_con = self.jac(v)

            x_new = self.recompose(inputs["x"], v)
            outputs["x"] = x_new
            outputs["phi"] = np.array(coupling(jnp.asarray(x_new)))

        def residual(self, inputs) -> float:
            """Projected-gradient KKT residual, with the sphere constraint's multiplier and the bounds."""
            v = np.array(self.decompose(inputs["x"]))
            grad = self.grad(v, *(jnp.asarray(inputs[name]) for name in ("x", "y", "mu")))
            grad += self._jac_con.T @ self._multiplier
            return float(np.max(np.abs(v - np.clip(v - grad, xl[self.index], xu[self.index]))))

    blocks = [Block(slice(i * m, (i + 1) * m)) for i in range(N - 1)]
    blocks.append(Block(slice((N - 1) * m, N * m + 1))) # the last block also owns the slack

    x0 = np.zeros(N * m + 1)
    mu0 = np.ones((N - 1) * n0 + 1)

    # compile both block sizes before timing the solve
    t0 = time.perf_counter()
    args = (jnp.asarray(x0), jnp.asarray(mu0), jnp.asarray(mu0))
    for block in (blocks[0], blocks[-1]):
        v = x0[block.index]
        jax.block_until_ready((obj_fn(v, *args, block.index.start), grad_fn(v, *args, block.index.start),
                               con_fn(v), jac_fn(v)))
    compile_time = time.perf_counter() - t0

    opt = ALBCD(subproblems=blocks, x0=x0, mu0=mu0, **SETTINGS)
    opt.solve()
    costs = dict(counts) # before the optimality check below, which isn't part of the solve

    w = opt.x[:N * m].reshape(N, m)
    return dict(costs, wall_time=opt.tf, compile_time=compile_time, success=opt.success,
                objective=float(sum(local_objective(wi) for wi in w)),
                feasibility=float(np.max(np.abs(opt.phi))),
                optimality=max(block.residual(dict(x=opt.x, y=opt.y, mu=opt.mu)) for block in blocks))


results = []
for N, n0, ni in CASES:
    result = solve_case(N, n0, ni)
    results.append(dict(N=N, n0=n0, ni=ni, **result))
    print(f"N={N:3d} n0={n0:3d} ni={ni:3d} | success {result['success']!s:5} | solves {result['solves']:5d} | "
          f"analyses {result['analyses']:6d} | "
          f"derivatives {result['derivatives']:6d} | wall time {result['wall_time']:7.2f} s")

    # saved after every case, so a partial study is kept
    np.savez(os.path.join(HERE, "scalable_test_problem_scaling.npz"),
             **{key: np.array([r[key] for r in results]) for key in results[0]},
             settings=str(SETTINGS))
