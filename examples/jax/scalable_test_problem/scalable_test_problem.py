import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from modopt import JaxProblem, SLSQP

N = 4 # number of subproblems
n0 = 10 # number of global variables
ni = 20 # number of local variables


def jaxobj(v):

    x0 = v[:n0]
    xs = v[n0:].reshape(N, ni)
    w = jnp.hstack([jnp.tile(x0, (N, 1)), xs])

    terms = 100 * (w[:, 1:] - w[:, :-1] ** 2) ** 2 + (1.0 - w[:, :-1]) ** 2

    return jnp.mean(jnp.sum(terms, axis=1))


def jaxcon(v):

    x0 = v[:n0]
    xs = v[n0:].reshape(N, ni)
    w = jnp.hstack([jnp.tile(x0, (N, 1)), xs])

    spheres = jnp.sum(w ** 2, axis=1) / (n0 + ni)       # N local constraints
    plane = jnp.mean(xs[:, 0])      # 1 global constraint
    return jnp.concatenate([spheres, plane[None]])


x0 = np.ones(n0 + N * ni) * 0
xl, xu = -1.5, 1.5
cu = np.concatenate([np.full(N, 0.8 ** 2), [0.93]])
cl = np.full(N + 1, -np.inf)

jaxprob = JaxProblem(x0=x0, jax_obj=jaxobj, jax_con=jaxcon, 
                     order=1, xl=xl, xu=xu, cl=cl, cu=cu, o_scaler=1e-2)

# a tight ftol for a high quality reference solution: the subproblems are identical, so their
# local variables agree at the exact solution, and here they agree to ~1e-9 (~1e-7 with ftol=1e-14)
optimizer = SLSQP(jaxprob, solver_options={'maxiter': 1000, 'ftol': 1e-16}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()

x = optimizer.results['x']
print(x)

# saved for solution_error.py, which measures the ALBCD solution's error against it
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"monolithic_solution_N_{N}_n0_{n0}_ni_{ni}.npz"),
         x0=x[:n0], xs=x[n0:].reshape(N, ni), objective=float(jaxobj(x)))