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

optimizer = SLSQP(jaxprob, solver_options={'maxiter': 300, 'ftol': 1e-7}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()

x = optimizer.results['x']
print(x)