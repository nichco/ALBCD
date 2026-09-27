import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from modopt import JaxProblem, SLSQP

N = 4     # number of subproblems
n0 = 10   # number of global variables
ni = 20   # number of local variables per subproblem (must be a multiple of n0)

m = ni // n0                  # local components driven by each global component
assert ni == m * n0
s = np.sqrt(m * N)            # global-variable scale factor
drv = np.arange(ni) // m      # drv[k] = global component driving local component k


def jaxobj(v):
    u = v[:n0][drv] / s                     # scaled driver for each local component, (ni,)
    xs = v[n0:].reshape(N, ni)
    terms = (1.0 - u) ** 2 + 100.0 * (xs - u ** 2) ** 2    # (N, ni) Rosenbrock pairs
    return jnp.sum(jnp.mean(terms, axis=1))                # sum over subproblems of f_i


def jaxcon(v):
    xs = v[n0:].reshape(N, ni)
    return 4.0 * jnp.mean(xs ** 2, axis=1) - 1.0           # N local hyperspheres, <= 0


v0 = np.zeros(n0 + N * ni)
xl = np.concatenate([np.full(n0, -1.5 * s), np.full(N * ni, -1.5)])
xu = -xl
cl = np.full(N, -np.inf)
cu = np.zeros(N)

jaxprob = JaxProblem(x0=v0, jax_obj=jaxobj, jax_con=jaxcon,
                     order=1, xl=xl, xu=xu, cl=cl, cu=cu)

optimizer = SLSQP(jaxprob, solver_options={'maxiter': 1000, 'ftol': 1e-16}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()

# compare with the closed-form solution
x = optimizer.results['x']
c = np.roots([400.0, 0.0, -198.0, -2.0]).real.max()   # u* = 0.708560
print('max |x0 - x0*|:', np.abs(x[:n0] - c * s).max())
print('max |xi - 1/2|:', np.abs(x[n0:] - 0.5).max())
print('f:', float(jaxobj(x)), ' f*:', N * ((1 - c) ** 2 + 100 * (0.5 - c ** 2) ** 2))
