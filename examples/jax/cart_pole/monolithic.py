"""Monolithic reference solution of the uncertain cart-pole problem in cart_pole.py.

The same problem, solved with a single SLSQP over one shared pole design [l, mp]
and every scenario's trajectory. The solution is saved to monolithic_solution_N{N}.npz,
which cart_pole.py compares the ALBCD solution against.
"""

import os
# SLSQP's matrices are small at moderate N (about 150N x 150N), and OpenBLAS's multithreading
# costs more than it saves on them. Must be set before numpy/scipy load OpenBLAS.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import jax
import jax.numpy as jnp
import modopt as mo
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from models import N, samples, n, dt, nv, xl, xu, v0, x_scaler, c_scaler, effort, collocation

HERE = os.path.dirname(os.path.abspath(__file__))

# z = [l, mp, trajectory_0, ..., trajectory_{N-1}], where trajectory_i = [states_i, u_i]
nt = nv - 2


def scenario(z, i):
    """Scenario i's variables [l, mp, states_i, u_i]."""
    return jnp.concatenate([z[:2], z[2 + i * nt: 2 + (i + 1) * nt]])


def objective(z):
    """Mean control effort over the scenarios."""
    return sum(effort(scenario(z, i)) for i in range(N)) / N


def constraints(z):
    return jnp.concatenate([collocation(scenario(z, i), samples[i]) for i in range(N)])


# jitted once, since SLSQP calls them at every iteration
obj, grad = jax.jit(objective), jax.jit(jax.grad(objective))
con, jac = jax.jit(constraints), jax.jit(jax.jacfwd(constraints))

tile = lambda a: np.concatenate([a[:2], np.tile(a[2:], N)])

problem = mo.ProblemLite(
    x0=tile(v0),
    obj=lambda z: float(obj(jnp.asarray(z))),
    grad=lambda z: np.asarray(grad(jnp.asarray(z))),
    con=lambda z: np.asarray(con(jnp.asarray(z))),
    jac=lambda z: np.asarray(jac(jnp.asarray(z))),
    xl=tile(xl), xu=tile(xu), cl=0, cu=0,
    x_scaler=tile(x_scaler), c_scaler=np.tile(c_scaler, N), o_scaler=1e-2,
)

optimizer = mo.SLSQP(problem, solver_options={"maxiter": 3000, "ftol": 1e-9}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()

z = optimizer.results["x"] / tile(x_scaler)
l, mp = z[0], z[1]
trajectories = z[2:].reshape(N, nt)
states = trajectories[:, :4 * n].reshape(N, 4, n)
u = trajectories[:, 4 * n:]

print(f"l: {l:.4f} m, mp: {mp:.4f} kg, mean control effort: {float(objective(z)):.2f}")

np.savez(os.path.join(HERE, f"monolithic_solution_N{N}.npz"),
         l=l, mp=mp, states=states, u=u, effort=float(objective(z)), samples=samples)


t = dt * np.arange(n)
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(5, 5))
for i in range(N):
    ax[0].plot(t, states[i, 0], label=f"Scenario {i}")
    ax[1].plot(t, states[i, 1])
    ax[2].plot(t, u[i])
ax[0].set_ylabel("Cart position (m)")
ax[1].set_ylabel("Pole angle (rad)")
ax[2].set_ylabel("Cart force (N)")
ax[2].set_xlabel("Time (s)")
ax[0].legend()
fig.tight_layout()
plt.show()
