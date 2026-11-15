# Distribution Statement A. Approved for public release: distribution is unlimited. Approved AFRL-2026-1671 28-09-2026.

"""Plain block coordinate descent on a coupling constraint, and why it fails:

    min x1^2 + x2^2 - 1.5 x1 x2  s.t.  x1 - 0.5 x2 >= 0

Block 1 owns x1 and block 2 owns x2. Instead of entering an augmented Lagrangian,
the coupling constraint is handed to both blocks as a local constraint, with the
other block's variable held fixed, and a plain cyclic block coordinate descent loop
is written out below. From x0 = (1, -1), the first block solve lands on the
constraint line at (-0.5, -1), where each block is optimal with the other fixed:
lowering x1 or raising x2 would leave the feasible set. BCD stops there with every
block residual at zero, although the optimum is the origin, further along the
constraint line. quadratic_global_linear.py solves the same problem with ALBCD.

Gradients come from JAX. Run this file to solve the problem and plot the iterates.
"""

import numpy as np
import modopt as mo
import jax
jax.config.update("jax_enable_x64", True) # jax defaults to float32
import jax.numpy as jnp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def f(x):
    return x[0]**2 + x[1]**2 - 1.5 * x[0] * x[1]


def con(x):
    """
    The coupling constraint, con(x) >= 0.
    """
    return jnp.array([x[0] - 0.5 * x[1]])


def minimize_block(x, i):
    """
    Minimizes f over x[i] subject to con >= 0, with the other variable fixed.
    Returns the new x[i] and SLSQP's multipliers.
    """
    xj = jnp.asarray(x)
    put = lambda v: jnp.concatenate([xj[:i], v, xj[i+1:]])
    prob = mo.JaxProblem(x0=x[i:i+1].copy(), jax_obj=lambda v: f(put(v)), jax_con=lambda v: con(put(v)),
                         cl=0., cu=np.inf, order=1)
    optimizer = mo.SLSQP(prob, solver_options={'maxiter': 100, 'ftol': 1e-8}, turn_off_outputs=True)
    optimizer.solve()
    return optimizer.results['x'][0], optimizer.results['multipliers']


def stationarity(x, multipliers):
    """
    Each block's stationarity residual df/dx_i - (dcon/dx_i)^T lambda_i, with lambda_i from its last solve.
    """
    g = np.asarray(jax.grad(f)(x))
    J = np.asarray(jax.jacobian(con)(x))
    return np.array([g[i] - J[:, i] @ multipliers[i] for i in range(len(x))])


def bcd(x0, tol=1e-4, max_sweeps=100):
    """
    Cyclic block coordinate descent.
    """
    x = x0.copy()
    history = [x.copy()]
    multipliers = [None] * len(x)
    for sweep in range(1, max_sweeps + 1):
        for i in range(len(x)):
            x[i], multipliers[i] = minimize_block(x, i)
            history.append(x.copy())
        res = np.max(np.abs(stationarity(x, multipliers)))
        print(f'sweep {sweep:3d} | max stationarity residual {res:.3e}')
        if res <= tol:
            break
    return np.array(history)


history = bcd(np.array([1.0, -1.0]))

print('Solution: ', history[-1], '(the optimum is the origin)')

x1_history = history[:, 0]
x2_history = history[:, 1]

plt.rcParams.update({'font.size': 14})

x = np.linspace(-1.5, 1.5, 200)
y = np.linspace(-1.5, 1.5, 200)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2 - 1.5 * X * Y
levels = np.linspace(0, max(Z.flatten()), 30)
plt.contour(X, Y, Z, levels=levels, cmap='Greens_r', alpha=0.4, linewidths=0.5)
plt.contourf(X, Y, Z, levels=levels, cmap='Greens_r', alpha=0.5)

plt.plot(x1_history, x2_history, 'o-', color='tab:red', linewidth=2.5, markersize=6, mec='k', zorder=10)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x, 2*x, '--', color='k', linewidth=2, alpha=0.5)

plt.fill_between(x,
                 1.5,        # top of plot
                 2*x,        # constraint line
                 color='k',
                 alpha=0.4,
                 )

dZ_dx1 = 2 * X - 1.5 * Y
dZ_dx2 = 2 * Y - 1.5 * X

# zero level sets of the derivatives: the lines where block 1 and block 2 are optimal
# without the constraint
plt.contour(X, Y, dZ_dx1, levels=[0], colors='tab:purple', linewidths=2, linestyles='dotted', alpha=1)
plt.contour(X, Y, dZ_dx2, levels=[0], colors='tab:olive', linewidths=2, linestyles='dotted', alpha=1)

ticks = [-1, 0, 1]
plt.xticks(ticks)
plt.yticks(ticks)
plt.gca().set_aspect('equal')

plt.show()
