# Distribution Statement A. Approved for public release: distribution is unlimited. Approved AFRL-2026-1671 28-09-2026.

"""Unconstrained two-dimensional Rosenbrock function (with coefficient 1 instead of 100):

    min (1 - x1)^2 + (x2 - x1^2)^2

Block 1 owns x1 and block 2 owns x2, and a plain cyclic block coordinate descent loop
is written out below. Each block solve lands on the curve where that block is optimal (df/dx1 = 0 or df/dx2 = 0), so the iterates zigzag between the two
curves toward the minimum at (1, 1).

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
    return (1 - x[0])**2 + (x[1] - x[0]**2)**2


def minimize_block(x, i):
    """
    Minimizes f over x[i] with the other variable fixed.
    """
    xj = jnp.asarray(x)
    prob = mo.JaxProblem(x0=x[i:i+1].copy(), jax_obj=lambda v: f(xj.at[i].set(v[0])), order=1)

    # a tight ftol: near the solution a block solve can only lower f by about
    # grad^2 / (2 d2f/dv2), which falls below a looser ftol (e.g. 1e-7) before the
    # gradient reaches tol, so SLSQP would stop without moving and BCD would stall
    optimizer = mo.SLSQP(prob, solver_options={'maxiter': 100, 'ftol': 1e-12}, turn_off_outputs=True)
    optimizer.solve()
    return optimizer.results['x'][0]


def bcd(x0, tol=1e-4, max_sweeps=300):
    """
    Cyclic block coordinate descent.
    """
    x = x0.copy()
    history = [x.copy()]
    for sweep in range(1, max_sweeps + 1):
        for i in range(len(x)):
            x[i] = minimize_block(x, i)
            history.append(x.copy())
        res = np.max(np.abs(jax.grad(f)(x)))
        print(f'sweep {sweep:3d} | max |gradient| {res:.3e}')
        if res <= tol:
            break
    return np.array(history)


history = bcd(np.array([-1.0, -1.0]))

print('Solution: ', history[-1])

x1_history = history[:, 0]
x2_history = history[:, 1]

plt.figure(figsize=(4, 4))

x = np.linspace(-1.5, 1.5, 200)
y = np.linspace(-1.5, 1.5, 200)
X, Y = np.meshgrid(x, y)
Z = (1 - X)**2 + (Y - X**2)**2
levels = np.linspace(0, max(Z.flatten()), 30)
plt.contour(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.4, linewidths=0.5)
plt.contourf(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.5)

dZ_dx1 = -2 * (1 - X) - 4 * X * (Y - X**2)
dZ_dx2 = 2 * (Y - X**2)

# zero level sets of the derivatives: the curves where block 1 and block 2 are optimal
plt.contour(X, Y, dZ_dx1, levels=[0], colors='tab:purple', linewidths=2, linestyles='-.', alpha=1)
plt.contour(X, Y, dZ_dx2, levels=[0], colors='tab:olive', linewidths=2, linestyles='-.', alpha=1)

plt.plot(x1_history, x2_history, '-o', mec='k', color='tab:red', linewidth=2.5, markersize=7, zorder=10)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xlabel('x')
plt.ylabel('y')

ticks = [-1, 0, 1]
plt.xticks(ticks)
plt.yticks(ticks)

plt.show()
