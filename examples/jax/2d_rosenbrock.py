"""Unconstrained two-dimensional Rosenbrock function (with coefficient 1 instead of 100):

    min (1 - x1)^2 + (x2 - x1^2)^2

Block 1 owns x1 and block 2 owns x2. There are no constraints, so ALBCD runs only its
BCD inner loop (unconstrained=True). Each block solve lands on the curve where that
block is optimal (df/dx1 = 0 or df/dx2 = 0), so the iterates zigzag between the two
curves toward the minimum at (1, 1).

Gradients come from JAX. Run this file to solve the problem and plot the iterates.
"""

import numpy as np
import modopt as mo
import jax
jax.config.update("jax_enable_x64", True) # jax defaults to float32
import jax.numpy as jnp
import matplotlib.pyplot as plt
from albcd import ALBCD, Subproblem
import warnings
warnings.filterwarnings("ignore")


class Subproblem1(Subproblem):

    def setup(self) -> None:
        self.add_input("x")  # [x1, x2]
        self.add_output("x") # no "phi" output: there are no coupling constraints

    def objective(self, v, other):
        x1 = v[0]
        x2 = other[0]
        return jnp.squeeze((1 - x1)**2 + (x2 - x1**2)**2)

    def solve(self, inputs, outputs) -> None:
        x = inputs["x"]

        v0 = np.asarray(self.decompose(x), dtype=float)
        other = jnp.asarray(self.other(x))

        jax_obj = lambda v: self.objective(v, other)

        jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, order=1)

        # a tight ftol: near the solution a block solve can only lower f by about
        # grad^2 / (2 d2f/dv2), which falls below a looser ftol (e.g. 1e-7) before the
        # residual reaches opt_tol, so SLSQP would stop without moving and BCD would stall
        optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 100, 'ftol': 1e-12}, turn_off_outputs=True)
        optimizer.solve()

        outputs["x"] = self.recompose(x, optimizer.results['x'])

    def residual(self, inputs) -> float:

        x = inputs["x"]

        v = jnp.asarray(self.decompose(x))
        other = jnp.asarray(self.other(x))

        # v is unbounded, so the stationarity residual is just the gradient
        grad_f = np.asarray(jax.grad(self.objective)(v, other))
        return float(np.max(np.abs(grad_f)))


class Subproblem2(Subproblem):

    def setup(self) -> None:
        self.add_input("x")  # [x1, x2]
        self.add_output("x") # no "phi" output: there are no coupling constraints

    def objective(self, v, other):
        x2 = v[0]
        x1 = other[0]
        return jnp.squeeze((1 - x1)**2 + (x2 - x1**2)**2)

    def solve(self, inputs, outputs) -> None:
        x = inputs["x"]

        v0 = np.asarray(self.decompose(x), dtype=float)
        other = jnp.asarray(self.other(x))

        jax_obj = lambda v: self.objective(v, other)

        jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, order=1)

        optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 100, 'ftol': 1e-12}, turn_off_outputs=True)
        optimizer.solve()

        outputs["x"] = self.recompose(x, optimizer.results['x'])

    def residual(self, inputs) -> float:

        x = inputs["x"]

        v = jnp.asarray(self.decompose(x))
        other = jnp.asarray(self.other(x))

        # v is unbounded, so the stationarity residual is just the gradient
        grad_f = np.asarray(jax.grad(self.objective)(v, other))
        return float(np.max(np.abs(grad_f)))


opt = ALBCD(subproblems=[Subproblem1(index=slice(0, 1)),  # owns x1
                         Subproblem2(index=slice(1, 2))], # owns x2
            x0=np.array([-1.0, -1.0]),
            unconstrained=True,
            opt_tol=1e-4,
            max_inner_iter=300)

opt.solve()

print('Solution: ', opt.x)


history = np.array(opt.history)
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
