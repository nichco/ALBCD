"""Quadratic objective with a nonlinear inequality coupling two blocks:

    min x1^2 + x2^2 - 1.5 x1 x2  s.t.  x1^2 + x2^2 >= 0.25

Block 1 owns (x1, s), where s >= 0 is the slack, and block 2 owns x2.

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
        self.add_input("x")  # [x1, s, x2]
        self.add_input("y")  # Lagrange multipliers
        self.add_input("mu") # penalty parameter(s)
        self.add_output("x")
        self.add_output("phi") # coupling constraints at the new x

    def objective(self, v, other, y, mu):
        x1 = v[0]
        s = v[1]
        x2 = other[0]
        obj = jnp.squeeze(x1**2 + x2**2 - 1.5 * x1 * x2)

        g0 = -1 * (x1**2 + x2**2) + 0.5**2 + s

        return obj + jnp.sum(y * g0) + 0.5 * jnp.sum(mu * g0**2)

    def solve(self, inputs, outputs) -> None:
        x, y, mu = inputs["x"], inputs["y"], inputs["mu"]

        v0 = np.asarray(self.decompose(x), dtype=float)
        other = jnp.asarray(self.other(x))

        xl = np.array([-np.inf, 0])
        xu = np.array([np.inf, np.inf])

        jax_obj = lambda v: self.objective(v, other, y, mu)

        jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, xl=xl, xu=xu)

        optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 100, 'ftol': 1e-8}, turn_off_outputs=True)
        optimizer.solve()
        # optimizer.print_results()
        v_new = optimizer.results['x']

        # No general (in)equality constraints at the block level -- only box
        # bounds -- so cache the bounds for the projected-gradient stationarity
        # check in residual() instead of constraint multipliers/Jacobian.
        self._xl = xl
        self._xu = xu

        x_new = self.recompose(x, v_new)

        # coupling constraint at the new x
        x1, s, x2 = x_new
        g0 = -1 * (x1**2 + x2**2) + 0.5**2 + s

        outputs["x"] = x_new
        outputs["phi"] = np.array([g0])

    def residual(self, inputs) -> float:

        x = inputs["x"]
        y = inputs["y"]
        mu = inputs["mu"]

        v = jnp.asarray(self.decompose(x))
        other = jnp.asarray(self.other(x))
        grad_f = jax.grad(self.objective)(v, other, y, mu)

        v = np.asarray(v)
        grad_f = np.asarray(grad_f)

        # projected-gradient stationarity residual for box-constrained v:
        # zero iff v satisfies the KKT conditions for xl <= v <= xu
        proj = np.clip(v - grad_f, self._xl, self._xu)
        resid = v - proj
        return float(np.max(np.abs(resid)))


class Subproblem2(Subproblem):

    def setup(self) -> None:
        self.add_input("x")  # [x1, s, x2]
        self.add_input("y")  # Lagrange multipliers
        self.add_input("mu") # penalty parameter(s)
        self.add_output("x")
        self.add_output("phi") # coupling constraints at the new x

    def objective(self, v, other, y, mu):
        x2 = v[0]
        x1, s = other[0], other[1]
        obj = jnp.squeeze(x1**2 + x2**2 - 1.5 * x1 * x2)

        g0 = -1 * (x1**2 + x2**2) + 0.5**2 + s
        
        return obj + jnp.sum(y * g0) + 0.5 * jnp.sum(mu * g0**2)

    def solve(self, inputs, outputs) -> None:
        x, y, mu = inputs["x"], inputs["y"], inputs["mu"]

        v0 = np.asarray(self.decompose(x), dtype=float)
        other = jnp.asarray(self.other(x))

        jax_obj = lambda v: self.objective(v, other, y, mu)

        jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj)

        optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 100, 'ftol': 1e-8}, turn_off_outputs=True)
        optimizer.solve()
        # optimizer.print_results()
        v_new = optimizer.results['x']

        # No box bounds on this block either -- record unbounded limits so
        # residual() can use the same projected-gradient check.
        self._xl = np.full_like(v_new, -np.inf)
        self._xu = np.full_like(v_new, np.inf)

        x_new = self.recompose(x, v_new)

        # coupling constraint at the new x
        x1, s, x2 = x_new
        g0 = -1 * (x1**2 + x2**2) + 0.5**2 + s

        outputs["x"] = x_new
        outputs["phi"] = np.array([g0])

    def residual(self, inputs) -> float:

        x = inputs["x"]
        y = inputs["y"]
        mu = inputs["mu"]

        v = jnp.asarray(self.decompose(x))
        other = jnp.asarray(self.other(x))
        grad_f = jax.grad(self.objective)(v, other, y, mu)

        v = np.asarray(v)
        grad_f = np.asarray(grad_f)

        # projected-gradient stationarity residual for box-constrained v:
        # zero iff v satisfies the KKT conditions for xl <= v <= xu
        proj = np.clip(v - grad_f, self._xl, self._xu)
        resid = v - proj
        return float(np.max(np.abs(resid)))


v_init = np.array([-0.5, 0.0, 1.0])
# v_init = np.array([-0.5, 0.2, 1.0])

opt = ALBCD(subproblems=[Subproblem1(index=slice(0, 2)), # owns x1 and s
                         Subproblem2(index=slice(2, 3))], # owns x2
            x0=np.array(v_init),
            mu0=np.array([1.0]),
            max_mu=1e3,
            rho=1.2,
            tau=0.5,
            feas_tol=1e-3,
            opt_tol=[1e-2, 1e-4],
            max_y=1e6,
            max_outer_iter=100,
            max_inner_iter=100)

opt.solve()


history = np.array(opt.history)
x1_history = history[:, 0]
s_history = history[:, 1]
x2_history = history[:, 2]


plt.rcParams.update({'font.size': 14})

x = np.linspace(-1.5, 1.5, 200)
y = np.linspace(-1.5, 1.5, 200)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2 - 1.5 * X * Y
levels = np.linspace(0, max(Z.flatten()), 30)
plt.contour(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.4, linewidths=0.5)
plt.contourf(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.5)

plt.plot(x1_history, x2_history, 's-', color='tab:orange', linewidth=2.5, markersize=6, zorder=10, mec='k', label=r'$(x, y)$')
plt.plot(x1_history[-1], x2_history[-1], 's', color='#D462AD', markersize=8, zorder=11, mec='k')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xlabel('x')
plt.ylabel('y')

theta = np.linspace(0, 2*np.pi, 100)
circle_x = 0.5 * np.cos(theta)
circle_y = 0.5 * np.sin(theta)
plt.plot(circle_x, circle_y, '--', color='black', linewidth=2, alpha=0.5)
plt.fill(circle_x, circle_y, color='black', alpha=0.3)

ticks = [-1, 0, 1]
plt.xticks(ticks)
plt.yticks(ticks)
# plt.legend()
plt.gca().set_aspect('equal')

plt.show()