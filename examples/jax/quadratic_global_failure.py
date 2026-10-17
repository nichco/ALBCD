# Distribution Statement A. Approved for public release: distribution is unlimited. Approved AFRL-2026-1671 28-09-2026.

"""Plain block coordinate descent on a coupling constraint, and why it fails:

    min x1^2 + x2^2 - 1.5 x1 x2  s.t.  x1 - 0.5 x2 >= 0

Block 1 owns x1 and block 2 owns x2. Instead of entering an augmented Lagrangian,
the coupling constraint is handed to both blocks as a local constraint, with the
other block's variable held fixed, so ALBCD runs only its BCD inner loop
(unconstrained=True). From x0 = (1, -1), the first block solve lands on the
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
from albcd import ALBCD, Subproblem
import warnings
warnings.filterwarnings("ignore")


class Subproblem1(Subproblem):

    def objective(self, v, other):
        x1 = v[0]
        x2 = other[0]
        return jnp.squeeze(x1**2 + x2**2 - 1.5 * x1 * x2)

    def local_constraints(self, v, other):
        x1 = v[0]
        x2 = other[0]
        con = x1 - 0.5 * x2
        return con.flatten()

    def solve(self, x, y, mu, data, outputs) -> None:
        v0 = np.asarray(self.decompose(x), dtype=float)
        other = jnp.asarray(self.other(x))

        jax_obj = lambda v: self.objective(v, other)
        jax_con = lambda v: self.local_constraints(v, other)

        jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, jax_con=jax_con, cl=0., cu=np.inf, order=1)

        optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 100, 'ftol': 1e-8}, turn_off_outputs=True)
        optimizer.solve()
        v_new = optimizer.results['x']

        # Cache SLSQP's own converged multipliers and the constraint Jacobian at the
        # solution, so residual() can reuse them
        self._multipliers = optimizer.results['multipliers']
        self._jac_con = jax.jacobian(jax_con)(jnp.asarray(v_new))

        outputs["x"] = self.recompose(x, v_new)

    def residual(self, x, y, mu, data) -> float:

        v = jnp.asarray(self.decompose(x))
        other = jnp.asarray(self.other(x))
        grad_f = jax.grad(self.objective)(v, other)

        grad_f = np.asarray(grad_f)
        jac_con = np.atleast_2d(np.asarray(self._jac_con))
        multipliers = np.asarray(self._multipliers)
        resid = grad_f - jac_con.T @ multipliers
        return float(np.max(np.abs(resid)))


class Subproblem2(Subproblem):

    def objective(self, v, other):
        x2 = v[0]
        x1 = other[0]
        return jnp.squeeze(x1**2 + x2**2 - 1.5 * x1 * x2)

    def local_constraints(self, v, other):
        x2 = v[0]
        x1 = other[0]
        con = x1 - 0.5 * x2
        return con.flatten()

    def solve(self, x, y, mu, data, outputs) -> None:
        v0 = np.asarray(self.decompose(x), dtype=float)
        other = jnp.asarray(self.other(x))

        jax_obj = lambda v: self.objective(v, other)
        jax_con = lambda v: self.local_constraints(v, other)

        jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, jax_con=jax_con, cl=0., cu=np.inf, order=1)

        optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 100, 'ftol': 1e-8}, turn_off_outputs=True)
        optimizer.solve()
        v_new = optimizer.results['x']

        # Cache SLSQP's own converged multipliers and the constraint Jacobian at the
        # solution, so residual() can reuse them
        self._multipliers = optimizer.results['multipliers']
        self._jac_con = jax.jacobian(jax_con)(jnp.asarray(v_new))

        outputs["x"] = self.recompose(x, v_new)

    def residual(self, x, y, mu, data) -> float:

        v = jnp.asarray(self.decompose(x))
        other = jnp.asarray(self.other(x))
        grad_f = jax.grad(self.objective)(v, other)

        grad_f = np.asarray(grad_f)
        jac_con = np.atleast_2d(np.asarray(self._jac_con))
        multipliers = np.asarray(self._multipliers)
        resid = grad_f - jac_con.T @ multipliers
        return float(np.max(np.abs(resid)))


opt = ALBCD(subproblems=[Subproblem1(index=slice(0, 1)),  # owns x1
                         Subproblem2(index=slice(1, 2))], # owns x2
            x0=np.array([1.0, -1.0]),
            unconstrained=True,
            opt_tol=1e-4,
            max_inner_iter=100)

opt.solve()

print('Solution: ', opt.x, '(the optimum is the origin)')


history = np.array(opt.x_history)
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
