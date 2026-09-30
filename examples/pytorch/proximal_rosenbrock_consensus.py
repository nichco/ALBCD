# Distribution Statement A. Approved for public release: distribution is unlimited. Approved AFRL-2026-1671 28-09-2026.

"""Same problem as rosenbrock_consensus.py, with a proximal term 0.5 * tau * ||v - v_k||^2
added to each block's subproblem.

Gradients come from PyTorch. Run this file to solve the problem and plot the iterates.
"""

import numpy as np
import modopt as mo
import torch
import matplotlib.pyplot as plt
from albcd import ALBCD, Subproblem
import warnings
warnings.filterwarnings("ignore")

# modopt and SLSQP work in float64; torch defaults to float32, so widen the
# default dtype once here instead of annotating every tensor below.
torch.set_default_dtype(torch.float64)


class Subproblem1(Subproblem):

    tau = 1.0            # proximal weight for this block

    def objective(self, v, other, y, mu, v0):
        x1_1, x2_1 = v[0], v[1]
        x1_2, x2_2 = other[0], other[1]
        obj = torch.squeeze((1 - x1_1)**2 + 100 * (x2_1 - x1_1**2)**2)

        c_1 = x1_1 - x1_2
        c_2 = x2_1 - x2_2
        c = torch.stack([c_1, c_2])

        prox = 0.5 * self.tau * torch.sum((v - v0)**2)

        return obj + y.T @ c + 0.5 * c.T @ torch.diag(mu) @ c + prox

    def local_constraints(self, v):
        x1_1, x2_1 = v[0], v[1]
        con = x1_1**2 + x2_1**2
        return con.flatten()

    def solve(self, x, y, mu, data, outputs) -> None:
        v0 = np.asarray(self.decompose(x), dtype=float)
        other = torch.as_tensor(self.other(x))
        y = torch.as_tensor(y)
        mu = torch.as_tensor(mu)

        # x_i^k: this block's value at the start of this solve, held fixed
        # as the proximal-term anchor for both the solve and residual
        self._v0 = torch.as_tensor(v0).clone()

        torch_obj = lambda v: self.objective(v, other, y, mu, self._v0)

        # modopt drives the solve with plain numpy callbacks, so torch.func
        # (the functional autograd API) supplies the exact objective gradient
        # and constraint Jacobian that ProblemLite would otherwise have to
        # finite-difference.
        obj_fn = lambda v: np.float64(torch_obj(torch.as_tensor(v)))
        grad_fn = lambda v: np.array(torch.func.grad(torch_obj)(torch.as_tensor(v)))
        con_fn = lambda v: np.array(self.local_constraints(torch.as_tensor(v)))
        jac_fn = lambda v: np.array(torch.func.jacrev(self.local_constraints)(torch.as_tensor(v)))

        prob = mo.ProblemLite(x0=v0, obj=obj_fn, grad=grad_fn, con=con_fn, jac=jac_fn,
                              cl=0.5**2, cu=np.inf, name='subproblem1')

        optimizer = mo.SLSQP(prob, solver_options={'maxiter': 100, 'ftol': 1e-8}, turn_off_outputs=True)
        optimizer.solve()
        v_new = optimizer.results['x']

        # Cache SLSQP's own converged multipliers and this block's (other-
        # independent) constraint Jacobian at the solution, so residual() can reuse them
        self._multipliers = optimizer.results['multipliers']
        self._jac_con = torch.func.jacrev(self.local_constraints)(torch.as_tensor(v_new))

        x_new = self.recompose(x, v_new)

        # coupling constraints at the new x
        x1_1, x2_1, x1_2, x2_2 = x_new
        c_1 = x1_1 - x1_2
        c_2 = x2_1 - x2_2

        outputs["x"] = x_new
        outputs["phi"] = np.array([c_1, c_2])

    def residual(self, x, y, mu, data) -> float:

        v = torch.as_tensor(self.decompose(x))
        other = torch.as_tensor(self.other(x))
        grad_f = torch.func.grad(self.objective)(v, other, torch.as_tensor(y), torch.as_tensor(mu), self._v0)

        grad_f = np.asarray(grad_f)
        jac_con = np.atleast_2d(np.asarray(self._jac_con))
        multipliers = np.asarray(self._multipliers)
        resid = grad_f - jac_con.T @ multipliers
        return float(np.max(np.abs(resid)))


class Subproblem2(Subproblem):

    tau = 10.0            # proximal weight for this block

    def objective(self, v, other, y, mu, v0):
        x1_2, x2_2 = v[0], v[1]
        x1_1, x2_1 = other[0], other[1]
        obj = torch.squeeze((1 - x1_2)**2 + 100 * (x2_2 - x1_2**2)**2)

        c_1 = x1_1 - x1_2
        c_2 = x2_1 - x2_2
        c = torch.stack([c_1, c_2])

        prox = 0.5 * self.tau * torch.sum((v - v0)**2)

        return obj + y.T @ c + 0.5 * c.T @ torch.diag(mu) @ c + prox

    def local_constraints(self, v):
        x1_2, x2_2 = v[0], v[1]
        con = x1_2**2 + x2_2**2
        return con.flatten()

    def solve(self, x, y, mu, data, outputs) -> None:
        v0 = np.asarray(self.decompose(x), dtype=float)
        other = torch.as_tensor(self.other(x))
        y = torch.as_tensor(y)
        mu = torch.as_tensor(mu)

        # x_i^k: this block's value at the start of this solve, held fixed
        # as the proximal-term anchor for both the solve and residual
        self._v0 = torch.as_tensor(v0).clone()

        torch_obj = lambda v: self.objective(v, other, y, mu, self._v0)

        # See Subproblem1.solve(): torch autograd supplies the exact gradient
        # and constraint Jacobian.
        obj_fn = lambda v: np.float64(torch_obj(torch.as_tensor(v)))
        grad_fn = lambda v: np.array(torch.func.grad(torch_obj)(torch.as_tensor(v)))
        con_fn = lambda v: np.array(self.local_constraints(torch.as_tensor(v)))
        jac_fn = lambda v: np.array(torch.func.jacrev(self.local_constraints)(torch.as_tensor(v)))

        prob = mo.ProblemLite(x0=v0, obj=obj_fn, grad=grad_fn, con=con_fn, jac=jac_fn,
                              cl=0.5**2, cu=np.inf, name='subproblem2')

        optimizer = mo.SLSQP(prob, solver_options={'maxiter': 100, 'ftol': 1e-8}, turn_off_outputs=True)
        optimizer.solve()
        v_new = optimizer.results['x']

        # Cache SLSQP's own converged multipliers and this block's (other-
        # independent) constraint Jacobian at the solution, so residual() can reuse them
        self._multipliers = optimizer.results['multipliers']
        self._jac_con = torch.func.jacrev(self.local_constraints)(torch.as_tensor(v_new))

        x_new = self.recompose(x, v_new)

        # coupling constraints at the new x
        x1_1, x2_1, x1_2, x2_2 = x_new
        c_1 = x1_1 - x1_2
        c_2 = x2_1 - x2_2

        outputs["x"] = x_new
        outputs["phi"] = np.array([c_1, c_2])

    def residual(self, x, y, mu, data) -> float:

        v = torch.as_tensor(self.decompose(x))
        other = torch.as_tensor(self.other(x))
        grad_f = torch.func.grad(self.objective)(v, other, torch.as_tensor(y), torch.as_tensor(mu), self._v0)

        grad_f = np.asarray(grad_f)
        jac_con = np.atleast_2d(np.asarray(self._jac_con))
        multipliers = np.asarray(self._multipliers)
        resid = grad_f - jac_con.T @ multipliers
        return float(np.max(np.abs(resid)))


v_init = np.array([-0.5, 1.0, -0.5, 1.0])

opt = ALBCD(subproblems=[Subproblem1(index=slice(0, 2)), # owns [x1_1, x2_1]
                         Subproblem2(index=slice(2, 4))], # owns [x1_2, x2_2]
            x0=np.array(v_init),
            mu0=np.array([10.0, 10.0]),
            max_mu=1e3,
            rho=1.2,
            tau=0.5,
            feas_tol=1e-3,
            opt_tol=[1e-2, 1e-4],
            max_y=1e6,
            max_outer_iter=100,
            max_inner_iter=100)

opt.solve()


history = np.array(opt.x_history)
x1_1_history = history[:, 0]
x2_1_history = history[:, 1]
x1_2_history = history[:, 2]
x2_2_history = history[:, 3]

plt.rcParams.update({'font.size': 14})

x = np.linspace(-1.5, 1.5, 200)
y = np.linspace(-1.5, 1.5, 200)
X, Y = np.meshgrid(x, y)
Z = (1 - X)**2 + 100 * (Y - X**2)**2
levels = np.linspace(0, max(Z.flatten()), 30)
plt.contour(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.4, linewidths=0.5)
plt.contourf(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.5)

plt.plot(x1_1_history, x2_2_history, 's-', color='tab:orange', linewidth=2.5, markersize=6, zorder=10, mec='k', label=r'$(x_1, y_2)$')
plt.plot(x1_2_history, x2_1_history, 'o-', color='tab:purple', linewidth=2.5, markersize=6, zorder=10, mec='k', label=r'$(x_2, y_1)$')
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
plt.legend()
plt.gca().set_aspect('equal')

plt.show()
