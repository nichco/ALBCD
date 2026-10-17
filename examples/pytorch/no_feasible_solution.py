# Distribution Statement A. Approved for public release: distribution is unlimited. Approved AFRL-2026-1671 28-09-2026.

"""Plain block coordinate descent on two coupling constraints that form a wedge:

    min x1^2 + x2^2 - 1.5 x1 x2  s.t.  x1 - 0.5 x2 >= 0,  x1 + 0.5 x2 >= 0

Block 1 owns x1 and block 2 owns x2. As in quadratic_global_failure.py, the coupling
constraints are handed to both blocks as local constraints, with the other block's
variable held fixed, so ALBCD runs only its BCD inner loop (unconstrained=True).
With x2 fixed, block 1 can always satisfy both constraints (x1 >= 0.5 |x2|), but
with x1 fixed, block 2 needs -2 x1 <= x2 <= 2 x1, which has no feasible solution
when x1 < 0. From x0 = (1, 0.5), every block solve stays inside the wedge, where
the constraints are inactive, so the iterates zigzag between the lines where each
block is optimal to the minimum at the origin, the vertex of the wedge.

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

    def objective(self, v, other):
        x1 = v[0]
        x2 = other[0]
        return torch.squeeze(x1**2 + x2**2 - 1.5 * x1 * x2)

    def local_constraints(self, v, other):
        x1 = v[0]
        x2 = other[0]
        return torch.stack([x1 - 0.5 * x2, x1 + 0.5 * x2])

    def solve(self, x, y, mu, data, outputs) -> None:
        v0 = np.asarray(self.decompose(x), dtype=float)
        other = torch.as_tensor(self.other(x))

        torch_obj = lambda v: self.objective(v, other)
        torch_con = lambda v: self.local_constraints(v, other)

        # modopt drives the solve with plain numpy callbacks, so torch.func
        # (the functional autograd API) supplies the exact objective gradient
        # and constraint Jacobian that ProblemLite would otherwise have to
        # finite-difference.
        obj_fn = lambda v: np.float64(torch_obj(torch.as_tensor(v)))
        grad_fn = lambda v: np.array(torch.func.grad(torch_obj)(torch.as_tensor(v)))
        con_fn = lambda v: np.array(torch_con(torch.as_tensor(v)))
        jac_fn = lambda v: np.array(torch.func.jacrev(torch_con)(torch.as_tensor(v)))

        prob = mo.ProblemLite(x0=v0, obj=obj_fn, grad=grad_fn, con=con_fn, jac=jac_fn,
                              cl=0., cu=np.inf, name='subproblem1')

        # a tight ftol: near the origin a block solve can only lower f by a tiny
        # amount, which falls below a looser ftol before the residual reaches opt_tol
        optimizer = mo.SLSQP(prob, solver_options={'maxiter': 100, 'ftol': 1e-12}, turn_off_outputs=True)
        optimizer.solve()
        v_new = optimizer.results['x']

        # Cache SLSQP's own converged multipliers and the constraint Jacobian at the
        # solution, so residual() can reuse them
        self._multipliers = optimizer.results['multipliers']
        self._jac_con = torch.func.jacrev(torch_con)(torch.as_tensor(v_new))

        outputs["x"] = self.recompose(x, v_new)

    def residual(self, x, y, mu, data) -> float:

        v = torch.as_tensor(self.decompose(x))
        other = torch.as_tensor(self.other(x))
        grad_f = torch.func.grad(self.objective)(v, other)

        grad_f = np.asarray(grad_f)
        jac_con = np.atleast_2d(np.asarray(self._jac_con))
        multipliers = np.asarray(self._multipliers)
        resid = grad_f - jac_con.T @ multipliers
        return float(np.max(np.abs(resid)))


class Subproblem2(Subproblem):

    def objective(self, v, other):
        x2 = v[0]
        x1 = other[0]
        return torch.squeeze(x1**2 + x2**2 - 1.5 * x1 * x2)

    def local_constraints(self, v, other):
        x2 = v[0]
        x1 = other[0]
        return torch.stack([x1 - 0.5 * x2, x1 + 0.5 * x2])

    def solve(self, x, y, mu, data, outputs) -> None:
        v0 = np.asarray(self.decompose(x), dtype=float)
        other = torch.as_tensor(self.other(x))

        torch_obj = lambda v: self.objective(v, other)
        torch_con = lambda v: self.local_constraints(v, other)

        # See Subproblem1.solve(): torch autograd supplies the exact gradient
        # and constraint Jacobian.
        obj_fn = lambda v: np.float64(torch_obj(torch.as_tensor(v)))
        grad_fn = lambda v: np.array(torch.func.grad(torch_obj)(torch.as_tensor(v)))
        con_fn = lambda v: np.array(torch_con(torch.as_tensor(v)))
        jac_fn = lambda v: np.array(torch.func.jacrev(torch_con)(torch.as_tensor(v)))

        prob = mo.ProblemLite(x0=v0, obj=obj_fn, grad=grad_fn, con=con_fn, jac=jac_fn,
                              cl=0., cu=np.inf, name='subproblem2')

        optimizer = mo.SLSQP(prob, solver_options={'maxiter': 100, 'ftol': 1e-12}, turn_off_outputs=True)
        optimizer.solve()
        v_new = optimizer.results['x']

        # Cache SLSQP's own converged multipliers and the constraint Jacobian at the
        # solution, so residual() can reuse them
        self._multipliers = optimizer.results['multipliers']
        self._jac_con = torch.func.jacrev(torch_con)(torch.as_tensor(v_new))

        outputs["x"] = self.recompose(x, v_new)

    def residual(self, x, y, mu, data) -> float:

        v = torch.as_tensor(self.decompose(x))
        other = torch.as_tensor(self.other(x))
        grad_f = torch.func.grad(self.objective)(v, other)

        grad_f = np.asarray(grad_f)
        jac_con = np.atleast_2d(np.asarray(self._jac_con))
        multipliers = np.asarray(self._multipliers)
        resid = grad_f - jac_con.T @ multipliers
        return float(np.max(np.abs(resid)))


opt = ALBCD(subproblems=[Subproblem1(index=slice(0, 1)),  # owns x1
                         Subproblem2(index=slice(1, 2))], # owns x2
            x0=np.array([1.0, 0.5]),
            unconstrained=True,
            opt_tol=1e-4,
            max_inner_iter=100)

opt.solve()

print('Solution: ', opt.x)


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

plt.plot(x1_history, x2_history, 'o-', color='tab:blue', linewidth=2.5, markersize=6, mec='k', zorder=10)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x, 2*x, '--', color='black', linewidth=2, alpha=0.5)
plt.plot(x, -2*x, '--', color='black', linewidth=2, alpha=0.5)

plt.fill_between(x,
                 1.5,        # top of plot
                 2*x,        # constraint line
                 color='black',
                 alpha=0.4,
                 )

plt.fill_between(x,
                 -1.5,       # bottom of plot
                 -2*x,       # constraint line
                 color='black',
                 alpha=0.4,
                 )

dZ_dx1 = 2 * X - 1.5 * Y
dZ_dx2 = 2 * Y - 1.5 * X

# zero level sets of the derivatives: the lines where block 1 and block 2 are optimal
# without the constraints
plt.contour(X, Y, dZ_dx1, levels=[0], colors='tab:purple', linewidths=2, linestyles='dotted', alpha=1)
plt.contour(X, Y, dZ_dx2, levels=[0], colors='tab:olive', linewidths=2, linestyles='dotted', alpha=1)

ticks = [-1, 0, 1]
plt.xticks(ticks)
plt.yticks(ticks)
plt.gca().set_aspect('equal')

plt.show()
