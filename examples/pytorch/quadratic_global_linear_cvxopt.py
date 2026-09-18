"""Same problem as quadratic_global_linear.py, with each block solved by CVXOPT using exact
Hessians.

Gradients come from PyTorch. Run this file to solve the problem and plot the iterates.
"""

import numpy as np
import modopt as mo
import torch
import matplotlib.pyplot as plt
from albcd import ALBCD, Subproblem
import warnings
warnings.filterwarnings("ignore")

# modopt and CVXOPT work in float64; torch defaults to float32, so widen the
# default dtype once here instead of annotating every tensor below.
torch.set_default_dtype(torch.float64)


class Subproblem1(Subproblem):

    def setup(self) -> None:
        self.add_input("x")  # [x1, s, x2]
        self.add_input("y")  # Lagrange multipliers
        self.add_input("mu") # penalty parameter(s)
        self.add_output("x")
        self.add_output("phi") # coupling constraints at the new x

        # Build the autograd transforms ONCE against `other`, `y`, `mu` as
        # ordinary arguments (not closure constants). This Subproblem instance
        # is reused for every BCD sweep, so the transformed callables are made
        # here and every later solve() just dispatches them -- instead of
        # rebuilding a Python closure and re-deriving the transform each time.
        self._obj_fn = self.objective
        self._grad_fn = torch.func.grad(self.objective, argnums=0)
        self._hess_fn = torch.func.hessian(self.objective, argnums=0)

    def objective(self, v, other, y, mu):
        x1 = v[0]
        s = v[1]
        x2 = other[0]
        obj = torch.squeeze(x1**2 + x2**2 - 1.5 * x1 * x2)

        g0 = -1 * (x1 - 0.5 * x2) + s

        return obj + torch.sum(y * g0) + 0.5 * torch.sum(mu * g0**2)

    def solve(self, inputs, outputs) -> None:
        x, y, mu = inputs["x"], inputs["y"], inputs["mu"]

        v0 = np.asarray(self.decompose(x), dtype=float)
        other = torch.as_tensor(self.other(x))
        y = torch.as_tensor(y)
        mu = torch.as_tensor(mu)

        xl = np.array([-np.inf, 0])
        xu = np.array([np.inf, np.inf])

        obj_fn = lambda v: float(self._obj_fn(torch.as_tensor(v), other, y, mu))
        grad_fn = lambda v: np.array(self._grad_fn(torch.as_tensor(v), other, y, mu))
        hess_fn = lambda v: np.array(self._hess_fn(torch.as_tensor(v), other, y, mu))

        prob = mo.ProblemLite(x0=v0, obj=obj_fn, grad=grad_fn, obj_hess=hess_fn,
                              xl=xl, xu=xu, name='subproblem1')

        optimizer = mo.CVXOPT(prob, solver_options={'maxiters': 100, 'abstol': 1e-9,
                                                        'reltol': 1e-8, 'feastol': 1e-9,
                                                        'show_progress': False}, turn_off_outputs=True)
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
        g0 = -1 * (x1 - 0.5 * x2) + s

        outputs["x"] = x_new
        outputs["phi"] = np.array([g0])

    def residual(self, inputs) -> float:

        x = inputs["x"]
        y = inputs["y"]
        mu = inputs["mu"]

        v = torch.as_tensor(self.decompose(x))
        other = torch.as_tensor(self.other(x))
        grad_f = torch.func.grad(self.objective)(v, other, torch.as_tensor(y), torch.as_tensor(mu))

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

        # See Subproblem1.setup(): build the transforms ONCE and reuse the
        # resulting callables across every BCD sweep.
        self._obj_fn = self.objective
        self._grad_fn = torch.func.grad(self.objective, argnums=0)
        self._hess_fn = torch.func.hessian(self.objective, argnums=0)

    def objective(self, v, other, y, mu):
        x2 = v[0]
        x1, s = other[0], other[1]
        obj = torch.squeeze(x1**2 + x2**2 - 1.5 * x1 * x2)

        g0 = -1 * (x1 - 0.5 * x2) + s

        return obj + torch.sum(y * g0) + 0.5 * torch.sum(mu * g0**2)

    def solve(self, inputs, outputs) -> None:
        x, y, mu = inputs["x"], inputs["y"], inputs["mu"]

        v0 = np.asarray(self.decompose(x), dtype=float)
        other = torch.as_tensor(self.other(x))
        y = torch.as_tensor(y)
        mu = torch.as_tensor(mu)

        obj_fn = lambda v: float(self._obj_fn(torch.as_tensor(v), other, y, mu))
        grad_fn = lambda v: np.array(self._grad_fn(torch.as_tensor(v), other, y, mu))
        hess_fn = lambda v: np.array(self._hess_fn(torch.as_tensor(v), other, y, mu))

        prob = mo.ProblemLite(x0=v0, obj=obj_fn, grad=grad_fn, obj_hess=hess_fn,
                              name='subproblem2')

        optimizer = mo.CVXOPT(prob, solver_options={'maxiters': 100, 'abstol': 1e-9,
                                                        'reltol': 1e-8, 'feastol': 1e-9,
                                                        'show_progress': False}, turn_off_outputs=True)
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
        g0 = -1 * (x1 - 0.5 * x2) + s

        outputs["x"] = x_new
        outputs["phi"] = np.array([g0])

    def residual(self, inputs) -> float:

        x = inputs["x"]
        y = inputs["y"]
        mu = inputs["mu"]

        v = torch.as_tensor(self.decompose(x))
        other = torch.as_tensor(self.other(x))
        grad_f = torch.func.grad(self.objective)(v, other, torch.as_tensor(y), torch.as_tensor(mu))

        v = np.asarray(v)
        grad_f = np.asarray(grad_f)

        # projected-gradient stationarity residual for box-constrained v:
        # zero iff v satisfies the KKT conditions for xl <= v <= xu
        proj = np.clip(v - grad_f, self._xl, self._xu)
        resid = v - proj
        return float(np.max(np.abs(resid)))


v_init = np.array([1.0, 0.0, -1.0])

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

plt.plot(x1_history, x2_history, 's-', color='tab:purple', linewidth=2.5, markersize=6, zorder=10, mec='k', label=r'$(x, y)$')
plt.plot(x1_history[-1], x2_history[-1], 's', color='#D462AD', markersize=8, zorder=11, mec='k')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x, 2*x, '--', color='black', linewidth=2, alpha=0.5)

plt.fill_between(x,
                 1.5,        # top of plot
                 2*x,        # constraint line
                 color='black',
                 alpha=0.4,
                 )

ticks = [-1, 0, 1]
plt.xticks(ticks)
plt.yticks(ticks)
# plt.legend()
plt.gca().set_aspect('equal')

plt.show()
