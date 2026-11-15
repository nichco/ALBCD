# Distribution Statement A. Approved for public release: distribution is unlimited. Approved AFRL-2026-1671 28-09-2026.

"""Plain block coordinate descent on two coupling constraints that form a wedge:

    min x1^2 + x2^2 - 1.5 x1 x2  s.t.  x1 - 0.5 x2 >= 0,  x1 + 0.5 x2 >= 0

Block 1 owns x1 and block 2 owns x2. As in quadratic_global_failure.py, the coupling
constraints are handed to both blocks as local constraints, with the other block's
variable held fixed, and a plain cyclic block coordinate descent loop is written
out below. With x2 fixed, block 1 can always satisfy both constraints (x1 >= 0.5 |x2|), but
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
import warnings
warnings.filterwarnings("ignore")

# modopt and SLSQP work in float64; torch defaults to float32, so widen the
# default dtype once here instead of annotating every tensor below.
torch.set_default_dtype(torch.float64)


def f(x):
    return x[0]**2 + x[1]**2 - 1.5 * x[0] * x[1]


def con(x):
    """
    The two wedge constraints, con(x) >= 0.
    """
    return torch.stack([x[0] - 0.5 * x[1], x[0] + 0.5 * x[1]])


def minimize_block(x, i):
    """
    Minimizes f over x[i] subject to con >= 0, with the other variable fixed.
    Returns the new x[i] and SLSQP's multipliers.
    """
    xt = torch.tensor(x)
    put = lambda v: torch.cat([xt[:i], v, xt[i+1:]])
    obj = lambda v: f(put(v))
    c = lambda v: con(put(v))

    # modopt drives the solve with plain numpy callbacks, so torch.func
    # (the functional autograd API) supplies the exact gradient and Jacobian
    prob = mo.ProblemLite(x0=x[i:i+1].copy(),
                          obj=lambda v: np.float64(obj(torch.as_tensor(v))),
                          grad=lambda v: np.array(torch.func.grad(obj)(torch.as_tensor(v))),
                          con=lambda v: np.array(c(torch.as_tensor(v))),
                          jac=lambda v: np.array(torch.func.jacrev(c)(torch.as_tensor(v))),
                          cl=0., cu=np.inf)

    # a tight ftol: near the origin a block solve can only lower f by a tiny
    # amount, which falls below a looser ftol before the residual reaches tol
    optimizer = mo.SLSQP(prob, solver_options={'maxiter': 100, 'ftol': 1e-12}, turn_off_outputs=True)
    optimizer.solve()
    return optimizer.results['x'][0], optimizer.results['multipliers']


def stationarity(x, multipliers):
    """
    Each block's stationarity residual df/dx_i - (dcon/dx_i)^T lambda_i, with lambda_i from its last solve.
    """
    g = np.array(torch.func.grad(f)(torch.tensor(x)))
    J = np.array(torch.func.jacrev(con)(torch.tensor(x)))
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


history = bcd(np.array([1.0, 0.5]))

print('Solution: ', history[-1])

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
