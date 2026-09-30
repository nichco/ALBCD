"""ALBCD version of the scalable test problem in scalable_test_problem.py.

Each of the N subproblems owns a local copy z_i of the n0 global variables and its own ni
local variables, so its sphere constraint involves only its own variables and is handled
inside the block. The coupling constraints phi are

    z_i - z_{i+1} = 0,  i = 1, ..., N-1       (spanning-tree consensus, a chain)
    mean_i x_i[0] - 0.93 + s = 0              (global plane constraint, slack s >= 0)

and are enforced through the augmented Lagrangian. The last block also owns the slack.
"""

import numpy as np
import torch
torch.set_default_dtype(torch.float64) # modopt and SLSQP work in float64
import modopt as mo
from albcd import ALBCD, Subproblem
import warnings
warnings.filterwarnings("ignore")

N = 4 # number of subproblems
n0 = 10 # number of global variables
ni = 20 # number of local variables
m = n0 + ni # variables per subproblem: [z_i, x_i]

# x = [z_1, x_1, z_2, x_2, ..., z_N, x_N, s]
xl = np.full(N * m + 1, -1.5)
xu = np.full(N * m + 1, 1.5)
xl[-1], xu[-1] = 0.0, np.inf # slack


def local_objective(w):
    """Rosenbrock function of one subproblem's variables w = [z_i, x_i]."""
    return torch.sum(100 * (w[1:] - w[:-1] ** 2) ** 2 + (1.0 - w[:-1]) ** 2) / N


def local_constraint(w):
    """Sphere constraint of one subproblem, <= 0.8^2."""
    return torch.sum(w ** 2, dim=0, keepdim=True) / m


def coupling(x):
    """Consensus between neighboring copies of the global variables, then the global constraint."""
    w = x[:N * m].reshape(N, m)
    z, xs = w[:, :n0], w[:, n0:]
    plane = torch.mean(xs[:, 0]) - 0.93 + x[-1]
    return torch.cat([(z[:-1] - z[1:]).ravel(), plane[None]])


class Block(Subproblem):

    def merit(self, v, x, y, mu):
        """Augmented Lagrangian over this block's variables v, with the other blocks fixed."""
        c = coupling(torch.cat([x[:self.index.start], v, x[self.index.stop:]]))
        return local_objective(v[:m]) + y @ c + 0.5 * mu @ c ** 2

    def solve(self, x, y, mu, data, outputs) -> None:
        x, y, mu = (torch.as_tensor(a) for a in (x, y, mu))
        obj = lambda v: self.merit(v, x, y, mu)
        con = lambda v: local_constraint(v[:m])

        # modopt drives the solve with numpy callbacks; torch.func supplies the exact derivatives
        prob = mo.ProblemLite(x0=np.array(self.decompose(x)),
                              obj=lambda v: np.float64(obj(torch.as_tensor(v))),
                              grad=lambda v: np.array(torch.func.grad(obj)(torch.as_tensor(v))),
                              con=lambda v: np.array(con(torch.as_tensor(v))),
                              jac=lambda v: np.array(torch.func.jacrev(con)(torch.as_tensor(v))),
                              xl=xl[self.index], xu=xu[self.index], cl=-np.inf, cu=0.8 ** 2)
        optimizer = mo.SLSQP(prob, solver_options={'maxiter': 300, 'ftol': 1e-10}, turn_off_outputs=True)
        optimizer.solve()
        v = optimizer.results['x']

        # SLSQP's multiplier of 0.8^2 - c >= 0, and the constraint gradient, for residual()
        self._multiplier = np.asarray(optimizer.results['multipliers'])
        self._jac_con = np.array(torch.func.jacrev(con)(torch.as_tensor(v)))

        x_new = self.recompose(x, v)
        outputs["x"] = x_new
        outputs["phi"] = np.array(coupling(torch.as_tensor(x_new)))

    def residual(self, x, y, mu, data) -> float:
        """Projected-gradient KKT residual, with the sphere constraint's multiplier and the bounds."""
        x, y, mu = (torch.as_tensor(a) for a in (x, y, mu))
        v = np.array(self.decompose(x))
        grad = np.array(torch.func.grad(self.merit)(torch.as_tensor(v), x, y, mu))
        grad += self._jac_con.T @ self._multiplier
        return float(np.max(np.abs(v - np.clip(v - grad, xl[self.index], xu[self.index]))))


blocks = [Block(slice(i * m, (i + 1) * m)) for i in range(N - 1)]
blocks.append(Block(slice((N - 1) * m, N * m + 1))) # the last block also owns the slack

opt = ALBCD(subproblems=blocks,
            x0=np.zeros(N * m + 1),
            mu0=np.ones((N - 1) * n0 + 1),
            max_mu=1e3,
            rho=1.2,
            tau=0.5,
            feas_tol=1e-6,
            opt_tol=[1e-2, 1e-5],
            max_outer_iter=200,
            max_inner_iter=100)

opt.solve()

w = torch.as_tensor(opt.x[:N * m]).reshape(N, m)
print(f"Objective: {float(sum(local_objective(wi) for wi in w)):.6f}")
print(f"Sphere constraints: {[float(local_constraint(wi)[0]) for wi in w]} (<= {0.8 ** 2})")
print(f"Plane constraint: {float(torch.mean(w[:, n0])):.6f} (<= 0.93)")
print(f"Max consensus violation: {np.max(np.abs(opt.phi[:-1])):.2e}")
print(f"Global variables: {np.array(w[0, :n0])}")
