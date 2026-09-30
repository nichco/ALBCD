"""Cart-pole co-design under uncertainty with ALBCD.

Minimize the mean control effort of a cart-pole swing-up over N scenarios of the
uncertain gravity and friction, designing the pole's length l and mass mp together
with each scenario's trajectory. Each scenario is one block:

    ScenarioSubproblem i  owns [l_i, mp_i, states_i, u_i]  local constraints: collocated dynamics, boundary states

where [l_i, mp_i] is scenario i's copy of the pole design. The coupling constraints
phi force the copies to agree ([l_i, mp_i] = [l_{i+1}, mp_{i+1}]). Gradients come
from PyTorch. Run this file to solve the problem, compare the result with the
monolithic solution in monolithic_solution_N{N}.npz (generate it with monolithic.py),
and plot the convergence.
"""

import os
# SLSQP's matrices are small (about 150 x 150 per block), and OpenBLAS's multithreading
# costs more than it saves on them. Must be set before numpy/scipy load OpenBLAS.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import torch
import modopt as mo
import matplotlib.pyplot as plt
from albcd import ALBCD, Subproblem
import warnings
warnings.filterwarnings("ignore")

from models import N, samples, n, dt, nv, xl, xu, v0, x_scaler, c_scaler, effort, collocation

HERE = os.path.dirname(os.path.abspath(__file__))

# global x = [v_0, v_1, ..., v_{N-1}], where v_i = [l_i, mp_i, states_i, u_i]


def designs(x):
    """The N copies of the pole design [l, mp] in the global x."""
    return [torch.as_tensor(x[j * nv: j * nv + 2]) for j in range(N)]


def consensus(copies):
    """Coupling constraints: [l_j, mp_j] - [l_{j+1}, mp_{j+1}] for j = 0, ..., N-2."""
    return torch.cat([copies[j] - copies[j + 1] for j in range(N - 1)])


class ScenarioSubproblem(Subproblem):
    """Scenario i: owns [l_i, mp_i, states_i, u_i] and flies the swing-up under its own gravity and friction."""

    def __init__(self, i):
        self.i = i
        super().__init__(slice(i * nv, (i + 1) * nv))

    def setup(self) -> None:
        # torch.func (the functional autograd API) supplies the exact gradient and Jacobian
        self._grad = torch.func.grad(self.objective)
        self._jac = torch.func.jacfwd(self.local_constraints)

    def objective(self, v, x, y, mu):
        """Augmented Lagrangian as a function of this block's variables v."""
        copies = designs(x)
        copies[self.i] = v[:2]
        c = consensus(copies)

        # this scenario's share of the mean effort (the other scenarios' share is constant in this block)
        return 1e-2 * effort(v) / N + torch.sum(y * c) + 0.5 * torch.sum(mu * c**2)

    def local_constraints(self, v):
        return collocation(v, samples[self.i])

    def solve(self, x, y, mu, data, outputs) -> None:
        args = tuple(torch.as_tensor(a) for a in (x, y, mu))

        prob = mo.ProblemLite(x0=np.array(self.decompose(x)),
                              obj=lambda v: float(self.objective(torch.as_tensor(v), *args)),
                              grad=lambda v: self._grad(torch.as_tensor(v), *args).numpy(),
                              con=lambda v: self.local_constraints(torch.as_tensor(v)).numpy(),
                              jac=lambda v: self._jac(torch.as_tensor(v)).numpy(),
                              xl=xl, xu=xu, cl=0, cu=0, x_scaler=x_scaler, c_scaler=c_scaler)
        optimizer = mo.SLSQP(prob, solver_options={"maxiter": 1000, "ftol": 1e-9}, turn_off_outputs=True)
        optimizer.solve()
        v_new = optimizer.results["x"] / x_scaler

        # cache SLSQP's multipliers (rescaled from the c_scaler-scaled constraints SLSQP solves
        # with) and the local constraints' Jacobian at the solution for residual()
        self._multipliers = np.asarray(optimizer.results["multipliers"]) * c_scaler
        self._jac_con = self._jac(torch.as_tensor(v_new)).numpy()

        outputs["x"] = self.recompose(x, v_new)
        outputs["phi"] = consensus(designs(outputs["x"])).numpy()

    def residual(self, x, y, mu, data) -> float:
        args = tuple(torch.as_tensor(a) for a in (x, y, mu))
        v = self.decompose(x)
        grad_f = self._grad(torch.as_tensor(v), *args).numpy()

        # Lagrangian gradient (with SLSQP's multipliers for the local constraints) projected onto
        # the box bounds: zero iff v is a KKT point of this block's subproblem
        grad_L = grad_f - self._jac_con.T @ self._multipliers
        return float(np.max(np.abs(v - np.clip(v - grad_L, xl, xu))))


# every scenario starts from the same initial guess
opt = ALBCD(subproblems=[ScenarioSubproblem(i) for i in range(N)],
            x0=np.tile(v0, N),
            mu0=np.full(2 * (N - 1), 10.0),
            max_mu=1e6,
            rho=1.3,
            tau=0.5,
            feas_tol=1e-5,
            opt_tol=[1e-3, 1e-5],
            max_y=1e6,
            max_outer_iter=100,
            max_inner_iter=20)

opt.solve()


blocks = opt.x.reshape(N, nv)  # row i = v_i
efforts = [float(effort(torch.as_tensor(v))) for v in blocks]
for i, (g, mu_cart, mu_pole) in enumerate(samples):
    print(f"Scenario {i} (g {g:.3f}, mu_cart {mu_cart:.4f}, mu_pole {mu_pole:.4f}): "
          f"l {blocks[i, 0]:.4f} m, mp {blocks[i, 1]:.4f} kg, effort {efforts[i]:.2f}")
print(f"Mean control effort: {np.mean(efforts):.2f}")

# compare with the monolithic solution (the same problem solved with one SLSQP)
reference = os.path.join(HERE, f"monolithic_solution_N{N}.npz")
if os.path.exists(reference):
    solution = np.load(reference)
    x_star = np.concatenate([np.concatenate([[solution["l"], solution["mp"]], solution["states"][i].ravel(), solution["u"][i]])
                             for i in range(N)])
    error = np.linalg.norm(np.array(opt.x_history) - x_star, axis=1) / np.linalg.norm(x_star)
    print(f"Monolithic: l {float(solution['l']):.4f} m, mp {float(solution['mp']):.4f} kg, "
          f"effort {float(solution['effort']):.2f}")
    print("Relative error: ", error[-1])
else:
    error = None
    print(f"No monolithic solution for N = {N}; run monolithic.py to create it.")

# convergence data, for plotting without rerunning the solve. opt_history and feas_history have one
# entry per sweep; error has one per subproblem solve, and a leading entry for x0
np.savez(os.path.join(HERE, f"convergence_N{N}.npz"),
         opt_history=opt.opt_history, feas_history=opt.feas_history,
         error=np.array([]) if error is None else error,
         opt_tol=opt.opt_tol, feas_tol=opt.feas_tol, x=opt.x, success=opt.success, time=opt.tf)


t = dt * np.arange(n)
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(5, 5))
for i, v in enumerate(blocks):
    states, u = v[2:2 + 4 * n].reshape(4, n), v[2 + 4 * n:]
    ax[0].plot(t, states[0], label=f"Scenario {i}")
    ax[1].plot(t, states[1])
    ax[2].plot(t, u)
ax[0].set_ylabel("Cart position (m)")
ax[1].set_ylabel("Pole angle (rad)")
ax[2].set_ylabel("Cart force (N)")
ax[2].set_xlabel("Time (s)")
ax[0].legend()
fig.tight_layout()

fig, ax = plt.subplots(1, 2, figsize=(8, 2.5))
if error is not None:
    ax[0].semilogy(error, linewidth=2, color="tab:blue")
ax[0].set_xlabel("Subproblem solve")
ax[0].set_ylabel("Relative error")
ax[0].grid(color="lavender", alpha=0.5, axis="y")

ax[1].semilogy(opt.feas_history, linewidth=2, color="tab:orange")
ax[1].axhline(opt.feas_tol, color="gray", linewidth=1, linestyle="--", alpha=0.8)
ax[1].set_xlabel("Sweep")
ax[1].set_ylabel("Feasibility")
ax[1].grid(color="lavender", alpha=0.5, axis="y")

plt.tight_layout()
plt.show()
