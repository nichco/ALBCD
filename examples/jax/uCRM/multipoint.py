"""Multipoint aerodynamic design of the uCRM wing with ALBCD.

Minimize the average Breguet fuel burn of N missions flown by the same wing
over its twist distribution. Each mission is one block:

    MissionSubproblem i  owns [twist_cp_i, alpha_i]  local constraint: lift = weight

where twist_cp_i is mission i's copy of the wing's twist control points. The
coupling constraints phi force the copies to agree (twist_cp_i = twist_cp_{i+1}).
Gradients come from JAX. Run this file to solve the problem, compare the
result with the monolithic solution in monolithic_solution_N{N}.npz (generate
it with monolithic.py), and plot the convergence.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import modopt as mo
import matplotlib.pyplot as plt
from albcd import ALBCD, Subproblem
import warnings
warnings.filterwarnings("ignore")

from vlm_jax import geometry
from models import (N, problems, evaluate, fuelburn, memoize_last, mesh0, twist_cp0, alpha0, num_twist_cp,
                    twist_bounds, alpha_bounds)

HERE = os.path.dirname(os.path.abspath(__file__))

# global x = [twist_cp_0, alpha_0, twist_cp_1, alpha_1, ..., twist_cp_{N-1}, alpha_{N-1}]
block_size = num_twist_cp + 1


def twist_copies(x):
    """The N copies of the twist control points in the global x."""
    return [jnp.asarray(x[j * block_size: j * block_size + num_twist_cp]) for j in range(N)]


def consensus(copies):
    """Coupling constraints: twist_cp_j - twist_cp_{j+1} for j = 0, ..., N-2."""
    return jnp.concatenate([copies[j] - copies[j + 1] for j in range(N - 1)])


# each mission's latest fuel burn, and their average after every subproblem solve (for the convergence plot)
fuelburns = [fuelburn(i, twist_cp0, alpha0) for i in range(N)]
fuelburn_history = []


class MissionSubproblem(Subproblem):
    """Mission i: owns [twist_cp_i, alpha_i] and trims to lift = weight."""

    XL = np.concatenate([np.full(num_twist_cp, twist_bounds[0]), [alpha_bounds[0]]])
    XU = np.concatenate([np.full(num_twist_cp, twist_bounds[1]), [alpha_bounds[1]]])

    def __init__(self, i):
        self.i = i
        super().__init__(slice(i * block_size, (i + 1) * block_size))

    def setup(self) -> None:
        self.add_input("x")
        self.add_input("y")  # Lagrange multipliers
        self.add_input("mu") # penalty parameters
        self.add_output("x")
        self.add_output("phi") # coupling constraints at the new x

        # jit functions() and its Jacobian once. The inputs that change between solves (v, x,
        # y, mu) are arguments of the compiled functions rather than constants in a closure,
        # so every solve reuses the same compiled code instead of retracing the VLM
        self._values = jax.jit(self.functions)
        self._derivs = jax.jit(jax.jacrev(self.functions))

    def functions(self, v, x, y, mu):
        """[augmented Lagrangian, lift - weight] from a single VLM solve.

        Returned together so that one forward pass serves modopt's objective and
        constraint callbacks, and one reverse pass its gradient and Jacobian.
        """
        out = evaluate(problems[self.i], v[:num_twist_cp], v[-1])

        copies = twist_copies(x)
        copies[self.i] = v[:num_twist_cp]
        c = consensus(copies)

        # this mission's share of the average fuel burn (the other missions' share is constant in this block)
        aug_lagrangian = 1e-5 * out["fuelburn"] / N + jnp.sum(y * c) + 0.5 * jnp.sum(mu * c**2)
        return jnp.stack([aug_lagrangian, out["L_equals_W"]])

    def solve(self, inputs, outputs) -> None:
        x = inputs["x"]
        args = tuple(jnp.asarray(a) for a in (x, inputs["y"], inputs["mu"]))
        values = memoize_last(lambda v: np.asarray(self._values(jnp.asarray(v), *args)))
        derivs = memoize_last(lambda v: np.asarray(self._derivs(jnp.asarray(v), *args)))

        prob = mo.ProblemLite(x0=np.array(self.decompose(x)),
                              obj=lambda v: values(v)[0], grad=lambda v: derivs(v)[0],
                              con=lambda v: values(v)[1:], jac=lambda v: derivs(v)[1:],
                              xl=self.XL, xu=self.XU, cl=0, cu=0)
        optimizer = mo.SLSQP(prob, solver_options={"maxiter": 200, "ftol": 1e-8}, turn_off_outputs=True)
        optimizer.solve()
        v_new = optimizer.results["x"]

        # cache SLSQP's multiplier and the trim constraint's Jacobian at the solution for residual()
        self._multipliers = np.asarray(optimizer.results["multipliers"])
        self._jac_con = derivs(v_new)[1:]

        outputs["x"] = self.recompose(x, v_new)
        outputs["phi"] = np.asarray(consensus(twist_copies(outputs["x"])))

        fuelburns[self.i] = fuelburn(self.i, v_new[:num_twist_cp], v_new[-1])
        fuelburn_history.append(np.mean(fuelburns))

    def residual(self, inputs) -> float:
        x = inputs["x"]
        args = tuple(jnp.asarray(a) for a in (x, inputs["y"], inputs["mu"]))
        v = self.decompose(x)
        # row 0 of the compiled Jacobian is the augmented Lagrangian's gradient
        grad_f = np.asarray(self._derivs(jnp.asarray(v), *args))[0]

        # Lagrangian gradient (with SLSQP's multiplier for the trim constraint) projected onto
        # the box bounds: zero iff v is a KKT point of this block's subproblem
        grad_L = grad_f - self._jac_con.T @ self._multipliers
        return float(np.max(np.abs(v - np.clip(v - grad_L, self.XL, self.XU))))


# every mission starts from the jig twist and alpha0
x0 = np.concatenate([np.concatenate([twist_cp0, [alpha0]]) for _ in range(N)])

opt = ALBCD(subproblems=[MissionSubproblem(i) for i in range(N)],
            x0=x0,
            mu0=np.full((N - 1) * num_twist_cp, 0.002),
            max_mu=1e6,
            rho=1.2,
            tau=0.5,
            feas_tol=1e-4,
            opt_tol=[1e-3, 5e-4],
            max_y=1e6,
            max_outer_iter=100,
            max_inner_iter=7)

opt.solve()


blocks = opt.x.reshape(N, block_size)  # row i = [twist_cp_i, alpha_i]
for i in range(N):
    print(f"Mission {i}: alpha {blocks[i, -1]:.3f} deg, fuel burn {fuelburns[i]:.1f} kg")
print(f"Average fuel burn (kg): {np.mean(fuelburns):.1f}")

# compare with the monolithic solution (the same problem solved with one SLSQP). Compare the
# fuel burn rather than the design: a uniform twist change offset by the opposite change in
# alpha barely changes the fuel burn, so near-optimal designs can differ noticeably.
reference = os.path.join(HERE, f"monolithic_solution_N{N}.npz")
if os.path.exists(reference):
    solution = np.load(reference)
    f_star = float(solution["fuelburn"])
    error = np.abs(np.array(fuelburn_history) - f_star) / f_star
    print(f"Average fuel burn, monolithic (kg): {f_star:.1f}")
    print("Relative fuel burn error: ", error[-1])
else:
    solution, error = None, None
    print(f"No monolithic solution for N = {N}; run monolithic.py to create it.")


y_nodes = mesh0[0, :, 1]
b_twist = problems[0].b_twist
twist = lambda twist_cp: np.asarray(geometry.compute_twist(jnp.asarray(twist_cp, dtype=float), b_twist))

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(y_nodes, twist(twist_cp0), "o-", color="0.6", label="Initial (jig)")
for i in range(N):
    ax.plot(y_nodes, twist(blocks[i, :num_twist_cp]), "-", linewidth=2, label=f"Mission {i}")
if solution is not None:
    ax.plot(y_nodes, twist(solution["twist_cp"]), "k--", label="Monolithic")
ax.set_xlabel("Spanwise location, y (m)")
ax.set_ylabel("Twist (deg)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()

fig, ax = plt.subplots(1, 2, figsize=(8, 2.5))
if error is not None:
    ax[0].semilogy(error, linewidth=2, color="tab:blue")
ax[0].set_xlabel("Subproblem solve")
ax[0].set_ylabel("Fuel burn error")
ax[0].grid(color="lavender", alpha=0.5, axis="y")

ax[1].semilogy(opt.feas_history, linewidth=2, color="tab:orange")
ax[1].axhline(opt.feas_tol, color="gray", linewidth=1, linestyle="--", alpha=0.8)
ax[1].set_xlabel("Sweep")
ax[1].set_ylabel("Feasibility")
ax[1].grid(color="lavender", alpha=0.5, axis="y")

plt.tight_layout()
plt.show()
