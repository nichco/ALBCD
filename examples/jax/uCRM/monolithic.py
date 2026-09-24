"""Monolithic reference solution of the multipoint problem in multipoint.py.

The same problem, solved with a single SLSQP over one shared set of twist
control points and every mission's angle of attack. The solution is saved to
monolithic_solution_N{N}.npz, which multipoint.py compares the ALBCD solution against.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import modopt as mo
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from vlm_jax import geometry
from models import (N, problems, evaluate, memoize_last, mesh0, twist_cp0, alpha0, num_twist_cp,
                    twist_bounds, alpha_bounds, payloads, ranges)

HERE = os.path.dirname(os.path.abspath(__file__))


def stacked(x):
    """[mean fuel burn, lift - weight of each mission] for x = [twist_cp, alpha_0, ..., alpha_{N-1}].

    Row 0 is the objective and the other rows are the constraints, so a single
    jacrev gives the gradient and the Jacobian together.
    """
    twist_cp, alphas = x[:num_twist_cp], x[num_twist_cp:]
    outs = [evaluate(p, twist_cp, a) for p, a in zip(problems, alphas)]
    fuelburn = jnp.stack([o["fuelburn"] for o in outs])
    l_equals_w = jnp.stack([o["L_equals_W"] for o in outs])
    return jnp.concatenate([fuelburn.mean().reshape(1), l_equals_w])


# jitted once, since SLSQP calls them at every iteration
stacked_jit = jax.jit(stacked)
jacobian_jit = jax.jit(jax.jacrev(stacked))
values = memoize_last(lambda x: np.asarray(stacked_jit(jnp.asarray(x))))
jacobian = memoize_last(lambda x: np.asarray(jacobian_jit(jnp.asarray(x))))

x0 = np.concatenate([twist_cp0, np.full(N, alpha0)])
xl = np.concatenate([np.full(num_twist_cp, twist_bounds[0]), np.full(N, alpha_bounds[0])])
xu = np.concatenate([np.full(num_twist_cp, twist_bounds[1]), np.full(N, alpha_bounds[1])])

problem = mo.ProblemLite(
    x0=x0,
    obj=lambda x: values(x)[0],
    grad=lambda x: jacobian(x)[0],
    con=lambda x: values(x)[1:],
    jac=lambda x: jacobian(x)[1:],
    xl=xl, xu=xu, cl=np.zeros(N), cu=np.zeros(N), o_scaler=1e-5,
)

optimizer = mo.SLSQP(problem, solver_options={"maxiter": 300, "ftol": 1e-9}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()

x_opt = optimizer.results["x"]
twist_cp_opt, alphas_opt = x_opt[:num_twist_cp], x_opt[num_twist_cp:]
twist_cp_opt_j = jnp.asarray(twist_cp_opt)

print(f"\n{'Mission':<10}{'Payload (kg)':>13}{'Range (km)':>12}{'alpha (deg)':>13}{'CL':>8}{'CD':>9}{'Fuel (kg)':>12}")
fuelburns = []
for i, (p, payload, R, a) in enumerate(zip(problems, payloads, ranges, alphas_opt)):
    out = evaluate(p, twist_cp_opt_j, a)
    fuelburns.append(float(out["fuelburn"]))
    print(f"{i:<10}{payload:>13.0f}{R / 1e3:>12.0f}{float(a):>13.3f}{float(out['CL']):>8.3f}"
          f"{float(out['CD']):>9.4f}{float(out['fuelburn']):>12.1f}")
print(f"\nAverage fuel burn (kg): {np.mean(fuelburns):.1f}")

np.savez(os.path.join(HERE, f"monolithic_solution_N{N}.npz"),
         twist_cp=twist_cp_opt, alphas=alphas_opt, fuelburn=np.mean(fuelburns))


b_twist = problems[0].b_twist
twist0 = np.asarray(geometry.compute_twist(jnp.asarray(twist_cp0, dtype=float), b_twist))
twist_opt = np.asarray(geometry.compute_twist(twist_cp_opt_j, b_twist))
y_nodes = mesh0[0, :, 1]

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(y_nodes, twist0, "o-", color="#00629B", label="Initial (jig)")
ax.plot(y_nodes, twist_opt, "o-", color="#C69214", label="Multipoint optimized")
ax.axhline(0, color="0.85", linewidth=1, zorder=0)
ax.set_xlabel("Spanwise location, y (m)")
ax.set_ylabel("Twist (deg)")
ax.set_title(f"uCRM twist distribution, {N}-point optimization")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
