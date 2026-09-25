"""Monolithic solution of Sobieski's supersonic business jet (SSBJ) problem, in the SAND form.

The simultaneous analysis and design (SAND) formulation gives the optimizer every
disciplinary state alongside the 10 design variables, and makes the disciplines' equations
equality constraints, so the disciplines are never solved, only satisfied at the solution.
The SSBJ disciplines are explicit, so their states are their outputs,

    structure    -> total weight, fuel weight, wing twist
    aerodynamics -> drag, lift-to-drag ratio
    propulsion   -> specific fuel consumption (SFC), engine weight, engine scale factor (ESF)

with residuals R(z, s) = s - F(z, s), where F evaluates each discipline's equations from the
states s. Unlike IDF there are no copies: each quantity is one variable, used by every
discipline that needs it, including the states that don't couple the disciplines (fuel weight,
lift-to-drag ratio and SFC). The range comes directly from the states. The problem and its
disciplines are in models.py.
"""

import numpy as np
import jax.numpy as jnp
import modopt as mo
import warnings
warnings.filterwarnings("ignore")

from models import (x0, xl, xu, structure, aerodynamics, propulsion, breguet_range, design_constraints,
                    cl, cu, report)

# states s, starting from rough guesses that don't satisfy the residuals. Their bounds are
# loose, only keeping the models defined during the search. The total weight is at least
# 27000 lb (the miscellaneous and minimum fuel weights), and 25000 lb is above the fuel weight
# of every design that satisfies the pressure gradient constraint (t/c <= 0.06), so the
# Breguet logarithm stays defined without cutting into the feasible designs.
state_names = ["total weight", "fuel weight", "twist", "drag", "lift/drag", "SFC", "engine weight", "ESF"]
s0 = np.array([50000.0, 7000.0, 1.0, 12000.0, 4.0, 1.1, 6000.0, 0.5])
sl = np.array([27000.0, 2000.0, 0.5, 1e3, 0.1, 0.1, 1e3, 0.1])
su = np.array([2e5, 25000.0, 1.5, 1e5, 20.0, 5.0, 3e4, 2.0])
nz = len(x0)


def equations(v):
    """Each discipline's equations evaluated from the states, for v = [z, s].

    Returns F(z, s) in the order of s, and the other outputs.
    """
    z, s = v[:nz], v[nz:]
    x_shared, x_1, cf, throttle = z[:6], z[6:8], z[8], z[9]
    total_weight, fuel_weight, twist, drag, lift_to_drag, sfc, engine_weight, esf = s
    # the lift equals the total weight
    total_weight_f, fuel_weight_f, twist_f, stress = structure(x_shared, x_1, total_weight, engine_weight)
    _, drag_f, lift_to_drag_f, pressure_gradient = aerodynamics(x_shared, cf, total_weight, twist, esf)
    sfc_f, engine_weight_f, esf_f, temperature, throttle_ratio = propulsion(x_shared, throttle, drag)
    F = jnp.stack([total_weight_f, fuel_weight_f, twist_f, drag_f, lift_to_drag_f, sfc_f, engine_weight_f, esf_f])
    return F, dict(stress=stress, pressure_gradient=pressure_gradient, temperature=temperature,
                   throttle_ratio=throttle_ratio)


def objective(v):
    total_weight, fuel_weight, _, _, lift_to_drag, sfc, _, _ = v[nz:]
    return -breguet_range(v[:6], total_weight, fuel_weight, lift_to_drag, sfc)


def constraints(v):
    """[design constraints (10), residuals (8)], with the residuals relative to the states' initial guesses."""
    s = v[nz:]
    F, out = equations(v)
    twist, esf = s[2], s[7]
    return jnp.concatenate([design_constraints(out["stress"], twist, out["pressure_gradient"], esf,
                                               out["throttle_ratio"], out["temperature"]), (s - F) / s0])


n_residuals = len(s0)
# JaxProblem jits the objective, the constraints and their derivatives (jax.grad, jax.jacrev)
prob = mo.JaxProblem(x0=np.concatenate([x0, s0]), jax_obj=objective, jax_con=constraints,
                     xl=np.concatenate([xl, sl]), xu=np.concatenate([xu, su]),
                     cl=np.concatenate([cl, np.zeros(n_residuals)]), cu=np.concatenate([cu, np.zeros(n_residuals)]),
                     x_scaler=np.concatenate([1 / (xu - xl), 1 / s0]), o_scaler=1e-3)

optimizer = mo.SLSQP(prob, solver_options={"maxiter": 200, "ftol": 1e-12}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()

v_star = optimizer.results["x"] / np.concatenate([1 / (xu - xl), 1 / s0])
z_star = v_star[:nz]
g = np.asarray(constraints(v_star))
report(z_star, -float(objective(v_star)), g[:-n_residuals])

print(f"\n{'State':<18}{'Initial':>12}{'Optimal':>12}")
for name, a, b in zip(state_names, s0, v_star[nz:]):
    print(f"{name:<18}{a:>12.5g}{b:>12.5g}")
print(f"Max relative residual: {np.max(np.abs(g[-n_residuals:])):.2e}")
