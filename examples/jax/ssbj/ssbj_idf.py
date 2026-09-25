"""Monolithic solution of Sobieski's supersonic business jet (SSBJ) problem, in the IDF form.

The individual discipline feasible (IDF) formulation gives the optimizer a target copy of
every coupling variable that crosses between disciplines, alongside the 10 design variables:

    structure    <- lift, engine weight
    aerodynamics <- total weight, wing twist, engine scale factor (ESF)
    propulsion   <- drag

Each discipline is evaluated once per design, from the targets rather than from the other
disciplines, so there is no coupled analysis. Equality constraints force every target to
equal the output of the discipline that computes it, so the disciplines are consistent only
at the solution. The range comes from the disciplines' outputs. The problem and its
disciplines are in models.py.
"""

import numpy as np
import jax.numpy as jnp
import modopt as mo
import warnings
warnings.filterwarnings("ignore")

from models import (x0, xl, xu, structure, aerodynamics, propulsion, breguet_range, design_constraints,
                    cl, cu, report)

# targets y = [lift, engine weight, total weight, twist, ESF, drag], starting from rough guesses
# that aren't consistent. Their bounds are loose, only keeping the models defined during the
# search: they don't cut into the designs that satisfy the design constraints.
coupling_names = ["lift", "engine weight", "total weight", "twist", "ESF", "drag"]
y0 = np.array([50000.0, 6000.0, 50000.0, 1.0, 0.5, 12000.0])
yl = np.array([1e4, 1e3, 1e4, 0.5, 0.1, 1e3])
yu = np.array([2e5, 3e4, 2e5, 1.5, 2.0, 1e5])
nz = len(x0)


def disciplines(v):
    """Every discipline's outputs, each evaluated from the targets, for v = [z, y]."""
    z, y = v[:nz], v[nz:]
    x_shared, x_1, cf, throttle = z[:6], z[6:8], z[8], z[9]
    lift_t, engine_weight_t, total_weight_t, twist_t, esf_t, drag_t = y
    total_weight, fuel_weight, twist, stress = structure(x_shared, x_1, lift_t, engine_weight_t)
    lift, drag, lift_to_drag, pressure_gradient = aerodynamics(x_shared, cf, total_weight_t, twist_t, esf_t)
    sfc, engine_weight, esf, temperature, throttle_ratio = propulsion(x_shared, throttle, drag_t)
    return dict(total_weight=total_weight, fuel_weight=fuel_weight, twist=twist, stress=stress, lift=lift,
                drag=drag, lift_to_drag=lift_to_drag, pressure_gradient=pressure_gradient, sfc=sfc,
                engine_weight=engine_weight, esf=esf, temperature=temperature, throttle_ratio=throttle_ratio,
                range=breguet_range(x_shared, total_weight, fuel_weight, lift_to_drag, sfc))


def objective(v):
    return -disciplines(v)["range"]


def consistency(v):
    """Each target minus the output it copies, relative to the target's initial guess."""
    out = disciplines(v)
    outputs = jnp.stack([out["lift"], out["engine_weight"], out["total_weight"], out["twist"], out["esf"],
                         out["drag"]])
    return (v[nz:] - outputs) / y0


def constraints(v):
    """[design constraints (10), consistency constraints (6)]."""
    out = disciplines(v)
    return jnp.concatenate([design_constraints(out["stress"], out["twist"], out["pressure_gradient"], out["esf"],
                                               out["throttle_ratio"], out["temperature"]), consistency(v)])


n_consistency = len(y0)
# JaxProblem jits the objective, the constraints and their derivatives (jax.grad, jax.jacrev)
prob = mo.JaxProblem(x0=np.concatenate([x0, y0]), jax_obj=objective, jax_con=constraints,
                     xl=np.concatenate([xl, yl]), xu=np.concatenate([xu, yu]),
                     cl=np.concatenate([cl, np.zeros(n_consistency)]), cu=np.concatenate([cu, np.zeros(n_consistency)]),
                     x_scaler=np.concatenate([1 / (xu - xl), 1 / y0]), o_scaler=1e-3)

optimizer = mo.SLSQP(prob, solver_options={"maxiter": 200, "ftol": 1e-12}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()

v_star = optimizer.results["x"] / np.concatenate([1 / (xu - xl), 1 / y0])
z_star = v_star[:nz]
g = np.asarray(constraints(v_star))
report(z_star, -float(objective(v_star)), g[:-n_consistency])

print(f"\n{'Coupling target':<18}{'Initial':>12}{'Optimal':>12}")
for name, a, b in zip(coupling_names, y0, v_star[nz:]):
    print(f"{name:<18}{a:>12.5g}{b:>12.5g}")
print(f"Max relative consistency error: {np.max(np.abs(g[-n_consistency:])):.2e}")
