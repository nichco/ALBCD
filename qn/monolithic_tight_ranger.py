"""Tight monolithic reference for ranger (Task 5).

The problem of examples/jax/ranger/monolithic.py (objective and constraints copied from its lines 24-52), solved
with SLSQP at ftol = 1e-14, maxiter = 1000. Saves qn/results/monolithic_solution_tight_ranger.npz (twist_cp,
thickness_cp, CD, plus SLSQP's multipliers and KKT residuals) and compares with the example's reference.
No plotting or pyvista.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ranger_blocks import import_example_models, EXAMPLE  # noqa: E402

models = import_example_models(EXAMPLE, "ranger_models")

import numpy as np  # noqa: E402
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import modopt as mo  # noqa: E402
import warnings  # noqa: E402
warnings.filterwarnings("ignore")

aero_model, structures_model = models.aero_model, models.structures_model
num_cp_twist, num_cp_thickness = models.num_cp_twist, models.num_cp_thickness
twist_cp0, thickness_cp0 = models.twist_cp0, models.thickness_cp0
min_gauge, sigma_yield_mpa, thickness_max = models.min_gauge, models.sigma_yield_mpa, models.thickness_max

TWIST_SLICE = slice(0, num_cp_twist)
THICKNESS_SLICE = slice(num_cp_twist, num_cp_twist + num_cp_thickness)


def objective(x):
    CD, _, _ = aero_model(x[TWIST_SLICE])
    return CD


def constraints(x):
    _, aero_loads, lift = aero_model(x[TWIST_SLICE])
    max_sigma_mpa, weight = structures_model(aero_loads, x[THICKNESS_SLICE])
    return jnp.stack([max_sigma_mpa - sigma_yield_mpa,
                      lift - weight])


x0 = np.concatenate([twist_cp0, thickness_cp0])

cl = np.array([-np.inf, 0.0])  # stress <= yield, lift = weight
cu = np.array([0.0, 0.0])
c_scaler = np.array([1e-2, 1e-2])
x_scaler = np.concatenate([10 * np.ones(num_cp_twist), 100 * np.ones(num_cp_thickness)])
xl = np.concatenate([np.full(num_cp_twist, -np.inf), np.full(num_cp_thickness, min_gauge)])
xu = np.concatenate([np.full(num_cp_twist, np.inf), np.full(num_cp_thickness, thickness_max)])

if __name__ == "__main__":
    prob = mo.JaxProblem(x0=x0, jax_obj=objective, jax_con=constraints, xl=xl, xu=xu,
                         cl=cl, cu=cu, x_scaler=x_scaler, c_scaler=c_scaler, o_scaler=1e2)

    optimizer = mo.SLSQP(prob, solver_options={'maxiter': 1000, 'ftol': 1e-14}, turn_off_outputs=True)
    t0 = time.perf_counter()
    optimizer.solve()
    elapsed = time.perf_counter() - t0
    optimizer.print_results()

    x = optimizer.results['x'] / x_scaler
    CD_star = float(objective(x))

    # KKT residual of the unscaled problem min 1e2 CD (the aero block's objective scale): SLSQP's multipliers
    # m (eq, then the upper-bounded stress row) satisfy grad(o CD)_s = J_s^T m in its scaled variables/constraints
    m = np.asarray(optimizer.results["multipliers"])
    g = np.asarray(jax.grad(objective)(jnp.asarray(x))) * 1e2
    Jc = np.asarray(jax.jacrev(constraints)(jnp.asarray(x)))
    # rows in SLSQP order: equality (lift = weight), then stress (as cu - c >= 0)
    rows = np.vstack([Jc[1] * c_scaler[1], -Jc[0] * c_scaler[0]])
    resid = g - rows.T @ m
    on_lower = np.isfinite(xl) & (x <= xl + 1e-8 * np.maximum(1, np.abs(xl)))
    free = ~(on_lower & (resid >= 0))
    c = np.asarray(constraints(jnp.asarray(x)))
    stationarity = float(np.max(np.abs(resid[free])))
    feasibility = float(max(abs(c[1]), max(c[0], 0.0)))

    ref = np.load(EXAMPLE / "monolithic_solution.npz")
    dist = np.linalg.norm(np.concatenate([ref["twist_cp"], ref["thickness_cp"]]) - x)
    print(f"CD tight {CD_star:.12e}  example reference {float(ref['CD']):.12e}  difference {CD_star - float(ref['CD']):+.2e}")
    print(f"||x_tight - x_ref|| (twist, thickness, unscaled) {dist:.3e}")
    print(f"KKT: stationarity (1e2 CD scale) {stationarity:.2e}, feasibility {feasibility:.2e}, time {elapsed:.1f} s")

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    np.savez(out / "monolithic_solution_tight_ranger.npz", twist_cp=x[TWIST_SLICE], thickness_cp=x[THICKNESS_SLICE],
             CD=CD_star, multipliers=m, stationarity=stationarity, feasibility=feasibility, time=elapsed,
             nit=optimizer.results.get("nit", -1), success=optimizer.results["success"],
             message=str(optimizer.results["message"]), CD_example=float(ref["CD"]), dist_example=dist)
