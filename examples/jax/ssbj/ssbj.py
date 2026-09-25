"""Monolithic solution of Sobieski's supersonic business jet (SSBJ) problem, in the MDF form.

The multidisciplinary feasible (MDF) formulation is a single SLSQP over the 10 design
variables, with the coupled analysis (models.analysis) converged by Gauss-Seidel iteration
at every design and differentiated implicitly with jax.lax.custom_root. The problem and its
disciplines are in models.py; ssbj_idf.py and ssbj_sand.py solve the same problem in the IDF and SAND forms.
The solution is saved to monolithic_solution.npz.
"""

import os
import numpy as np
import modopt as mo
import warnings
warnings.filterwarnings("ignore")

from models import x0, xl, xu, analysis, design_constraints, cl, cu, report

HERE = os.path.dirname(os.path.abspath(__file__))


def objective(z):
    return -analysis(z)["range"]


def constraints(z):
    out = analysis(z)
    return design_constraints(out["stress"], out["twist"], out["pressure_gradient"], out["esf"],
                              out["throttle_ratio"], out["temperature"])


# JaxProblem jits the objective, the constraints and their derivatives (jax.grad, jax.jacrev)
prob = mo.JaxProblem(x0=x0, jax_obj=objective, jax_con=constraints, xl=xl, xu=xu, cl=cl, cu=cu,
                     x_scaler=1 / (xu - xl), o_scaler=1e-3)

optimizer = mo.SLSQP(prob, solver_options={"maxiter": 200, "ftol": 1e-12}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()

z_star = optimizer.results["x"] * (xu - xl)
range_star = -float(objective(z_star))
report(z_star, range_star, np.asarray(constraints(z_star)))

np.savez(os.path.join(HERE, "monolithic_solution.npz"), z=z_star, range=range_star)
