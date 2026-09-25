"""Compares ALBCD with collaborative optimization (CO) on the SSBJ problem, by their solution
error against the number of discipline evaluations.

Runs ssbj_albcd.py and ssbj_co.py and plots each one's design error against its cumulative
evaluations of the structure, aerodynamics and propulsion disciplines. To keep the comparison
fair, both scripts count evaluations the same way, in models.evaluations:

- An analysis is one evaluation of a discipline's outputs, and a derivative evaluation is one
  evaluation of its derivatives (a gradient or Jacobian through the discipline), which also
  needs an analysis at a new input. The Breguet range isn't one of the three disciplines, and
  is counted for neither method: ALBCD evaluates it in every block's augmented Lagrangian, and
  CO at the system level from the targets.
- Each discipline remembers only its last input, as a discipline's solver would keep its last
  solution, so a repeated request at that input is free. SLSQP asks for the objective and the
  constraints separately at the same point, for example, and the second is free.
- Every evaluation the algorithms make counts: ALBCD's initial pass through the disciplines,
  its subproblem solves and its optimality checks, and CO's subproblem solves and envelope
  gradients. CO computes the gradients only when the system optimizer asks for the Jacobian,
  not at every line search point. Evaluations made only to record histories or report results
  don't count.
- Both start from the same design, solve their subproblems with modopt's SLSQP, warm started
  from the previous solution, and compute derivatives with JAX.
- The design error is the largest error, relative to the bound widths, over every copy of every
  design variable the method holds: all of ALBCD's block copies, and CO's system-level shared
  variables and every subproblem's copies and local variables.

The two methods stop on different criteria: ALBCD at its feasibility and optimality tolerances,
and CO where its system optimizer converges, with an error floor of order sqrt(EPS) from its
relaxed compatibility constraints. Run ssbj.py first to write the MDF solution,
monolithic_solution.npz, that the errors are measured against.
"""

import os
import io
import runpy
import contextlib
import numpy as np
import matplotlib.pyplot as plt

import models

HERE = os.path.dirname(os.path.abspath(__file__))
solution = np.load(os.path.join(HERE, "monolithic_solution.npz"))
width = models.xu - models.xl


def run(script):
    """Runs a script without its plots or printed output, and returns its evaluation counts and
    design error at each point in its trace, with its last lines of output."""
    show = plt.show
    plt.show = lambda *args, **kwargs: None
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            result = runpy.run_path(os.path.join(HERE, script), run_name="__main__")
    finally:
        plt.show = show
        plt.close("all")

    z_index = np.array(result["INSTANCE_Z"])
    analyses = np.array([a for a, _, _ in result["trace"]])
    derivatives = np.array([d for _, d, _ in result["trace"]])
    error = np.array([np.max(np.abs(instances - solution["z"][z_index]) / width[z_index])
                      for _, _, instances in result["trace"]])
    summary = [line for line in output.getvalue().splitlines()
               if line.startswith(("Converged", "Did not converge", "\tSuccess", "Range (nm)", "Max design",
                                   "Discipline evaluations"))]
    return analyses, derivatives, error, summary


results = {}
for name, script in [("ALBCD", "ssbj_albcd.py"), ("CO", "ssbj_co.py")]:
    results[name] = run(script)
    analyses, derivatives, error, summary = results[name]
    print(f"{name}:")
    for line in summary:
        print(f"    {line.strip()}")
    print(f"    Largest error over every copy of every design variable: {error[-1]:.2e}")

# the curves, to replot without rerunning both methods
np.savez(os.path.join(HERE, 'ssbj_co_albcd_comparison.npz'),
         **{f"{name.lower()}_{key}": value for name, (analyses, derivatives, error, _) in results.items()
            for key, value in [("analyses", analyses), ("derivatives", derivatives), ("error", error)]})


colors = {"ALBCD": "tab:blue", "CO": "tab:red"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
for name, (analyses, derivatives, error, _) in results.items():
    ax1.semilogy(analyses, error, color=colors[name], linewidth=2, label=name)
    ax2.semilogy(derivatives, error, color=colors[name], linewidth=2, label=name)
ax1.set_xlabel('Cumulative discipline analyses')
ax1.set_ylabel('Design error')
ax2.set_xlabel('Cumulative discipline derivative evaluations')
for ax in (ax1, ax2):
    ax.grid(color='lavender', alpha=0.5, axis='y')
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(HERE, 'ssbj_co_albcd_comparison.pdf'))
plt.show()
