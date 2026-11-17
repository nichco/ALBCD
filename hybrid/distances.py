"""Distance from the solution of every iterate in convergence_<problem>.npz, for fig_convergence.py.

    python distances.py ranger
    python distances.py fleet 8

The reference solution is computed here, more accurately than any of the runs: OpenSQP on the full problem,
started from the most accurate final iterate of the runs, with tight tolerances (REFERENCE_OPTIONS). The
iterate of that solve with the smallest max(optimality, feasibility) is kept.

The distance is ||z - z_ref|| in the full problem's scaled variables (z * x_scaler, as OpenSQP sees them),
where z is the full problem's design: for the ALBCD iterates of the aerostructural (ranger) problem, the
twist and thickness, without the copies of the loads and weight.

Writes distance_<problem>.npz:
    z_ref, z_ref_optimality, z_ref_feasibility    the reference solution and its accuracy
    <run>_distance                                one entry per entry of <run>_time in convergence_<problem>.npz
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import sys
import pickle
import numpy as np
import hybrid_albcd as H

HERE = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1] if len(sys.argv) > 1 else "ranger"
REFERENCE_OPTIONS = dict(maxiter=60, opt_tol=1e-11, feas_tol=1e-12)

if name == "ranger":
    import ranger as problem
    tag = "ranger"
    scale = problem.Z_SCALER
    to_z = lambda x: np.concatenate([x[problem.TWIST], x[problem.THICKNESS]])  # ALBCD's x -> the full problem's z
    full_problem = problem.full_problem
else:
    import airliner_fleet as problem
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    tag = f"fleet_N{N}"
    scale = np.tile(problem.x_scaler, N)
    to_z = lambda x: x  # the blocks and the full problem share their variables
    _, _, flights, T_budget = problem.make_fleet(N)
    full_problem = lambda z: problem.make_full_model(flights, T_budget, warm_multipliers=False)(z, None, {})

d = np.load(os.path.join(HERE, f"convergence_{tag}.npz"))
runs = sorted({key[:-len("_time")] for key in d.files if key.endswith("_time") and not key.endswith("switch_time")})


def iterates(run):
    """The full problem's design at every entry of <run>_time."""
    if f"{run}_z" in d:  # a hybrid run: the ALBCD states, then OpenSQP's iterates
        return np.vstack([np.array([to_z(x) for x in d[f"{run}_x"]]), d[f"{run}_z"]])
    if run == "albcd":
        return np.array([to_z(x) for x in d[f"{run}_x"]])
    return d[f"{run}_x"]


# ---------------------------------------------------------------- reference solution
finals = {run: max(d[f"{run}_optimality"][-1], d[f"{run}_feasibility"][-1]) for run in runs if run != "albcd"}
start = min(finals, key=finals.get)
sqp = H.RecordingOpenSQP(full_problem(iterates(start)[-1]), turn_off_outputs=True, **REFERENCE_OPTIONS)
# start from that run's final multipliers (the same problem, so OpenSQP orders them the same way): from zero
# multipliers, OpenSQP can wander away from the solution instead of refining it
with open(os.path.join(HERE, f"convergence_{tag}.pkl"), "rb") as f:
    sqp.pi_0 = np.array(pickle.load(f)[start]["results"]["pi"], dtype=float)
sqp.solve()
hist = sqp.history
best = min(range(len(hist)), key=lambda i: max(hist[i][2], hist[i][3]))
z_ref = hist[best][1] / sqp.problem.x_scaler
print(f"{tag}: reference from the end of '{start}' (optimality {d[f'{start}_optimality'][-1]:.1e}): OpenSQP iterate "
      f"{best} of {len(hist) - 1}, optimality {hist[best][2]:.1e}, feasibility {hist[best][3]:.1e}, "
      f"moved {np.linalg.norm((z_ref - iterates(start)[-1]) * scale):.1e}")

# ---------------------------------------------------------------- distances
out = dict(z_ref=z_ref, z_ref_optimality=hist[best][2], z_ref_feasibility=hist[best][3])
for run in runs:
    z = iterates(run)
    assert len(z) == len(d[f"{run}_time"]), run
    out[f"{run}_distance"] = np.linalg.norm((z - z_ref) * scale, axis=1)
    print(f"  {run:19s} final distance {out[f'{run}_distance'][-1]:.2e}")
np.savez(os.path.join(HERE, f"distance_{tag}.npz"), **out)
