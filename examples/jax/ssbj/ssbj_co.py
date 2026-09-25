"""Sobieski's supersonic business jet (SSBJ) problem with collaborative optimization (CO).

Braun's bi-level CO architecture. The system level optimizes the shared design variables and
targets for the coupling variables to maximize the range, computed from the targets, subject
to one compatibility constraint per discipline:

    maximize    range(altitude, Mach number, total weight, fuel weight, lift/drag, SFC targets)
    w.r.t.      s = [t/c, altitude, Mach number, aspect ratio, sweep, wing area,
                     total weight, fuel weight, twist, engine weight, ESF, drag, lift/drag, SFC]
    subject to  J_i*(s) <= eps  for each discipline i

Each discipline's subproblem finds the local design, its own copies of the shared design
variables it uses and copies of its coupling inputs that best match the system's values,
subject to its local constraints:

    J_i*(s) = min  sum of squared, scaled differences between the discipline's copies and
                   outputs and the corresponding system values
              s.t. the discipline's local constraints

The subproblems' constraints don't depend on s, so the envelope theorem gives the exact
gradient of J_i* as the partial derivative of J_i with respect to s at the subproblem's
solution. The compatibility constraints are the inequalities J_i* <= eps rather than the
equalities J_i* = 0: J_i* and its gradient are both zero wherever the disciplines agree with
s, so the equalities violate the constraint qualification that SLSQP relies on. The
solution's error is then of order sqrt(eps). Run this file to solve the problem and compare
it with the MDF solution in monolithic_solution.npz (regenerate it with ssbj.py).
"""

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import modopt as mo
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import models
from models import structure, aerodynamics, propulsion, breguet_range, report

HERE = os.path.dirname(os.path.abspath(__file__))

EPS = 1e-10  # compatibility tolerance on each J_i*; the design error is of order sqrt(EPS)

# system variables s = [z_shared, targets]
(TC, H, M, AR, SWEEP, AREA, WT, WF, TWIST, WE, ESF, DRAG, LD, SFC) = range(14)
target_names = ["total weight", "fuel weight", "twist", "engine weight", "ESF", "drag", "lift/drag", "SFC"]
# rough guesses for the targets, which also scale them, and bounds that keep the models defined.
# The total weight is at least 5000 lb above the fuel weight, so the range's logarithm is defined.
t0 = np.array([50000.0, 7000.0, 1.0, 6000.0, 0.5, 12000.0, 4.0, 1.1])
tl = np.array([30000.0, 2000.0, 0.5, 1e3, 0.1, 1e3, 0.1, 0.5])
tu = np.array([77250.0, 25000.0, 1.5, 3e4, 2.0, 1e5, 20.0, 5.0])
s0 = np.concatenate([models.x0[:6], t0])
sl = np.concatenate([models.xl[:6], tl])
su = np.concatenate([models.xu[:6], tu])
s_scaler = np.concatenate([1 / (models.xu[:6] - models.xl[:6]), 1 / t0])  # scales every difference in J_i


def system_objective(s):
    """Negative range in 1000 nm, from the targets."""
    return -breguet_range(jnp.stack([0.0, s[H], s[M]]), s[WT], s[WF], s[LD], s[SFC]) / 1e3


# Each discipline: v -> (the values it matches to the system, its local constraints). Discipline()
# takes the index in s of each value, and the local design variables and system copies that make up v.
def structure_values(v):
    taper, wingbox, tc, AR_, sweep, S, lift, engine_weight = v
    total_weight, fuel_weight, twist, stress = structure(jnp.stack([tc, 0.0, 0.0, AR_, sweep, S]), v[:2],
                                                         lift, engine_weight)
    values = jnp.stack([tc, AR_, sweep, S, lift, engine_weight, total_weight, fuel_weight, twist])
    return values, jnp.concatenate([stress, jnp.stack([twist])])


def aerodynamics_values(v):
    cf, tc, h, M_, sweep, S, total_weight, twist, esf = v
    _, drag, lift_to_drag, pressure_gradient = aerodynamics(jnp.stack([tc, h, M_, 0.0, sweep, S]), cf,
                                                            total_weight, twist, esf)
    values = jnp.stack([tc, h, M_, sweep, S, total_weight, twist, esf, drag, lift_to_drag])
    return values, jnp.stack([pressure_gradient])


def propulsion_values(v):
    throttle, h, M_, drag = v
    sfc, engine_weight, esf, temperature, throttle_ratio = propulsion(jnp.stack([0.0, h, M_]), throttle, drag)
    values = jnp.stack([h, M_, drag, sfc, engine_weight, esf])
    return values, jnp.stack([esf, throttle_ratio, temperature])


class Discipline:
    """One CO subproblem: J_i*(s) and its gradient."""

    def __init__(self, name, values, match, local, copies, cl, cu):
        self.name = name
        self.match = np.array(match)
        # local variables: the local design variables (indices into models.x0), then copies of the
        # system variables (indices into s), in the order values() unpacks them
        index = [("z", j) for j in local] + [("s", j) for j in copies]
        self.v = np.array([models.x0[j] if k == "z" else s0[j] for k, j in index])
        self.vl = np.array([models.xl[j] if k == "z" else sl[j] for k, j in index])
        self.vu = np.array([models.xu[j] if k == "z" else su[j] for k, j in index])
        self.v_scaler = 1 / (self.vu - self.vl)
        self.cl, self.cu = cl, cu
        # the entries of v that are design variables or copies of them, and the entry of z each copies
        design = [(p, j) for p, (k, j) in enumerate(index) if k == "z" or j < 6]
        self.design_positions = [p for p, _ in design]
        self.design_z = [j for _, j in design]

        def J(v, s):
            return jnp.sum(((values(v)[0] - s[self.match]) * s_scaler[self.match])**2)

        # every call evaluates this discipline at v, or its derivatives, and is counted in
        # models.evaluations. The discipline's inputs are all in v.
        self.J = self.counted(jax.jit(J), False)
        self.dJdv = self.counted(jax.jit(jax.grad(J)), True)
        self.dJds = self.counted(jax.jit(jax.grad(J, argnums=1)), True)
        self.con = self.counted(jax.jit(lambda v: values(v)[1]), False)
        self.jac = self.counted(jax.jit(jax.jacfwd(lambda v: values(v)[1])), True)
        self.solves = 0

    def counted(self, fn, derivative):
        """fn(v, ...), counted as an evaluation of this discipline (or its derivatives) at v."""
        def wrapper(v, *args):
            models.evaluations(self.name, v, derivative)
            return fn(v, *args)
        return wrapper

    def solve(self, s):
        """J_i*(s), warm started from the previous solution."""
        s = jnp.asarray(s)
        prob = mo.ProblemLite(x0=self.v, obj=lambda v: float(self.J(jnp.asarray(v), s)),
                              grad=lambda v: np.array(self.dJdv(jnp.asarray(v), s)),
                              con=lambda v: np.array(self.con(jnp.asarray(v))),
                              jac=lambda v: np.array(self.jac(jnp.asarray(v))),
                              xl=self.vl, xu=self.vu, cl=self.cl, cu=self.cu, x_scaler=self.v_scaler,
                              name=self.name)
        optimizer = mo.SLSQP(prob, solver_options={"maxiter": 500, "ftol": 1e-16}, turn_off_outputs=True)
        optimizer.solve()
        self.v = np.clip(optimizer.results["x"] / self.v_scaler, self.vl, self.vu)
        self.solves += 1
        return float(self.J(jnp.asarray(self.v), s))

    def gradient(self, s):
        """dJ_i*/ds at the last solution, by the envelope theorem."""
        return np.array(self.dJds(jnp.asarray(self.v), jnp.asarray(s)))


disciplines = [
    # structure owns [taper ratio, wingbox area] and copies of t/c, aspect ratio, sweep, wing area,
    # the lift (the total weight target) and the engine weight
    Discipline("structure", structure_values, [TC, AR, SWEEP, AREA, WT, WE, WT, WF, TWIST], [6, 7],
               [TC, AR, SWEEP, AREA, WT, WE], models.cl[:6], models.cu[:6]),
    # aerodynamics owns [skin friction] and copies of t/c, altitude, Mach number, sweep, wing area,
    # the total weight, twist and ESF
    Discipline("aerodynamics", aerodynamics_values, [TC, H, M, SWEEP, AREA, WT, TWIST, ESF, DRAG, LD], [8],
               [TC, H, M, SWEEP, AREA, WT, TWIST, ESF], models.cl[6:7], models.cu[6:7]),
    # propulsion owns [throttle] and copies of altitude, Mach number and drag
    Discipline("propulsion", propulsion_values, [H, M, DRAG, SFC, WE, ESF], [9],
               [H, M, DRAG], models.cl[7:], models.cu[7:]),
]

# the evaluation counts and every copy of every design variable at each system point, for
# comparing the cost of distributed methods (compare_co_albcd.py)
INSTANCE_Z = list(range(6)) + [j for d in disciplines for j in d.design_z]  # the entry of z each copies
trace = []
models.evaluations.reset()


def record(s):
    instances = np.concatenate([s[:6]] + [d.v[d.design_positions] for d in disciplines])
    trace.append((models.evaluations.analyses, models.evaluations.derivatives, instances))


solved = {"s": None}  # the system point the subproblems were last solved at, and their results


def compatibility(s):
    """[J_1*, J_2*, J_3*], solving the subproblems once per system point."""
    if solved["s"] is None or not np.array_equal(s, solved["s"]):
        solved.update(s=s.copy(), J=np.array([d.solve(s) for d in disciplines]), grad=None)
        history.append((s.copy(), solved["J"]))
        record(s)
    return solved["J"]


def compatibility_jacobian(s):
    """dJ_i*/ds, only when the system optimizer asks for it, so line search points don't pay for it."""
    compatibility(s)
    if solved["grad"] is None:
        solved["grad"] = np.array([d.gradient(s) for d in disciplines])
        # the same design, now also charged for the gradients
        trace[-1] = (models.evaluations.analyses, models.evaluations.derivatives, trace[-1][2])
    return solved["grad"]


history = []  # (s, [J_i*]) at every system point where the subproblems were solved
record(s0)
obj, grad = jax.jit(system_objective), jax.jit(jax.grad(system_objective))

# modopt scales s by s_scaler before SLSQP sees it, and J_i* <= eps by 1 / eps
prob = mo.ProblemLite(x0=s0, obj=lambda s: float(obj(jnp.asarray(s))), grad=lambda s: np.array(grad(jnp.asarray(s))),
                      con=lambda s: compatibility(np.asarray(s)), jac=lambda s: compatibility_jacobian(np.asarray(s)),
                      xl=sl, xu=su, cl=np.full(3, -np.inf), cu=np.full(3, EPS),
                      x_scaler=s_scaler, c_scaler=1 / EPS, name="co_system")

# J_i* is only as accurate as the subproblem solves, and with a tighter ftol the system SLSQP
# stops on a failed line search near the solution instead
optimizer = mo.SLSQP(prob, solver_options={"maxiter": 500, "ftol": 1e-8}, turn_off_outputs=True)
t_start = time.perf_counter()
optimizer.solve()
t_solve = time.perf_counter() - t_start
optimizer.print_results()

s_star = optimizer.results["x"] / s_scaler
Js = solved["J"]  # SLSQP returns the last system point, so the subproblems needn't be solved again

# the design: the shared design variables from the system level and the local ones from the subproblems
z = np.concatenate([s_star[:6], disciplines[0].v[:2], disciplines[1].v[:1], disciplines[2].v[:1]])
out = jax.jit(models.analysis)(z)  # the coupled analysis at that design, to check it
g = np.array(models.design_constraints(out["stress"], out["twist"], out["pressure_gradient"], out["esf"],
                                       out["throttle_ratio"], out["temperature"]))
report(z, float(out["range"]), g)

solution = np.load(os.path.join(HERE, "monolithic_solution.npz"))
scale = models.xu - models.xl
print(f"\nRange from the system targets: {-1e3 * float(obj(jnp.asarray(s_star))):.2f} nm; "
      f"range (MDF): {float(solution['range']):.2f} nm")
print(f"Compatibility J_i*: {Js}")
print(f"Max design error relative to the bound widths: {np.max(np.abs(z - solution['z']) / scale):.2e}")
print(f"System iterations: {optimizer.results['nit']}, system points: {len(history)}, "
      f"subproblem solves: {sum(d.solves for d in disciplines)}, time: {t_solve:.1f} s")
print(f"Discipline evaluations: {models.evaluations.analyses} analyses, {models.evaluations.derivatives} derivatives")

print(f"\n{'Target':<18}{'Initial':>12}{'Optimal':>12}")
for name, a, b in zip(target_names, t0, s_star[6:]):
    print(f"{name:<18}{a:>12.5g}{b:>12.5g}")


# the shared design variables' error and the compatibility at each system point
error = [np.max(np.abs(s[:6] - solution["z"][:6]) / scale[:6]) for s, _ in history]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.25))
ax1.semilogy([np.max(J) for _, J in history], color='tab:orange', linewidth=2)
ax1.axhline(EPS, color='tab:gray', linewidth=1, linestyle='--')
ax1.set_xlabel('System point')
ax1.set_ylabel('max J_i*')
ax2.semilogy(error, color='tab:red', linewidth=2)
ax2.set_xlabel('System point')
ax2.set_ylabel('Shared design error')
plt.tight_layout()
plt.show()
