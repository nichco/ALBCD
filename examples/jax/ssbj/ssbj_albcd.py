"""Sobieski's supersonic business jet (SSBJ) problem with ALBCD.

Maximize the Breguet range of a supersonic business jet. The ALBCD problem is the IDF
problem in ssbj_idf.py with the shared design variables copied into each discipline that
uses them. The three disciplines are the three blocks, solved in this order:

    StructureBlock     owns [taper ratio, wingbox area, aspect ratio, t/c, sweep, wing area,
                             lift copy, engine weight copy]
                       local constraints: stresses, twist
    AerodynamicsBlock  owns [skin friction, t/c, altitude, Mach number, sweep, wing area,
                             total weight copy, twist copy, ESF copy]
                       local constraints: pressure gradient
    PropulsionBlock    owns [throttle, altitude, Mach number, drag copy]
                       local constraints: ESF, throttle setting, engine temperature

The coupling constraints phi force each copy of a coupling variable to equal the output it
copies (6, as in IDF) and the copies of each shared design variable to agree (5). The
objective is the global range, from the structure's weights, the aerodynamic lift-to-drag
ratio and the propulsion's SFC; the blocks exchange these outputs through ALBCD's data.
Gradients come from JAX. Run this file to solve the problem, compare the result with the MDF
solution in monolithic_solution.npz (regenerate it with ssbj.py), and plot the convergence.
"""

import os
from collections import namedtuple
import numpy as np
import jax
import jax.numpy as jnp
import modopt as mo
import matplotlib.pyplot as plt
from albcd import ALBCD, Subproblem
import warnings
warnings.filterwarnings("ignore")

import models
from models import structure, aerodynamics, propulsion, breguet_range, report

HERE = os.path.dirname(os.path.abspath(__file__))

# global x is a single flat vector, ordered so that each block's variables are contiguous
(S_TAPER, S_WINGBOX, S_AR, S_TC, S_SWEEP, S_AREA, S_LIFT, S_WE,
 A_CF, A_TC, A_H, A_M, A_SWEEP, A_AREA, A_WT, A_TWIST, A_ESF,
 P_THROTTLE, P_H, P_M, P_DRAG) = range(21)
STRUCT_INDEX = slice(S_TAPER, S_WE + 1)
AERO_INDEX = slice(A_CF, A_ESF + 1)
PROP_INDEX = slice(P_THROTTLE, P_DRAG + 1)
N = P_DRAG + 1

# the entry of models.x0 (the design vector z) behind each design variable in x
DESIGN = {S_TAPER: 6, S_WINGBOX: 7, S_AR: 3, S_TC: 0, S_SWEEP: 4, S_AREA: 5,
          A_CF: 8, A_TC: 0, A_H: 1, A_M: 2, A_SWEEP: 4, A_AREA: 5,
          P_THROTTLE: 9, P_H: 1, P_M: 2}
# the design vector z from x, taking one copy of each shared design variable (they agree at the solution)
Z_FROM_X = [S_TC, A_H, A_M, S_AR, S_SWEEP, S_AREA, S_TAPER, S_WINGBOX, A_CF, P_THROTTLE]

# copies of coupling variables: [lift, engine weight, total weight, twist, ESF, drag], with
# rough guesses (which also scale them), and the loose bounds of ssbj_idf.py, except for the
# structure's copies. At the optimum the wingbox area is at its lower bound, where the wing
# weight is negative. The structure block's lift copy is free of the lift = total weight
# coupling, so without a bound it grows until the negative wing weight brings the total weight
# down to the fuel weight, where the Breguet logarithm, and the augmented Lagrangian with it,
# is unbounded. GEMSEO's coupling bounds on the lift (24850 to 77250 lb) and the engine weight
# (at least 2960 lb) keep the total weight at least 3600 lb above the fuel weight.
COPIES = [S_LIFT, S_WE, A_WT, A_TWIST, A_ESF, P_DRAG]
copy_ref = np.array([50000.0, 6000.0, 50000.0, 1.0, 0.5, 12000.0])
copy_l = np.array([24850.0, 2960.0, 1e4, 0.5, 0.1, 1e3])
copy_u = np.array([77250.0, 3e4, 2e5, 1.5, 2.0, 1e5])

# copies of shared design variables that must agree: structure's and aerodynamics' t/c, sweep
# and wing area, and aerodynamics' and propulsion's altitude and Mach number
CONSENSUS = ([S_TC, S_SWEEP, S_AREA, A_H, A_M], [A_TC, A_SWEEP, A_AREA, P_H, P_M])

xl, xu, x_scaler = np.zeros(N), np.zeros(N), np.zeros(N)
for i, j in DESIGN.items():
    xl[i], xu[i] = models.xl[j], models.xu[j]
    x_scaler[i] = 1 / (models.xu[j] - models.xl[j])
xl[COPIES], xu[COPIES], x_scaler[COPIES] = copy_l, copy_u, 1 / copy_ref
N_CON = len(COPIES) + len(CONSENSUS[0])


def structure_model(x):
    """Structure outputs [total weight, fuel weight, twist] and its local constraints [stresses, twist]."""
    x_shared = jnp.stack([x[S_TC], 0.0, 0.0, x[S_AR], x[S_SWEEP], x[S_AREA]])  # altitude and Mach are unused
    total_weight, fuel_weight, twist, stress = structure(x_shared, x[S_TAPER:S_WINGBOX + 1], x[S_LIFT], x[S_WE])
    return jnp.stack([total_weight, fuel_weight, twist]), jnp.concatenate([stress, jnp.stack([twist])])


def aerodynamics_model(x):
    """Aerodynamic outputs [lift, drag, lift-to-drag ratio] and its local constraint [pressure gradient]."""
    x_shared = jnp.stack([x[A_TC], x[A_H], x[A_M], 0.0, x[A_SWEEP], x[A_AREA]])  # the aspect ratio is unused
    lift, drag, lift_to_drag, pressure_gradient = aerodynamics(x_shared, x[A_CF], x[A_WT], x[A_TWIST], x[A_ESF])
    return jnp.stack([lift, drag, lift_to_drag]), jnp.stack([pressure_gradient])


def propulsion_model(x):
    """Propulsion outputs [SFC, engine weight, ESF] and its local constraints [ESF, throttle / max, temperature]."""
    x_shared = jnp.stack([0.0, x[P_H], x[P_M], 0.0, 0.0, 0.0])  # only altitude and Mach are used
    sfc, engine_weight, esf, temperature, throttle_ratio = propulsion(x_shared, x[P_THROTTLE], x[P_DRAG])
    return jnp.stack([sfc, engine_weight, esf]), jnp.stack([esf, throttle_ratio, temperature])


def objective(x, outs):
    """Negative range in 1000 nm, from every discipline's outputs outs, keyed by discipline."""
    total_weight, fuel_weight, _ = outs["structure"]
    _, _, lift_to_drag = outs["aerodynamics"]
    sfc, _, _ = outs["propulsion"]
    x_shared = jnp.stack([0.0, x[A_H], x[A_M]])  # the range depends on altitude and Mach only
    return -breguet_range(x_shared, total_weight, fuel_weight, lift_to_drag, sfc) / 1e3


def coupling(x, outs):
    """Each coupling variable copy minus its output, relative to its guess, then each pair of design copies'
    difference, relative to its bound width."""
    total_weight, _, twist = outs["structure"]
    lift, drag, _ = outs["aerodynamics"]
    _, engine_weight, esf = outs["propulsion"]
    actual = jnp.stack([lift, engine_weight, total_weight, twist, esf, drag])
    first, second = CONSENSUS
    return jnp.concatenate([(x[jnp.array(COPIES)] - actual) / copy_ref,
                            (x[jnp.array(first)] - x[jnp.array(second)]) * x_scaler[first]])


# a block's augmented Lagrangian, its gradient, its local constraints and their Jacobian, each jitted
Compiled = namedtuple("Compiled", "obj grad con jac")


class DisciplineBlock(Subproblem):
    """One discipline's block: minimizes the augmented Lagrangian over the discipline's variables,
    subject to its local constraints and bounds.

    The augmented Lagrangian depends on the other disciplines' outputs, which are fixed while this
    block solves; each block reads them from ALBCD's data and writes its own outputs there.
    """

    NAME = None    # key of this discipline's outputs in the data
    MODEL = None   # x -> (outputs, local constraints)
    CL = CU = None # bounds on the local constraints
    FTOL = 1e-10   # SLSQP tolerance

    def setup(self) -> None:
        self.others = [name for name in ("structure", "aerodynamics", "propulsion") if name != self.NAME]
        self.add_input("x")
        self.add_input("y")  # Lagrange multipliers
        self.add_input("mu") # penalty parameters
        for name in self.others:
            self.add_input(name)  # the other disciplines' outputs
        self.add_output("x")
        self.add_output("phi") # coupling constraints at the new x
        self.add_output(self.NAME)

        self.own = np.arange(N)[self.index]
        self.rest = np.setdiff1d(np.arange(N), self.own)  # the order of Subproblem.other()
        self.xl, self.xu, self.x_scaler = xl[self.index], xu[self.index], x_scaler[self.index]
        # modopt's SLSQP splits the local constraints into c - cl >= 0 and cu - c >= 0, and
        # returns a multiplier for each, in that order
        self.lower = np.where(self.CL > -np.inf)[0]
        self.upper = np.where(self.CU < np.inf)[0]

        # compiled once: the inputs that change between solves are arguments, not constants
        self._fns = Compiled(jax.jit(self.lagrangian), jax.jit(jax.grad(self.lagrangian)),
                             jax.jit(self.local_constraints), jax.jit(jax.jacrev(self.local_constraints)))

    def join(self, v, other):
        """The global x from this block's variables v and the others' variables."""
        return jnp.zeros(N).at[self.own].set(v).at[self.rest].set(other)

    def lagrangian(self, v, other, y, mu, outs):
        x = self.join(v, other)
        outs = dict(outs, **{self.NAME: type(self).MODEL(x)[0]})
        c = coupling(x, outs)
        return objective(x, outs) + jnp.sum(y * c) + 0.5 * jnp.sum(mu * c**2)

    def local_constraints(self, v, other):
        return type(self).MODEL(self.join(v, other))[1]

    def arguments(self, inputs):
        x = inputs["x"]
        return (jnp.asarray(self.other(x)), jnp.asarray(inputs["y"]), jnp.asarray(inputs["mu"]),
                {name: jnp.asarray(inputs[name]) for name in self.others})

    def solve(self, inputs, outputs) -> None:
        x = inputs["x"]
        other, y, mu, outs = args = self.arguments(inputs)
        fns = self._fns

        # modopt passes numpy arrays; convert them so the compiled functions always see JAX arrays
        prob = mo.ProblemLite(x0=np.array(self.decompose(x)),
                              obj=lambda v: np.float64(fns.obj(jnp.asarray(v), *args)),
                              grad=lambda v: np.array(fns.grad(jnp.asarray(v), *args)),
                              con=lambda v: np.array(fns.con(jnp.asarray(v), other)),
                              jac=lambda v: np.array(fns.jac(jnp.asarray(v), other)),
                              xl=self.xl, xu=self.xu, cl=self.CL, cu=self.CU, x_scaler=self.x_scaler,
                              name=type(self).__name__)
        optimizer = mo.SLSQP(prob, solver_options={"maxiter": 500, "ftol": self.FTOL}, turn_off_outputs=True)
        optimizer.solve()
        v = optimizer.results["x"] / self.x_scaler

        # one signed multiplier per local constraint, so that the gradient equals jac.T @ multipliers
        # at a stationary point. Neither the objective nor the constraints are scaled, and scaling
        # x leaves the multipliers unchanged.
        m = np.asarray(optimizer.results["multipliers"])
        self._multipliers = np.zeros(len(self.CL))
        self._multipliers[self.lower] += m[:len(self.lower)]
        self._multipliers[self.upper] -= m[len(self.lower):]
        self._jac_con = np.array(fns.jac(jnp.asarray(v), other))

        x_new = self.recompose(x, v)
        own = type(self).MODEL(jnp.asarray(x_new))[0]
        outputs["x"] = x_new
        outputs[self.NAME] = np.array(own)
        # the other blocks' variables, and so their outputs, are unchanged by this solve
        outputs["phi"] = np.array(coupling(jnp.asarray(x_new), dict(outs, **{self.NAME: own})))

    def residual(self, inputs) -> float:
        """Max-norm KKT stationarity residual, with respect to the scaled variables."""
        v = jnp.asarray(self.decompose(inputs["x"]))
        grad = np.array(self._fns.grad(v, *self.arguments(inputs)))

        # KKT stationarity with SLSQP's multipliers for the local constraints. They depend only on
        # this block's variables, so their Jacobian is unchanged since this block last solved.
        resid = (grad - self._jac_con.T @ self._multipliers) / self.x_scaler

        # SLSQP's multipliers don't cover the bounds, so at an active bound only the
        # component of the residual pointing into the infeasible region counts
        v = np.array(v)
        violation = np.abs(resid)
        violation = np.where((v - self.xl) * self.x_scaler <= 1e-8, np.maximum(0.0, -resid), violation)
        violation = np.where((self.xu - v) * self.x_scaler <= 1e-8, np.maximum(0.0, resid), violation)
        return float(np.max(violation))


class StructureBlock(DisciplineBlock):
    NAME = "structure"
    MODEL = structure_model
    CL = models.cl[:6]  # stresses, twist
    CU = models.cu[:6]


class AerodynamicsBlock(DisciplineBlock):
    NAME = "aerodynamics"
    MODEL = aerodynamics_model
    CL = models.cl[6:7]  # pressure gradient
    CU = models.cu[6:7]


class PropulsionBlock(DisciplineBlock):
    NAME = "propulsion"
    MODEL = propulsion_model
    CL = models.cl[7:]  # ESF, throttle / max, temperature
    CU = models.cu[7:]


# the initial design, with every copy of a shared design variable at its initial value. The
# coupling variable copies come from one pass through the disciplines, starting from the guesses.
x0 = np.zeros(N)
for i, j in DESIGN.items():
    x0[i] = models.x0[j]
x0[COPIES] = copy_ref
x0[A_WT], _, x0[A_TWIST] = structure_model(x0)[0]
x0[S_LIFT], x0[P_DRAG], _ = aerodynamics_model(x0)[0]
_, x0[S_WE], x0[A_ESF] = propulsion_model(x0)[0]
data0 = {"structure": np.array(structure_model(x0)[0]),
         "aerodynamics": np.array(aerodynamics_model(x0)[0]),
         "propulsion": np.array(propulsion_model(x0)[0])}

opt = ALBCD(subproblems=[StructureBlock(STRUCT_INDEX), AerodynamicsBlock(AERO_INDEX), PropulsionBlock(PROP_INDEX)],
            x0=x0,
            mu0=np.ones(N_CON),
            data0=data0,
            max_mu=1e6,
            rho=1.2,
            tau=0.5,
            feas_tol=1e-6,
            opt_tol=[1e-2, 1e-5],
            max_outer_iter=100,
            # once the penalties are large, each sweep reduces the optimality residual by only
            # ~10%, so the inner loop needs many sweeps to reach the final opt_tol
            max_inner_iter=100)

opt.solve()


# compare with the MDF solution (the same problem solved with one SLSQP)
solution = np.load(os.path.join(HERE, "monolithic_solution.npz"))
z = opt.x[Z_FROM_X]
range_ = -1e3 * float(objective(jnp.asarray(opt.x), {k: jnp.asarray(v) for k, v in opt.data.items()}))
g = np.concatenate([np.array(model(jnp.asarray(opt.x))[1])
                    for model in (structure_model, aerodynamics_model, propulsion_model)])
report(z, range_, g)

scale = models.xu - models.xl
error = np.array([np.max(np.abs(h[Z_FROM_X] - solution["z"]) / scale) for h in opt.history])
print(f"\nRange (MDF): {float(solution['range']):.2f} nm")
print(f"Max design error relative to the bound widths: {error[-1]:.2e}")
print(f"Max coupling constraint violation: {np.max(np.abs(opt.phi)):.2e}")


# optimality and feasibility after every subproblem solve.
# Optimality oscillates because solving one block leaves the others non-stationary.
iterations = np.arange(1, len(opt.feas_history) + 1)  # opt_history[0] is nan; see ALBCD.opt_history

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 2.25))
ax1.semilogy(iterations, opt.opt_history, color='tab:blue', linewidth=2)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Optimality')
ax1.grid(color='lavender', alpha=0.5, axis='y')

ax2.semilogy(iterations, opt.feas_history, color='tab:orange', linewidth=2)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Feasibility')
ax2.grid(color='lavender', alpha=0.5, axis='y')

ax3.semilogy(error, linewidth=2, color='tab:red')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Design error')
ax3.grid(color='lavender', alpha=0.5, axis='y')

plt.tight_layout()
plt.show()
