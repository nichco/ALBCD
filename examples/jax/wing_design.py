# Distribution Statement A. Approved for public release: distribution is unlimited. Approved AFRL-2026-1671 28-09-2026.

"""Unconstrained two-dimensional wing design (from Martins and Ning, Engineering
Design Optimization, https://mdobook.github.io/):

    min P(b, c)  s.t.  b >= 0, c >= 0.01

where P is the power a propeller aircraft needs for steady level flight with a
rectangular wing of span b and chord c: the viscous and induced drag times the
flight speed, divided by a propeller efficiency that peaks at v_bar.

Block 1 owns the span b and block 2 owns the chord c. There are no coupling
constraints, so ALBCD runs only its BCD inner loop (unconstrained=True). Each block
solve lands on the curve where that block is optimal (dP/db = 0 or dP/dc = 0), so the
iterates zigzag between the two curves toward the minimum.

Gradients come from JAX. Run this file to solve the problem, compare the result
with a monolithic SLSQP solve over both variables, and plot the iterates.
"""

import numpy as np
import modopt as mo
import jax
jax.config.update("jax_enable_x64", True) # jax defaults to float32
import jax.numpy as jnp
import matplotlib.pyplot as plt
from albcd import ALBCD, Subproblem
import warnings
warnings.filterwarnings("ignore")

rho = 1.2      # air density (kg/m^3)
visc = 1.8e-5  # dynamic viscosity of air (kg/(m s))
k = 1.2        # form factor
C_L = 0.4      # lift coefficient
e = 0.8        # span efficiency factor
W_0 = 1000     # weight excluding the wing (N)
W_s = 8        # wing weight per unit area (N/m^2)
eta_max = 0.8  # peak propeller efficiency
v_bar = 20     # flight speed of peak propeller efficiency (m/s)
sigma = 5      # width of the propeller efficiency peak (m/s)


def power_required(b, c):
    S = b * c
    W = W_0 + W_s * S
    v = (2 * W / (rho * C_L * S)) ** 0.5
    q = 0.5 * rho * v**2
    S_wet = 2.05 * S
    Re = rho * v * c / visc
    C_f = 0.074 / Re**0.2
    viscous_drag = k * C_f * q * S_wet
    induced_drag = W**2 / (q * np.pi * b**2 * e)
    eta = eta_max * jnp.exp(-(v - v_bar)**2 / (2 * sigma**2))
    return (viscous_drag + induced_drag) * v / eta


class WingSubproblem(Subproblem):
    """Minimizes the power over the one variable in index (0: span, 1: chord), with the other fixed."""

    XL = np.array([0.0, 1e-2])  # lower bounds on [b, c]

    def objective(self, v, other):
        b, c = (v[0], other[0]) if self.index[0] == 0 else (other[0], v[0])
        return jnp.squeeze(power_required(b, c))

    def solve(self, x, y, mu, data, outputs) -> None:
        v0 = np.asarray(self.decompose(x), dtype=float)
        other = jnp.asarray(self.other(x))

        jaxprob = mo.JaxProblem(x0=v0, jax_obj=lambda v: self.objective(v, other), order=1,
                                xl=self.XL[self.index], xu=np.inf)

        # a tight ftol, as in 2d_rosenbrock.py: near the solution a block solve lowers P by
        # less than a looser ftol, so SLSQP would stop without moving and BCD would stall
        optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 300, 'ftol': 1e-12}, turn_off_outputs=True)
        optimizer.solve()

        outputs["x"] = self.recompose(x, optimizer.results['x'])

    def residual(self, x, y, mu, data) -> float:
        v = jnp.asarray(self.decompose(x))
        other = jnp.asarray(self.other(x))
        grad_f = np.asarray(jax.grad(self.objective)(v, other))

        # at the lower bound only a negative gradient (pointing into the feasible region) counts
        if np.asarray(v)[0] <= self.XL[self.index] + 1e-8:
            grad_f = np.minimum(grad_f, 0.0)
        return float(np.max(np.abs(grad_f)))


opt = ALBCD(subproblems=[WingSubproblem(index=np.array([0])),  # owns the span b
                         WingSubproblem(index=np.array([1]))], # owns the chord c
            x0=np.array([15.0, 1.2]),
            unconstrained=True,
            opt_tol=1e-3,
            max_inner_iter=300)

opt.solve()

print('Solution (span, chord): ', opt.x)
print('Power (W): ', float(power_required(*opt.x)))

# monolithic reference: the same problem solved with one SLSQP over both variables
jaxprob = mo.JaxProblem(x0=np.array([15.0, 1.2]), jax_obj=lambda x: power_required(x[0], x[1]),
                        xl=WingSubproblem.XL, xu=np.full(2, np.inf))
optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 300, 'ftol': 1e-12}, turn_off_outputs=True)
optimizer.solve()
x_star = optimizer.results['x']
print('Monolithic solution (span, chord): ', x_star)
print('Relative error: ', np.linalg.norm(opt.x - x_star) / np.linalg.norm(x_star))


history = np.array(opt.x_history)
b_history = history[:, 0]
c_history = history[:, 1]

plt.figure(figsize=(4, 4))

b_vals = np.linspace(5, 35, 100)
c_vals = np.linspace(0.3, 1.5, 100)
B, C = np.meshgrid(b_vals, c_vals)
Z = np.asarray(power_required(B, C))

levels = np.linspace(Z.min(), 3000, 30)
plt.contour(B, C, Z, levels=levels, cmap='Blues_r', alpha=0.4, linewidths=0.5)
plt.contourf(B, C, Z, levels=levels, cmap='Blues_r', alpha=0.5)

dZ_dc, dZ_db = np.gradient(Z, c_vals, b_vals)

# zero level sets of the derivatives: the curves where block 1 and block 2 are optimal
plt.contour(B, C, dZ_db, levels=[0], colors='tab:purple', linewidths=2, linestyles='-.', alpha=1)
plt.contour(B, C, dZ_dc, levels=[0], colors='tab:olive', linewidths=2, linestyles='-.', alpha=1)

plt.plot(b_history, c_history, '-o', mec='k', color='tab:red', linewidth=2.5, markersize=7, zorder=10)
plt.xlim(5, 35)
plt.ylim(0.3, 1.5)
plt.xlabel('Wing span (m)')
plt.ylabel('Chord (m)')

plt.show()
