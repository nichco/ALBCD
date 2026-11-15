# Distribution Statement A. Approved for public release: distribution is unlimited. Approved AFRL-2026-1671 28-09-2026.

"""Unconstrained two-dimensional wing design (from Martins and Ning, Engineering
Design Optimization, https://mdobook.github.io/):

    min P(b, c)  s.t.  b >= 0, c >= 0.01

where P is the power a propeller aircraft needs for steady level flight with a
rectangular wing of span b and chord c: the viscous and induced drag times the
flight speed, divided by a propeller efficiency that peaks at v_bar.

Block 1 owns the span b and block 2 owns the chord c, and a plain cyclic block
coordinate descent loop is written out below. Each block solve lands on the curve where that block is optimal (dP/db = 0 or dP/dc = 0), so the
iterates zigzag between the two curves toward the minimum.

Gradients come from PyTorch. Run this file to solve the problem, compare the result
with a monolithic SLSQP solve over both variables, and plot the iterates.
"""

import numpy as np
import modopt as mo
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# modopt and SLSQP work in float64; torch defaults to float32
torch.set_default_dtype(torch.float64)

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
    eta = eta_max * torch.exp(-(v - v_bar)**2 / (2 * sigma**2))
    return (viscous_drag + induced_drag) * v / eta


XL = np.array([0.0, 1e-2])  # lower bounds on [b, c]


def f(x):
    return power_required(x[0], x[1])


def minimize_block(x, i):
    """
    Minimizes the power over x[i] (0: span, 1: chord) with the other variable fixed.
    """
    xt = torch.tensor(x)
    obj = lambda v: f(torch.cat([xt[:i], v, xt[i+1:]]))
    prob = mo.ProblemLite(x0=x[i:i+1].copy(),
                          obj=lambda v: np.float64(obj(torch.as_tensor(v))),
                          grad=lambda v: np.array(torch.func.grad(obj)(torch.as_tensor(v))),
                          xl=XL[i:i+1], xu=np.inf)

    # a tight ftol, as in 2d_rosenbrock.py: near the solution a block solve lowers P by
    # less than a looser ftol, so SLSQP would stop without moving and BCD would stall
    optimizer = mo.SLSQP(prob, solver_options={'maxiter': 300, 'ftol': 1e-12}, turn_off_outputs=True)
    optimizer.solve()
    return optimizer.results['x'][0]


def projected_gradient(x):
    """
    Gradient of the power; at a lower bound only a negative entry (pointing into the feasible region) counts.
    """
    g = np.array(torch.func.grad(f)(torch.tensor(x)))
    return np.where(x <= XL + 1e-8, np.minimum(g, 0.0), g)


def bcd(x0, tol=1e-3, max_sweeps=300):
    """
    Cyclic block coordinate descent.
    """
    x = x0.copy()
    history = [x.copy()]
    for sweep in range(1, max_sweeps + 1):
        for i in range(len(x)):
            x[i] = minimize_block(x, i)
            history.append(x.copy())
        res = np.max(np.abs(projected_gradient(x)))
        print(f'sweep {sweep:3d} | max |gradient| {res:.3e}')
        if res <= tol:
            break
    return np.array(history)


history = bcd(np.array([15.0, 1.2]))
x = history[-1]

print('Solution (span, chord): ', x)
print('Power (W): ', float(f(torch.tensor(x))))

# monolithic reference: the same problem solved with one SLSQP over both variables
prob = mo.ProblemLite(x0=np.array([15.0, 1.2]),
                      obj=lambda x: np.float64(f(torch.as_tensor(x))),
                      grad=lambda x: np.array(torch.func.grad(f)(torch.as_tensor(x))),
                      xl=XL, xu=np.full(2, np.inf))
optimizer = mo.SLSQP(prob, solver_options={'maxiter': 300, 'ftol': 1e-12}, turn_off_outputs=True)
optimizer.solve()
x_star = optimizer.results['x']
print('Monolithic solution (span, chord): ', x_star)
print('Relative error: ', np.linalg.norm(x - x_star) / np.linalg.norm(x_star))

b_history = history[:, 0]
c_history = history[:, 1]

plt.figure(figsize=(4, 4))

b_vals = np.linspace(5, 35, 100)
c_vals = np.linspace(0.3, 1.5, 100)
B, C = np.meshgrid(b_vals, c_vals)
Z = np.asarray(power_required(torch.as_tensor(B), torch.as_tensor(C)))

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
