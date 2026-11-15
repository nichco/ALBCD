# Distribution Statement A. Approved for public release: distribution is unlimited. Approved AFRL-2026-1671 28-09-2026.

"""
Powell's example of block coordinate descent failing on an unconstrained problem:

    min -x1 x2 - x2 x3 - x1 x3 + sum_i (max(xi - 1, 0)^2 + max(-xi - 1, 0)^2)
"""

import numpy as np
import matplotlib.pyplot as plt


def powell(x1, x2, x3):
    return (-x1*x2 - x2*x3 - x1*x3
            + np.maximum(x1 - 1, 0)**2 + np.maximum(-x1 - 1, 0)**2
            + np.maximum(x2 - 1, 0)**2 + np.maximum(-x2 - 1, 0)**2
            + np.maximum(x3 - 1, 0)**2 + np.maximum(-x3 - 1, 0)**2)


def minimize_block(x, i):
    """
    Minimizes Powell's function over x[i] with the other variables fixed.
    """
    s = np.sum(np.delete(x, i))
    return np.sign(s) * (1 + abs(s) / 2)


def gradient(x):
    """
    Gradient of Powell's function.
    """
    s = np.sum(x) - x
    return -s + 2 * np.maximum(x - 1, 0) - 2 * np.maximum(-x - 1, 0)


def bcd(x0, tol=1e-6, max_sweeps=6):
    """
    Cyclic block coordinate descent.
    """
    x = x0.copy()
    history = [x.copy()]
    for sweep in range(1, max_sweeps + 1):
        for i in range(len(x)):
            x[i] = minimize_block(x, i)
            history.append(x.copy())
        res = np.max(np.abs(gradient(x)))
        print(f'sweep {sweep:3d} | max |gradient| {res:.3e}')
        if res <= tol:
            break
    return np.array(history)


eps = 0.01
x0 = np.array([-1 - eps, 1 + 0.5 * eps, -1 - 0.25 * eps]) # near the vertex (-1, 1, -1)

history = bcd(x0)

for k, x in enumerate(history):
    print(f'{k:3d}  ' + '  '.join(f'{xi:+.8f}' for xi in x))

plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(projection='3d', computed_zorder=False)

# the faces of the cube [-1, 1]^3, colored by Powell's function
t = np.linspace(-1, 1, 21)
T1, T2 = np.meshgrid(t, t)
ones = np.ones_like(T1)
faces = [(s * ones, T1, T2) for s in (-1, 1)] + [(T1, s * ones, T2) for s in (-1, 1)] + [(T1, T2, s * ones) for s in (-1, 1)]
norm = plt.Normalize(-3, 1) # the range of Powell's function over the cube
for X1, X2, X3 in faces:
    ax.plot_surface(X1, X2, X3, facecolors=plt.cm.viridis(norm(powell(X1, X2, X3))), alpha=0.6, shade=False, linewidth=0, zorder=1)

# the edges of the cube
for a in (-1, 1):
    for b in (-1, 1):
        for edge in ([(-1, 1), (a, a), (b, b)], [(a, a), (-1, 1), (b, b)], [(a, a), (b, b), (-1, 1)]):
            ax.plot(*edge, color='k', linewidth=0.8, alpha=0.5, zorder=2)

ax.plot(history[:, 0], history[:, 1], history[:, 2], 'o-', color='tab:red', linewidth=2.5, markersize=6, mec='k', zorder=10, label='iterates')
ax.plot(*history[0], 's', color='tab:orange', markersize=10, mec='k', zorder=11, label='$x_0$')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
ticks = [-1, 0, 1]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)
ax.set_box_aspect((1, 1, 1))
ax.legend(loc='upper left')

plt.show()
