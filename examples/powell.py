# Distribution Statement A. Approved for public release: distribution is unlimited. Approved AFRL-2026-1671 28-09-2026.

"""Powell's example of block coordinate descent failing on an unconstrained problem:

    min -x1 x2 - x2 x3 - x1 x3 + sum_i (max(xi - 1, 0)^2 + max(-xi - 1, 0)^2)

Each of the three blocks owns one variable, and its minimizer and stationarity
residual are both explicit, so no optimizer or automatic differentiation is needed.
There are no coupling constraints, so ALBCD runs only its BCD inner loop
(unconstrained=True). Started near the vertex (-1, 1, -1) of the cube [-1, 1]^3,
exact block minimization never converges: the iterates cycle around six of the
cube's vertices, where the gradient is not zero. See M. J. D. Powell, "On search
directions for minimization algorithms", Mathematical Programming 4, 193-201 (1973).

Run this file to solve the problem and plot the iterates.
"""

import numpy as np
import matplotlib.pyplot as plt
from albcd import ALBCD, Subproblem


def powell(x1, x2, x3):
    """Powell's function, which is symmetric in its three arguments."""
    return (-x1*x2 - x2*x3 - x1*x3
            + np.maximum(x1 - 1, 0)**2 + np.maximum(-x1 - 1, 0)**2
            + np.maximum(x2 - 1, 0)**2 + np.maximum(-x2 - 1, 0)**2
            + np.maximum(x3 - 1, 0)**2 + np.maximum(-x3 - 1, 0)**2)


class CoordinateSubproblem(Subproblem):
    """Minimizes Powell's function over one variable with the other two fixed.

    The function is symmetric, so one class serves all three blocks.
    """

    def setup(self) -> None:
        self.add_input("x")  # [x1, x2, x3]
        self.add_output("x") # no "phi" output: there are no coupling constraints

    def solve(self, inputs, outputs) -> None:
        x = inputs["x"]

        # With the other two variables fixed and s their sum, the objective is -v s
        # plus the penalties on v, up to a constant: linear in v on [-1, 1] and
        # quadratic outside. It is minimized by v = 1 + s/2 for s > 0, by
        # v = -1 + s/2 for s < 0, and by any v in [-1, 1] (here v = 0) for s = 0
        s = np.sum(self.other(x))
        v = np.sign(s) * (1 + abs(s) / 2)

        outputs["x"] = self.recompose(x, v)

    def residual(self, inputs) -> float:

        x = inputs["x"]

        v = self.decompose(x)[0]
        s = np.sum(self.other(x))

        # v is unbounded, so the stationarity residual is just the derivative of the
        # objective with respect to v: -s from the bilinear terms plus the penalties'
        grad_f = -s + 2 * max(v - 1, 0) - 2 * max(-v - 1, 0)
        return float(abs(grad_f))


eps = 0.01
x0 = np.array([-1 - eps, 1 + 0.5 * eps, -1 - 0.25 * eps]) # near the vertex (-1, 1, -1)

opt = ALBCD(subproblems=[CoordinateSubproblem(index=slice(0, 1)),  # owns x1
                         CoordinateSubproblem(index=slice(1, 2)),  # owns x2
                         CoordinateSubproblem(index=slice(2, 3))], # owns x3
            x0=x0,
            unconstrained=True,
            opt_tol=1e-6,
            max_inner_iter=6)

opt.solve()

print('\nIterates (one per block solve)')
for k, x in enumerate(opt.history):
    print(f'{k:3d}  ' + '  '.join(f'{xi:+.8f}' for xi in x))


history = np.array(opt.history)

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
