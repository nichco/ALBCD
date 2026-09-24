"""Setup of the uncertain cart-pole problem, shared by cart_pole.py and monolithic.py.

A cart-pole swing-up, transcribed with trapezoidal collocation: in a fixed time, the
pole swings from hanging below the cart (theta = pi) to balancing above it (theta = 0)
while the cart moves a distance d. The pole's length l and mass mp are designed
together with the trajectory, and must work for N scenarios of the uncertain gravity
and cart and pole friction, sampled with a Latin hypercube. Each scenario has its own
trajectory (states and cart force), while all scenarios share the pole design.

The variables of one scenario are v = [l, mp, states, u], where states is the 4 x n
array [x, theta, dx, dtheta] at the n collocation nodes (flattened row by row) and u
is the cart force at the nodes.
"""

import sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # modopt works in float64
import jax.numpy as jnp
from scipy.stats import qmc

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2  # number of scenarios, e.g. `python cart_pole.py 3`

# uncertain parameters of each scenario: gravity (m/s^2), cart friction, pole friction
nominal = np.array([9.81, 0.03, 0.03])
spread = np.array([0.1, 0.01, 0.01])
samples = qmc.scale(qmc.LatinHypercube(d=3, seed=0).random(N), nominal - spread, nominal + spread)

n = 30         # collocation nodes
dt = 2.0 / n   # time step (s)
mc = 2.0       # cart mass (kg)
d = 0.8        # cart travel (m)
state_initial = jnp.array([0.0, np.pi, 0.0, 0.0])  # [x, theta, dx, dtheta]: pole hanging down
state_final = jnp.array([d, 0.0, 0.0, 0.0])        # pole upright, cart moved by d

nv = 2 + 5 * n  # variables per scenario

# bounds on [l, mp, states, u]
xl = np.concatenate([[0.1, 0.1], np.full(4 * n, -np.inf), np.full(n, -50.0)])
xu = np.concatenate([[5.0, 3.0], np.full(4 * n, np.inf), np.full(n, 50.0)])

# initial guess: states interpolated linearly between the boundary states, no force
v0 = np.concatenate([[0.5, 0.4], np.linspace(0, d, n), np.linspace(np.pi, 0, n), np.zeros(2 * n), np.zeros(n)])

# SLSQP scaling of [l, mp, states, u] and of the local constraints [defects, boundary states]
x_scaler = np.concatenate([np.ones(2 + 4 * n), np.full(n, 0.1)])
c_scaler = np.concatenate([np.full(4 * (n - 1), 10.0), np.ones(8)])


def effort(v):
    """Control effort of one scenario: the integral of u^2 by the trapezoidal rule."""
    u = v[2 + 4 * n:]
    return 0.5 * dt * jnp.sum(u[:-1]**2 + u[1:]**2)


def dynamics(states, u, l, mp, g, mu_cart, mu_pole):
    """Time derivatives of states = [x, theta, dx, dtheta] (4 x n) under the cart force u."""
    theta, dx, dtheta = states[1], states[2], states[3]
    l_hat = l / 2  # distance from the pivot to the pole's center of mass
    sin, cos = jnp.sin(theta), jnp.cos(theta)

    ddx = ((mp * g * sin * cos - 7 / 3 * (u + mp * l_hat * dtheta**2 * sin - mu_cart * dx)
            - mu_pole * dtheta * cos / l_hat)
           / (mp * cos**2 - 7 / 3 * (mc + mp)))
    ddtheta = 3 * (g * sin - ddx * cos - mu_pole * dtheta / (mp * l_hat)) / (7 * l_hat)

    return jnp.stack([dx, dtheta, ddx, ddtheta])


def collocation(v, sample):
    """Local constraints of one scenario, all = 0: the trapezoidal collocation defects
    of the dynamics, and the initial and final states."""
    l, mp = v[0], v[1]
    states = v[2:2 + 4 * n].reshape(4, n)
    u = v[2 + 4 * n:]

    f = dynamics(states, u, l, mp, *sample)
    defects = states[:, 1:] - states[:, :-1] - 0.5 * dt * (f[:, 1:] + f[:, :-1])

    return jnp.concatenate([defects.ravel(), states[:, 0] - state_initial, states[:, -1] - state_final])
