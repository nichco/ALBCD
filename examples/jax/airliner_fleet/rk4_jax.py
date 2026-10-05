import jax
import jax.numpy as jnp


def rk4(f, y0, h, u_nodes, u_mid):
    """
    Classical RK4 for y' = f(y, u) with inputs u sampled at the step nodes and midpoints.

    f       : f(y, u) -> dy, where u is one row of u_nodes / u_mid
    y0      : initial state
    h       : step size, scalar or one value per step
    u_nodes : inputs at the n + 1 nodes, shape (n + 1, ...)
    u_mid   : inputs at the n step midpoints, shape (n, ...)
    returns the state at the nodes, shape (n + 1, ...)
    """
    y0 = jnp.asarray(y0)
    h = jnp.broadcast_to(h, (u_mid.shape[0],))

    def step(y, xs):
        u0, um, u1, dt = xs
        k1 = f(y, u0)
        k2 = f(y + 0.5 * dt * k1, um)
        k3 = f(y + 0.5 * dt * k2, um)
        k4 = f(y + dt * k3, u1)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return y, y

    _, ys = jax.lax.scan(step, y0, (u_nodes[:-1], u_mid, u_nodes[1:], h))
    return jnp.vstack((y0, ys))
