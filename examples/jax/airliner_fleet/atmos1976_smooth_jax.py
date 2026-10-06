"""
U.S. Standard Atmosphere 1976 (0-20 km) with the tropopause corner rounded, as differentiable JAX functions
of geometric altitude (m).

The standard's temperature is piecewise linear in geopotential altitude H, with a lapse rate of -6.5 K/km
up to 11 km and an isothermal layer at 216.65 K above it. The jump in the temperature gradient at 11 km puts
kinks in density and speed of sound, which can make trajectory optima depend on the corner rather than on
the atmosphere. Here the corner is replaced by a softplus of width w:

    T(H) = 216.65 + 0.0065 w softplus((11000 - H) / w)

which tends to the standard away from 11 km and lies above it by at most 0.0065 w ln 2 (6.8 K at w = 1.5 km)
at the tropopause. Pressure follows from hydrostatics, dln(p)/dH = -g0 / (R T): it is integrated once onto a
table and interpolated with cubic Hermite polynomials that use the exact slope, so density and speed of sound
are smooth. Valid from 0 to 20 km (the standard's next layer, which warms with altitude, is not modeled).
"""
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

g0 = 9.80665 # m/s^2
R = 8.31432 / 0.0289644 # J/(kg K), specific gas constant of air as defined in the standard
gamma = 1.4
r_earth = 6356766.0 # m, effective earth radius for geopotential altitude

w = 1500.0 # m, width of the rounded tropopause
T_strat, lapse, H_trop = 216.65, 0.0065, 11000.0


def _temperature(H, xp=jnp):
    return T_strat + lapse * w * xp.logaddexp(0.0, (H_trop - H) / w)


# ln(p) on a table of geopotential altitudes, integrated on a 1 m grid from 101325 Pa at sea level
_dH = 100.0
_H_fine = np.linspace(0.0, 20000.0, 20001)
_slope_fine = -g0 / (R * _temperature(_H_fine, np))
_lnp_fine = np.log(101325.0) + np.concatenate(([0.0], np.cumsum(0.5 * (_slope_fine[1:] + _slope_fine[:-1]))))
_H = jnp.asarray(_H_fine[::int(_dH)])
_lnp = jnp.asarray(_lnp_fine[::int(_dH)])
_slope = jnp.asarray(_slope_fine[::int(_dH)])


def _log_pressure(H):
    """Cubic Hermite interpolation of ln(p) with the exact hydrostatic slopes at the table points."""
    i = jnp.clip(jnp.floor(H / _dH).astype(int), 0, len(_H) - 2)
    s = (H - _H[i]) / _dH
    h00, h10, h01, h11 = 2*s**3 - 3*s**2 + 1, s**3 - 2*s**2 + s, -2*s**3 + 3*s**2, s**3 - s**2
    return h00 * _lnp[i] + h10 * _dH * _slope[i] + h01 * _lnp[i + 1] + h11 * _dH * _slope[i + 1]


def atmosphere(h):
    """temperature (K), pressure (Pa), density (kg/m^3), speed of sound (m/s), viscosity (Pa s)"""
    h = jnp.asarray(h, dtype=float)
    H = r_earth * h / (r_earth + h) # geopotential altitude
    T = _temperature(H)
    p = jnp.exp(_log_pressure(H))
    rho = p / (R * T)
    a = jnp.sqrt(gamma * R * T)
    mu = 1.458e-6 * T**1.5 / (T + 110.4) # Sutherland's law
    return T, p, rho, a, mu


def temperature(h):
    return atmosphere(h)[0]


def pressure(h):
    return atmosphere(h)[1]


def density(h):
    return atmosphere(h)[2]


def speed_of_sound(h):
    return atmosphere(h)[3]


def viscosity(h):
    return atmosphere(h)[4]


if __name__ == "__main__":
    # comparison with the sharp-cornered standard
    from atmos1976_jax import atmosphere as atmosphere_1976
    for h in [0, 5000, 9000, 10000, 11000, 12000, 13000, 15000, 20000]:
        T, p, rho, a, _ = atmosphere(h)
        T0, p0, rho0, a0, _ = atmosphere_1976(h)
        print('h=%6.0f m  T=%.2f (%.2f)  p=%.1f (%.1f)  rho=%.5f (%.5f)  a=%.2f (%.2f)' % (h, T, T0, p, p0, rho, rho0, a, a0))
    print('d(rho)/dh at 8 km:', jax.grad(density)(8000.0))
