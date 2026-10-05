"""
U.S. Standard Atmosphere 1976 (0-86 km) as differentiable JAX functions of geometric altitude (m).

Temperature is piecewise linear in geopotential altitude; pressure follows from hydrostatics in each
layer (power law for lapse-rate layers, exponential for isothermal layers). Values are exact to the
standard, not interpolated. Note that the temperature gradient jumps at layer boundaries (e.g. the
tropopause at 11 km), so density and speed of sound have kinks there.
"""
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

g0 = 9.80665 # m/s^2
R = 8.31432 / 0.0289644 # J/(kg K), specific gas constant of air as defined in the standard
gamma = 1.4
r_earth = 6356766.0 # m, effective earth radius for geopotential altitude

# layer bases: geopotential altitude (m), temperature (K), lapse rate (K/m)
_Hb = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0])
_Lb = np.array([-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002])
_Tb = np.zeros(7)
_Pb = np.zeros(7)
_Tb[0], _Pb[0] = 288.15, 101325.0
for i in range(1, 7):
    dH = _Hb[i] - _Hb[i - 1]
    _Tb[i] = _Tb[i - 1] + _Lb[i - 1] * dH
    if _Lb[i - 1] == 0.0:
        _Pb[i] = _Pb[i - 1] * np.exp(-g0 * dH / (R * _Tb[i - 1]))
    else:
        _Pb[i] = _Pb[i - 1] * (_Tb[i - 1] / _Tb[i])**(g0 / (R * _Lb[i - 1]))


def atmosphere(h):
    """temperature (K), pressure (Pa), density (kg/m^3), speed of sound (m/s), viscosity (Pa s)"""
    h = jnp.asarray(h, dtype=float)
    H = r_earth * h / (r_earth + h) # geopotential altitude
    i = jnp.clip(jnp.searchsorted(jnp.asarray(_Hb), H, side='right') - 1, 0, 6)
    Hb, Tb, Lb, Pb = jnp.asarray(_Hb)[i], jnp.asarray(_Tb)[i], jnp.asarray(_Lb)[i], jnp.asarray(_Pb)[i]
    T = Tb + Lb * (H - Hb)
    isothermal = Lb == 0.0
    L_safe = jnp.where(isothermal, 1.0, Lb) # keeps the unused branch finite so gradients stay clean
    p_lapse = Pb * (Tb / jnp.where(isothermal, Tb, T))**(g0 / (R * L_safe))
    p_iso = Pb * jnp.exp(-g0 * (H - Hb) / (R * Tb))
    p = jnp.where(isothermal, p_iso, p_lapse)
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
    # spot checks against the published table (geometric altitude)
    table = [(0, 288.15, 101325.0, 1.2250), (5000, 255.68, 54048.0, 0.73643), (11000, 216.77, 22700.0, 0.36480),
             (15000, 216.65, 12111.0, 0.19476), (20000, 216.65, 5529.3, 0.088910), (30000, 226.51, 1197.0, 0.018410)]
    for h, T_ref, p_ref, rho_ref in table:
        T, p, rho, a, mu = atmosphere(h)
        print('h=%6.0f m  T=%.2f (%.2f)  p=%.1f (%.1f)  rho=%.5f (%.5f)  a=%.2f' % (h, T, T_ref, p, p_ref, rho, rho_ref, a))
    print('d(rho)/dh at 8 km:', jax.grad(density)(8000.0))
