"""Sobieski's supersonic business jet (SSBJ) problem: data, disciplines, coupled analysis and reporting.

Shared by the monolithic formulations in ssbj.py (MDF), ssbj_idf.py (IDF) and ssbj_sand.py (SAND),
and the distributed ones in ssbj_albcd.py (ALBCD), ssbj_co.py (CO) and ssbj_bliss.py (BLISS).
Each maximizes the Breguet range of a supersonic business jet over 10 design variables,
subject to 12 inequality constraints, with the three disciplines coupled as

    structure    -> total weight, wing twist          -> aerodynamics
    aerodynamics -> lift (= total weight), drag       -> structure, propulsion
    propulsion   -> engine weight, engine scale factor -> structure, aerodynamics

The test problem was defined by Sobieszczanski-Sobieski, Agte and Sandusky in NASA/TM-1998-208715,
the paper that introduced the BLISS decomposition method, but nothing here uses BLISS: the
disciplinary analyses are the problem's own. The models follow GEMSEO's version
(gemseo.problems.mdo.sobieski), which the Python version by ONERA in WhatsOpt/SSBJ-OpenMDAO
matches. Units are lb, ft, ft^2 and nautical miles.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # modopt works in float64
import jax.numpy as jnp

C0 = 2000.0     # minimum fuel weight
C1 = 25000.0    # miscellaneous weight
C2 = 6.0        # maximum load factor
C3 = 4360.0     # reference engine weight
C4 = 0.01375    # minimum drag coefficient

# z = [x_shared, x_1, x_2, x_3]
names = ["thickness/chord", "altitude (ft)", "Mach number", "aspect ratio", "sweep (deg)",
         "wing area (ft^2)", "taper ratio", "wingbox area", "skin friction", "throttle"]
x0 = np.array([0.05, 45000.0, 1.6, 5.5, 55.0, 1000.0, 0.25, 1.0, 1.0, 0.5])
xl = np.array([0.01, 30000.0, 1.4, 2.5, 40.0, 500.0, 0.1, 0.75, 0.75, 0.1])
xu = np.array([0.09, 60000.0, 1.8, 8.5, 70.0, 1500.0, 0.4, 1.25, 1.25, 1.0])

# BLISS's optimum, as reported in GEMSEO. Its taper ratio is short of the upper bound
# 0.4, where these models give a longer range: 3963.38 nm, against 3963.17 nm at z_bliss.
z_bliss = np.array([0.06, 60000.0, 1.4, 2.5, 70.0, 1500.0, 0.38757, 0.75, 0.75, 0.15624])
range_bliss = 3963.98


# Part of the SSBJ analyses is defined by fixed quadratic polynomials in place of physics
# models: the wing twist, the five wing stresses, the pressure gradient and the engine
# temperature, and the effects of wingbox area, skin friction, engine scale factor and twist
# on the wing weight and drag. They aren't surrogates fitted to anything, and the wingbox
# area and skin friction enter the problem only through them. Each variable s_i is normalized
# by its value at x0 and clipped, z_i = clip(s_i / s_ref_i, 0.75, 1.25) - 1, and the
# polynomial is 1 + a.z + z.A.z / 2 with a and A set by the trend in each variable.
CROSS = np.array([[0.2736, 0.3970, 0.8152, 0.9230, 0.1108],
                  [0.4252, 0.4415, 0.6357, 0.7435, 0.1138],
                  [0.0329, 0.8856, 0.8390, 0.3657, 0.0019],
                  [0.0878, 0.7248, 0.1978, 0.0200, 0.0169],
                  [0.8955, 0.4568, 0.8075, 0.9239, 0.2525]])

# the polynomial's values at z_i = -bound_i and +bound_i, for each trend
TRENDS = {"linear increasing": (0.95, 1.05), "nonlinear increasing": (0.95, 1.1),
          "linear decreasing": (1.05, 0.95), "nonlinear decreasing": (1.05, 0.9),
          "parabolic": (1.0025, 1.0025)}


def polynomial_function(s_ref, trends, bound):
    """The polynomial through the trends in each variable, as a function of the variables."""
    n = len(trends)
    bound = np.broadcast_to(bound, n)
    a, A = np.zeros(n), np.zeros((n, n))
    for i, trend in enumerate(trends):
        f_lo, f_hi = TRENDS[trend]
        a[i] = (f_hi - f_lo) / (2 * bound[i])
        A[i, i] = (f_lo + f_hi - 2) / (2 * bound[i]**2)
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = A[j, i] = A[i, i] * CROSS[i, j]
    s_ref = np.asarray(s_ref)

    def poly(*s):
        z = jnp.clip(jnp.stack(s) / s_ref, 0.75, 1.25) - 1.0
        return 1.0 + a @ z + 0.5 * z @ A @ z
    return poly


half_span0 = 0.5 * np.sqrt(x0[3] * x0[5])
aero_center0 = (1 + 2 * x0[6]) / (3 * (1 + x0[6]))
# As in GEMSEO, the lift's reference value is 1 lb, so its normalized value is always
# clipped at the upper end and its terms in the twist and stress polynomials are constant.
twist_poly = polynomial_function(      # (wingbox area, half-span, aero center, lift)
    [x0[7], half_span0, aero_center0, 1.0],
    ["nonlinear increasing", "nonlinear decreasing", "nonlinear decreasing", "linear decreasing"], 0.25)
wing_weight_poly = polynomial_function([x0[7]], ["linear increasing"], 0.008)   # (wingbox area)
stress_polys = [polynomial_function(   # (t/c, lift, wingbox area, half-span, aero center)
    [x0[0], 1.0, x0[7], half_span0, aero_center0],
    ["nonlinear decreasing", "linear increasing", "nonlinear decreasing", "linear increasing",
     "linear increasing"], 0.1 + 0.05 * i) for i in range(5)]
cd_min_poly = polynomial_function([1.0, x0[8]], ["linear increasing"] * 2, 0.25)  # (ESF, skin friction)
twist_drag_poly = polynomial_function([1.0], ["parabolic"], 0.25)                 # (twist)
pressure_gradient_poly = polynomial_function([x0[0]], ["linear increasing"], 0.25)  # (t/c)
temperature_poly = polynomial_function(  # (Mach number, altitude, throttle)
    [x0[2], x0[1], x0[9]], ["nonlinear increasing", "nonlinear decreasing", "nonlinear increasing"], 0.25)


def atmosphere(h, M):
    """Air density (slug/ft^3), flight speed (ft/s) and sqrt of the temperature ratio at altitude h."""
    troposphere = h < 36089.0
    rho = jnp.where(troposphere, 2.377e-3 * (1 - 6.875e-6 * h)**4.2561,
                    2.377e-3 * 0.2971 * jnp.exp((36089.0 - h) / 20806.7))
    v = M * jnp.where(troposphere, 1116.39 * jnp.sqrt(1 - 6.875e-6 * h), 968.1)
    sqrt_theta = jnp.sqrt(jnp.where(troposphere, 1 - 6.875e-6 * h, 0.7519))
    return rho, v, sqrt_theta


def structure(x_shared, x_1, lift, engine_weight):
    """Total weight, fuel weight, wing twist and the five wing stresses."""
    tc, _, _, AR, sweep, S = x_shared
    taper, wingbox = x_1
    half_span = 0.5 * jnp.sqrt(AR * S)
    aero_center = (1 + 2 * taper) / (3 * (1 + taper))

    twist = twist_poly(wingbox, half_span, aero_center, lift)
    wing_weight = (0.0051 * (lift * C2)**0.557 * S**0.649 * jnp.sqrt(AR) * tc**-0.4 * (1 + taper)**0.1
                   * (0.1875 * S)**0.1 / jnp.cos(jnp.deg2rad(sweep)) * wing_weight_poly(wingbox))
    thickness = tc * S / jnp.sqrt(AR * S)
    fuel_weight = C0 + (5 * S / 18) * (2 / 3 * thickness) * 42.5
    total_weight = C1 + wing_weight + fuel_weight + engine_weight
    stress = jnp.stack([s(tc, lift, wingbox, half_span, aero_center) for s in stress_polys])
    return total_weight, fuel_weight, twist, stress


def aerodynamics(x_shared, cf, total_weight, twist, esf):
    """Lift, drag, lift-to-drag ratio and the adverse pressure gradient."""
    tc, h, M, _, sweep, S = x_shared
    rho, v, _ = atmosphere(h, M)
    q = 0.5 * rho * v**2
    lift = total_weight
    CL = lift / (q * S)
    CD_min = C4 * cd_min_poly(esf, cf) + 3.05 * tc**(5 / 3) * jnp.cos(jnp.deg2rad(sweep))**1.5
    k = (M**2 - 1) * jnp.cos(jnp.deg2rad(sweep)) / (4 * jnp.sqrt(sweep**2 - 1) - 2)  # sweep in deg, as in BLISS
    CD = twist_drag_poly(twist) * (CD_min + k * CL**2)
    drag = q * CD * S
    return lift, drag, CL / CD, pressure_gradient_poly(tc)


def propulsion(x_shared, throttle, drag):
    """Specific fuel consumption, engine weight, engine scale factor, engine temperature
    and the throttle setting relative to its upper limit."""
    h, M = x_shared[1], x_shared[2]
    T = 16168.6 * throttle
    sfc = (1.13238425638512 + 1.53436586044561 * M - 3.295564466e-5 * h - 1.6378694115e-4 * T
           - 0.31623315541888 * M**2 + 2 * 4.10691343e-6 * h * M - 2 * 5.24800059e-5 * T * M
           - 8.574e-11 * h**2 + 2 * 1.90214e-9 * T * h + 1.059951e-8 * T**2)
    T_max = (11483.7822254806 + 10856.2163466548 * M - 0.5080237941 * h + 3200.157926969 * M**2
             - 2 * 0.1466251679 * M * h + 6.8572e-6 * h**2)
    esf = drag / (3 * T)
    engine_weight = 3 * C3 * esf**1.05
    return sfc, engine_weight, esf, temperature_poly(M, h, throttle), T / T_max


def breguet_range(x_shared, total_weight, fuel_weight, lift_to_drag, sfc):
    """Range (nm) from the Breguet equation."""
    h, M = x_shared[1], x_shared[2]
    _, _, sqrt_theta = atmosphere(h, M)
    return M * lift_to_drag * 661.0 * sqrt_theta / sfc * jnp.log(total_weight / (total_weight - fuel_weight))


def disciplines(u, z):
    """Structure, aerodynamics and propulsion in sequence, from the couplings u = [lift, engine weight, ESF].

    Returns the updated couplings and every output.
    """
    x_shared, x_1, cf, throttle = z[:6], z[6:8], z[8], z[9]
    lift, engine_weight, esf = u
    total_weight, fuel_weight, twist, stress = structure(x_shared, x_1, lift, engine_weight)
    lift, drag, lift_to_drag, pressure_gradient = aerodynamics(x_shared, cf, total_weight, twist, esf)
    sfc, engine_weight, esf, temperature, throttle_ratio = propulsion(x_shared, throttle, drag)
    outputs = dict(total_weight=total_weight, fuel_weight=fuel_weight, twist=twist, stress=stress,
                   drag=drag, lift_to_drag=lift_to_drag, pressure_gradient=pressure_gradient, sfc=sfc,
                   engine_weight=engine_weight, esf=esf, temperature=temperature, throttle_ratio=throttle_ratio,
                   range=breguet_range(x_shared, total_weight, fuel_weight, lift_to_drag, sfc))
    return jnp.stack([lift, engine_weight, esf]), outputs


u0 = jnp.array([50000.0, 6000.0, 0.5])


def gauss_seidel(residual, u):
    """Iterates u <- u + residual(u), a Gauss-Seidel pass through the disciplines, to convergence."""
    def not_converged(state):
        u, du, k = state
        return (jnp.max(jnp.abs(du / u)) > 1e-14) & (k < 200)

    def iteration(state):
        u, _, k = state
        du = residual(u)
        return u + du, du, k + 1

    return jax.lax.while_loop(not_converged, iteration, (u, jnp.full_like(u, jnp.inf), 0))[0]


def analysis(z):
    """Every output at multidisciplinary feasibility.

    custom_root differentiates the converged couplings implicitly, rather than through
    the iterations, by solving with the Jacobian of the residual at the solution.
    """
    u = jax.lax.custom_root(lambda u: disciplines(u, z)[0] - u, u0, gauss_seidel,
                            lambda g, y: jnp.linalg.solve(jax.jacfwd(g)(y), y))
    return disciplines(u, z)[1]


def design_constraints(stress, twist, pressure_gradient, esf, throttle_ratio, temperature):
    """[stresses (5), twist, pressure gradient, ESF, throttle / max throttle, temperature], bounded by cl and cu."""
    return jnp.concatenate([stress, jnp.stack([twist, pressure_gradient, esf, throttle_ratio, temperature])])


con_names = [f"stress {i + 1}" for i in range(5)] + ["twist", "pressure gradient", "ESF", "throttle/max",
                                                     "temperature"]
cl = np.array([-np.inf] * 5 + [0.8, -np.inf, 0.5, -np.inf, -np.inf])
cu = np.array([1.09] * 5 + [1.04, 1.04, 1.5, 1.0, 1.02])


def report(z, range_, g):
    """Prints the design z against BLISS's, its range and its design constraints g."""
    print(f"\n{'Design variable':<18}{'Initial':>12}{'Optimal':>12}{'BLISS':>12}{'Bounds':>20}")
    for name, a, b, c, lo, hi in zip(names, x0, z, z_bliss, xl, xu):
        print(f"{name:<18}{a:>12.5g}{b:>12.5g}{c:>12.5g}{f'[{lo:g}, {hi:g}]':>20}")

    print(f"\nRange (nm): {range_:.2f} optimal, {range_bliss:.2f} BLISS")
    print(f"Relative difference from BLISS's range: {abs(range_ - range_bliss) / range_bliss:.2e}")

    print(f"\n{'Constraint':<18}{'Value':>12}{'Bounds':>20}")
    for name, value, lo, hi in zip(con_names, g, cl, cu):
        active = "  active" if np.isclose(value, lo, atol=1e-6) or np.isclose(value, hi, atol=1e-6) else ""
        print(f"{name:<18}{value:>12.6f}{f'[{lo:g}, {hi:g}]':>20}{active}")
