"""Aircraft and trajectory model of one flight, shared by airliner_fleet.py and monolithic.py.

Each flight is the inverse-dynamics trajectory problem of CoTest's
airliner/inverse_dynamics_bsplines.py: the altitude h(r) and airspeed v(r) profiles are
cubic B-splines in range r, the point-mass equations of motion are solved at every point
for the thrust and angle of attack, and only mass and time are integrated (RK4 in range).
The path constraints are 0 <= throttle <= 1 and CL <= CL_limit at every RK4 node.

The variables of one flight are v = [h coefficients, v coefficients, tau], where tau is the
flight's block-time allocation (ks), tied to its simulated flight time by a local equality
constraint.
"""

import sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # modopt works in float64
import jax.numpy as jnp
from scipy.interpolate import BSpline
from atmos1976_smooth_jax import atmosphere  # US 1976 with the tropopause corner rounded over 1.5 km
from rk4_jax import rk4

# aircraft
AR = 9.0
S = 80.0         # m^2
e = 0.75
CD0 = 0.02
CL_alpha = 2 * np.pi
CLmax = 1.2      # tanh saturation of the lift curve
tmax_sl = 100000 # N
g = 9.81

# mission: start level at 10,000 ft and 160 m/s, end level at 10,000 ft with a free arrival speed
h0 = hf = 3048.0 # m
v0 = 160.0       # m/s
CL_limit = CLmax / 1.3**2  # 1.3 V_stall margin

# engine fuel flow: part-power formulation with idle fuel burn and a convex throttle response
g_idle = 0.1     # idle fuel flow as a fraction of the full-throttle fuel flow
eta_star = 0.9   # throttle setting with the lowest tsfc
g2 = g_idle / eta_star**2
g1 = 1 - 2 * np.sqrt(g_idle * g2)  # normalized so the minimum tsfc equals the reference tsfc

# discretization (the same for every flight; the nodes scale with its range)
n_int = 50       # spline intervals (control points per profile = n_int + 3)
nr = 300         # RK4 steps
A_cluster = (9.0, 150.0)  # refinement near both ends: climb/descent scale and maneuver scale
L_cluster = (200e3, 8e3)  # m

n_cp = n_int + 3
n_h = n_cp - 4   # free altitude coefficients (two at each end impose h and level flight)
n_v = n_cp - 1   # free airspeed coefficients (the first imposes v0)
nvar = n_h + n_v + 1  # variables per flight: [h coefficients, v coefficients, tau]

# bounds of the local constraints: 0 <= throttle <= 1, CL <= CL_limit, flight time = tau
cl = np.concatenate((np.zeros(nr + 1), np.full(nr + 1, -np.inf), [0.0]))
cu = np.concatenate((np.ones(nr + 1), np.full(nr + 1, CL_limit), [0.0]))

# bounds and scaling of one flight's variables [h coefficients, v coefficients, tau (ks)];
# the mission never goes below 10,000 ft
xl = np.concatenate((np.full(n_h, h0), np.full(n_v, 60.0), [1.0]))
xu = np.concatenate((np.full(n_h, 16000.0), np.full(n_v, 300.0), [100.0]))
x_scaler = np.concatenate((np.full(n_h, 1e-4), np.full(n_v, 1e-2), [1.0]))


def clustered(n, rf):
    """n intervals on [0, rf], refined near both ends where the climb, descent and end maneuvers happen."""
    def W(r):
        w = r
        for A, L in zip(A_cluster, L_cluster):
            w = w + A * L * (1 - np.exp(-r / L)) + A * L * (np.exp(-(rf - r) / L) - np.exp(-rf / L))
        return w
    r_fine = np.linspace(0, rf, 400001)
    return np.interp(np.linspace(0, W(rf), n + 1), W(r_fine), r_fine)


def setup_flight(rf, m0):
    """Arrays describing a flight of range rf (m) and takeoff mass m0 (kg): spline basis matrices at the
    RK4 nodes and midpoints, step sizes, mass, nodes, and the Greville abscissae (for the initial guess).
    Every flight's arrays have the same shapes, so one compiled function serves them all."""
    knots = np.concatenate(([0.0] * 3, clustered(n_int, rf), [rf] * 3))
    spline = BSpline(knots, np.eye(n_cp), 3)
    r_nodes = clustered(nr, rf)
    basis = lambda r: np.stack([spline.derivative(k)(r) if k else spline(r) for k in range(3)])
    greville = np.array([knots[i + 1:i + 4].mean() for i in range(n_cp)])
    fl = dict(B_nodes=basis(r_nodes), B_mid=basis(0.5 * (r_nodes[1:] + r_nodes[:-1])), dr=np.diff(r_nodes),
              m0=m0, r_nodes=r_nodes, greville=greville)
    return {k: jnp.asarray(a) for k, a in fl.items()}


def korn(CL, tc=0.15, kappa=0.8, csweep=np.cos(np.deg2rad(35))):
    MDD = kappa / csweep - tc / csweep**2 - CL / (10 * csweep**3)
    return MDD - (0.1 / 80)**(1/3)  # Mcrit


def profiles(v, B):
    """Columns h, dh/dr, d2h/dr2, v, dv/dr at the points where the basis matrices B were evaluated."""
    ch = jnp.concatenate((jnp.array([h0, h0]), v[:n_h], jnp.array([hf, hf])))
    cv = jnp.concatenate((jnp.array([v0]), v[n_h:n_h + n_v]))
    return jnp.stack((B[0] @ ch, B[1] @ ch, B[2] @ ch, B[0] @ cv, B[1] @ cv), axis=-1)


def flight_point(p, m):
    """Point-mass equations of motion solved for thrust and angle of attack."""
    h, hp, hpp, v, vp = p[..., 0], p[..., 1], p[..., 2], p[..., 3], p[..., 4]

    gamma = jnp.arctan(hp)
    r_dot = v * jnp.cos(gamma)
    v_dot = r_dot * vp
    gamma_dot = r_dot * hpp / (1 + hp**2)

    temp, _, rho, sos, _ = atmosphere(h)
    mach = v / sos
    q = 0.5 * rho * v**2

    Fx = m * (v_dot + g * jnp.sin(gamma))          # T cos(alpha) - D
    Fz = m * (v * gamma_dot + g * jnp.cos(gamma))  # T sin(alpha) + L

    # fixed-point iteration on the thrust-lift coupling (contraction ~ T / (q S CL_alpha) << 1)
    T = jnp.zeros_like(h)
    alpha = jnp.zeros_like(h)
    for _ in range(5):
        CL = (Fz - T * jnp.sin(alpha)) / (q * S)
        alpha = CLmax / CL_alpha * jnp.arctanh(jnp.clip(CL / CLmax, -0.999, 0.999))
        wave = 20 * (jnp.logaddexp(0.0, 50 * (mach - korn(CL))) / 50)**4  # softplus wave drag
        D = q * S * (CD0 + CL**2 / (np.pi * e * AR) + wave)
        T = (Fx + D) / jnp.cos(alpha)

    tmax = tmax_sl * (rho / 1.225)
    eta = T / tmax
    tsfc = (temp / 288.15)**0.5 * (0.4 + 0.45 * mach) * 2.832545e-5  # kg fuel / (s N)
    fuel_flow = tsfc * tmax * (g_idle + g1 * eta + g2 * eta**2)
    return dict(h=h, eta=eta, CL=CL, mach=mach), fuel_flow, r_dot


def simulate(v, fl):
    """Flight conditions at the RK4 nodes, with the mass (kg) and time (s) integrated in range."""
    Pn = profiles(v, fl["B_nodes"])
    Pm = profiles(v, fl["B_mid"])

    def deriv(y, p):
        _, fuel_flow, r_dot = flight_point(p, y[0])
        return jnp.array([-fuel_flow / r_dot, 1 / r_dot])  # d(m, t)/dr

    ys = rk4(deriv, jnp.array([fl["m0"], 0.0]), fl["dr"], Pn, Pm)
    out, _, _ = flight_point(Pn, ys[:, 0])
    out["m"], out["t"] = ys[:, 0], ys[:, 1]
    return out


def flight_outputs(v, fl):
    """One flight's local constraints followed by its fuel burned (t): [throttle and CL at every node,
    flight time (ks) - tau, fuel]. Fuel and constraints come from one simulation, so one forward-mode
    Jacobian gives both the constraint Jacobian and the fuel gradient."""
    out = simulate(v, fl)
    return jnp.concatenate((out["eta"], out["CL"],
                            jnp.array([1e-3 * out["t"][-1] - v[-1], 1e-3 * (fl["m0"] - out["m"][-1])])))


def initial_guess(fl):
    """Climb over ~250 km, cruise at 12 km and 180 m/s, descend over the last ~200 km; tau is the resulting flight time."""
    gr, rf = np.asarray(fl["greville"]), float(fl["r_nodes"][-1])
    ramp = lambda z: 0.5 * (1 - np.cos(np.pi * np.clip(z, 0, 1)))
    shape = ramp(gr / 250e3) * ramp((rf - gr) / 200e3)
    v = np.concatenate((h0 + (12000 - h0) * shape[2:-2], v0 + (180 - v0) * shape[1:], [0.0]))
    v[-1] = 1e-3 * float(simulate(jnp.asarray(v), fl)["t"][-1])
    return v


def peak_memory():
    """Peak resident memory of this process so far (MB)."""
    if sys.platform == "win32":
        import psutil
        return psutil.Process().memory_info().peak_wset / 2**20
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**10  # kB on Linux


def last_call_cache(f):
    """Memoize f on its most recent input, since SLSQP asks for the objective and constraints (and their
    derivatives) at the same point."""
    key, value = None, None

    def cached(x):
        nonlocal key, value
        if key is None or not np.array_equal(x, key):
            key, value = np.array(x, copy=True), f(x)
        return value
    return cached
