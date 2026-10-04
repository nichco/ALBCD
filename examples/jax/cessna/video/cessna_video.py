# Distribution Statement A. Approved for public release: distribution is unlimited. Approved AFRL-2026-1671 28-09-2026.

"""Video of ALBCD designing the Cessna 182-like wing of cessna.py, for presentations.

The aerodynamics and the structure are solved as two separate subproblems, as if on
two different computers, coupled only through copies of the aero loads and the wing
weight. The video shows the wing on the airframe, the state of ALBCD's outer and inner
loops, its feasibility and optimality, and the spanwise lift (as computed by the
aerodynamics and as copied by the structure), twist, spar wall thickness and stress.

The subproblem classes below are copied from cessna.py (unchanged), so this script does
not run cessna.py's own solve and plots. Needs pyvista and ffmpeg (on the PATH, or pass
--ffmpeg):

    python examples/jax/cessna/cessna_video.py [-o out.mp4] [--ffmpeg path/to/ffmpeg]
"""

import os
import argparse
import shutil
from collections import namedtuple
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # modopt and the models work in float64
import jax.numpy as jnp
import modopt as mo
from albcd import ALBCD, Subproblem
import warnings
warnings.filterwarnings("ignore")

from models import (aero_model, structures_model, twist_cp0, thickness_cp0, num_cp_twist, num_cp_thickness,
                    num_nodes, bspline_twist, bspline_thickness, min_gauge, sigma_yield_mpa, y, q, v_inf, rho_atm,
                    mesh0_jnp, vlm_geom, solve_aero)

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------------------
# copied from cessna.py
# ---------------------------------------------------------------------------------------

# global x is a single flat vector, ordered so that each subproblem's block is contiguous:
# [twist_cp, weight_copy, thickness_cp, aero_loads_copy], where aero_loads_copy is
# [drag_0..drag_{num_nodes-1}, lift_0..lift_{num_nodes-1}]
TWIST_SLICE     = slice(0, num_cp_twist)
WEIGHT_SLICE    = slice(num_cp_twist, num_cp_twist + 1)
THICKNESS_SLICE = slice(num_cp_twist + 1, num_cp_twist + 1 + num_cp_thickness)
LOADS_SLICE     = slice(num_cp_twist + 1 + num_cp_thickness,
                        num_cp_twist + 1 + num_cp_thickness + 2 * num_nodes)

AERO_INDEX   = slice(TWIST_SLICE.start, WEIGHT_SLICE.stop)     # owns [twist_cp, weight_copy]
STRUCT_INDEX = slice(THICKNESS_SLICE.start, LOADS_SLICE.stop)  # owns [thickness_cp, aero_loads_copy]

# coupling constraints: drag and lift residual at each node, and the weight residual
N_CON = 2 * num_nodes + 1
f_scale = 1e-3
w_scale = 1e-4


def coupling(aero_loads, aero_loads_copy, weight, weight_copy):
    return jnp.concatenate([f_scale * (aero_loads - aero_loads_copy),
                            w_scale * jnp.reshape(weight - weight_copy, 1)])


# a subproblem's objective, its gradient, its local constraints and their Jacobian, each jitted
Compiled = namedtuple("Compiled", "obj grad con jac")


def compile_subproblem(objective, local_constraints):
    """Jit objective(v, *args), its gradient in v, local_constraints(v) and their Jacobian.

    Each subproblem does this once, in setup(). The inputs that change between solves
    (other, y, mu, ...) are arguments of the compiled functions rather than constants in a
    closure, so every solve reuses the same compiled code. Wrapping a new closure in
    mo.JaxProblem for each solve would instead retrace and recompile the VLM and beam
    models every time, at about a second per function.
    """
    return Compiled(jax.jit(objective), jax.jit(jax.grad(objective)),
                    jax.jit(local_constraints), jax.jit(jax.jacrev(local_constraints)))


def solve_slsqp(fns, args, v0, x_scaler, c_scaler, ftol, cl=0.0, cu=0.0):
    """Minimize fns.obj(v, *args) s.t. cl <= fns.con(v) <= cu with modopt's SLSQP.

    Returns the solution, SLSQP's constraint multipliers and the constraint Jacobian at the solution.
    """
    # modopt passes numpy arrays; convert them so the compiled functions always see JAX
    # arrays, as they do in residual(). jax.jit compiles numpy and JAX arguments separately.
    prob = mo.ProblemLite(x0=v0,
                          obj=lambda v: np.float64(fns.obj(jnp.asarray(v), *args)),
                          grad=lambda v: np.array(fns.grad(jnp.asarray(v), *args)),
                          con=lambda v: np.array(fns.con(jnp.asarray(v))),
                          jac=lambda v: np.array(fns.jac(jnp.asarray(v))),
                          x_scaler=x_scaler, cl=cl, cu=cu, c_scaler=c_scaler)
    optimizer = mo.SLSQP(prob, solver_options={'maxiter': 1000, 'ftol': ftol}, turn_off_outputs=True)
    optimizer.solve()
    v = optimizer.results['x'] / x_scaler
    # SLSQP solves in c_scaler-scaled constraint space, so its multipliers are
    # rescaled to the unscaled constraints used in the residual() methods below
    multipliers = np.asarray(optimizer.results['multipliers']) * c_scaler
    jac = np.atleast_2d(np.array(fns.jac(jnp.asarray(v))))
    return v, multipliers, jac


class AeroSubproblem(Subproblem):
    """Owns [twist_cp, weight_copy]: minimizes drag subject to lift = weight_copy."""

    X_SCALER = np.concatenate([np.full(num_cp_twist, 10.0), [1e-3]])  # twist_cp, weight_copy
    C_SCALER = 1e-2  # lift = weight_copy
    FTOL = 1e-8      # SLSQP tolerance

    def setup(self) -> None:
        self._fns = compile_subproblem(self.objective, self.local_constraints)

    def objective(self, v, other, y, mu, weight):
        twist_cp, weight_copy = v[:num_cp_twist], v[num_cp_twist]
        aero_loads_copy = other[num_cp_thickness:]

        CD, aero_loads, _ = aero_model(twist_cp)
        c = coupling(aero_loads, aero_loads_copy, weight, weight_copy)

        return 1e2 * CD + jnp.sum(y * c) + 0.5 * jnp.sum(mu * c**2)

    def local_constraints(self, v):
        _, _, lift = aero_model(v[:num_cp_twist])
        return jnp.reshape(lift - v[num_cp_twist], 1)

    def solve(self, x, y, mu, data, outputs) -> None:
        other, y, mu, weight = (jnp.asarray(a) for a in (self.other(x), y, mu, data["weight"]))

        v_new, self._multipliers, self._jac_con = solve_slsqp(
            self._fns, (other, y, mu, weight),
            np.array(self.decompose(x)), self.X_SCALER, self.C_SCALER, self.FTOL)

        x_new = self.recompose(x, v_new)
        CD, aero_loads, _ = aero_model(v_new[:num_cp_twist])

        outputs["x"] = x_new
        outputs["aero_loads"] = np.array(aero_loads)
        outputs["CD"] = float(CD)
        # StructSubproblem's variables, and so its weight, are unchanged by this solve
        outputs["phi"] = np.array(coupling(aero_loads, x_new[LOADS_SLICE], weight, x_new[WEIGHT_SLICE][0]))

    def residual(self, x, y, mu, data) -> float:
        v, other, y, mu, weight = (jnp.asarray(a) for a in (self.decompose(x), self.other(x),
                                                               y, mu, data["weight"]))
        grad_f = np.array(self._fns.grad(v, other, y, mu, weight))

        # KKT stationarity with SLSQP's multipliers for the local constraint
        resid = grad_f - self._jac_con.T @ self._multipliers
        return float(np.max(np.abs(resid)))


class StructSubproblem(Subproblem):
    """Owns [thickness_cp, aero_loads_copy]: sizes the spar for stress and minimum gauge."""

    X_SCALER = np.concatenate([np.full(num_cp_thickness, 1e2), np.full(2 * num_nodes, 1e-2)])  # thickness_cp, loads
    C_SCALER = np.array([1e-2, 1e-1])  # KS stress, KS min thickness
    FTOL = 1e-8                        # SLSQP tolerance

    def setup(self) -> None:
        self._fns = compile_subproblem(self.objective, self.local_constraints)

    def objective(self, v, other, y, mu, aero_loads):
        thickness_cp, aero_loads_copy = v[:num_cp_thickness], v[num_cp_thickness:]
        weight_copy = other[num_cp_twist]

        _, _, weight = structures_model(aero_loads_copy, thickness_cp)
        c = coupling(aero_loads, aero_loads_copy, weight, weight_copy)

        # the drag term of the augmented Lagrangian is constant in this block, so it is left out
        return jnp.sum(y * c) + 0.5 * jnp.sum(mu * c**2)

    def local_constraints(self, v):
        max_sigma_mpa, min_thickness_mm, _ = structures_model(v[num_cp_thickness:], v[:num_cp_thickness])
        return jnp.stack([max_sigma_mpa - sigma_yield_mpa,
                          min_gauge * 1e3 - min_thickness_mm])

    def solve(self, x, y, mu, data, outputs) -> None:
        other, y, mu, aero_loads = (jnp.asarray(a) for a in (self.other(x), y, mu, data["aero_loads"]))

        v_new, multipliers, self._jac_con = solve_slsqp(
            self._fns, (other, y, mu, aero_loads),
            np.array(self.decompose(x)), self.X_SCALER, self.C_SCALER, self.FTOL,
            cl=np.full(2, -np.inf), cu=np.zeros(2))
        # modopt hands SLSQP each upper-bounded inequality as cu - c(v) >= 0, with
        # Jacobian -J, so its multipliers enter the stationarity condition with the
        # opposite sign to the aero block's equality constraint
        self._multipliers = -multipliers

        x_new = self.recompose(x, v_new)
        aero_loads_copy = v_new[num_cp_thickness:]
        _, _, weight = structures_model(aero_loads_copy, v_new[:num_cp_thickness])

        outputs["x"] = x_new
        outputs["weight"] = float(weight)
        # AeroSubproblem's variables, and so its aero loads, are unchanged by this solve
        outputs["phi"] = np.array(coupling(aero_loads, aero_loads_copy, weight, x_new[WEIGHT_SLICE][0]))

    def residual(self, x, y, mu, data) -> float:
        v, other, y, mu, aero_loads = (jnp.asarray(a) for a in (self.decompose(x), self.other(x),
                                                                   y, mu, data["aero_loads"]))
        grad_f = np.array(self._fns.grad(v, other, y, mu, aero_loads))

        # KKT stationarity with SLSQP's multipliers for the stress and thickness constraints
        resid = grad_f - self._jac_con.T @ self._multipliers
        return float(np.max(np.abs(resid)))


# ---------------------------------------------------------------------------------------
# end of the code copied from cessna.py
# ---------------------------------------------------------------------------------------


def run_albcd():
    """Solves the problem as in cessna.py, logging the block, multipliers and penalties of every solve."""
    log = []

    class LoggedAero(AeroSubproblem):
        def solve(self, x, y, mu, data, outputs):
            log.append((0, np.array(y), np.array(mu)))
            super().solve(x, y, mu, data, outputs)

    class LoggedStruct(StructSubproblem):
        def solve(self, x, y, mu, data, outputs):
            log.append((1, np.array(y), np.array(mu)))
            super().solve(x, y, mu, data, outputs)

    CD_init, aero_loads_init, _ = aero_model(twist_cp0)
    _, _, weight_init = structures_model(aero_loads_init, thickness_cp0)
    x0 = np.concatenate([twist_cp0, [float(weight_init)], thickness_cp0, np.array(aero_loads_init)])

    opt = ALBCD(subproblems=[LoggedAero(AERO_INDEX), LoggedStruct(STRUCT_INDEX)],
                x0=x0,
                mu0=np.full(N_CON, 10.0),
                data0={"weight": float(weight_init), "aero_loads": np.array(aero_loads_init), "CD": float(CD_init)},
                max_mu=1e6, rho=1.2, tau=0.5, feas_tol=3e-4, opt_tol=[1e-1, 1e-3],
                max_outer_iter=40, max_inner_iter=12, verbose=False)
    opt.solve()
    return opt, log


# ---------------------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------------------

# two tones, neon green and cyan, on neutral grays
BG, FG, DIM, GRID = '#05080D', '#D3D9DF', '#6E7781', '#1A2027'  # background, text, ticks and references, grid
NEON, CYAN = '#39FF14', '#00E5FF'
AERO, STRUCT = CYAN, NEON  # aerodynamics, structures
dy = np.gradient(y)  # spanwise width of each station, to turn nodal loads into loads per unit span


def evaluate(x):
    """What the video shows of the design x: the spanwise distributions and the deflected wing."""
    from models import struct_mesh, r, E, G, rho_mat, load_factor, safety_factor, fixed_nodes
    from beam_jax import Beam, CSTube
    from vlm_jax.analysis import panel_pressure

    twist_cp, thickness_cp, loads_copy = x[TWIST_SLICE], x[THICKNESS_SLICE], x[LOADS_SLICE]
    _, aero_loads, _ = aero_model(twist_cp)

    # the wing as the aerodynamics sees it, colored by pressure coefficient
    def_mesh = vlm_geom.apply_twist_rotation(mesh0_jnp, jnp.rad2deg(bspline_twist @ twist_cp),
                                             ref_axis_pos=0.25, symmetry=False, rotate_x=True)
    out = solve_aero(def_mesh, v_inf, rho_atm, symmetry=False, with_viscous=False)
    cp = np.asarray(panel_pressure(out["mesh"], out["sec_forces"], out["normals"], orient="up")).ravel() / q

    # the spar under the (copied) sizing loads, as the structure sees it: stress and deflection
    F = np.zeros((num_nodes, 6))
    F[:, 0] = loads_copy[:num_nodes] * load_factor * safety_factor
    F[:, 2] = loads_copy[num_nodes:] * load_factor * safety_factor
    cs = CSTube(radius=r, thickness=bspline_thickness @ thickness_cp)
    beam = Beam(mesh=struct_mesh, E=E, G=G, rho=rho_mat, cs=cs, F=F, fixed_nodes=fixed_nodes)
    u = beam.solve()
    points = np.array(def_mesh)
    points[:, :, 2] += np.asarray(u)[None, :, 2]

    return dict(twist=np.degrees(np.asarray(bspline_twist @ twist_cp)),
                thickness=1e3 * np.asarray(bspline_thickness @ thickness_cp),
                stress=np.asarray(beam.recover_stress(u)) / 1e6,
                lift=np.asarray(aero_loads[num_nodes:]) / dy, lift_copy=loads_copy[num_nodes:] / dy,
                points=points.reshape(-1, 3), cp=cp)


def loop_states(opt, log):
    """The readout for each block solve: outer iteration, sweep within it, tolerance phase, block and max penalty."""
    starts = [0] + [i for i in range(1, len(log)) if not np.array_equal(log[i][1], log[i - 1][1])]
    # an outer iteration that ends feasible during phase 1 starts phase 2 (see ALBCD.solve)
    phases, phase = [], 1
    for k, start in enumerate(starts):
        phases.append(phase)
        end = starts[k + 1] if k + 1 < len(starts) else len(log)
        if phase < len(opt.opt_tol) and opt.feas_history[(end - 1) // 2] <= opt.feas_tol:
            phase += 1
    readouts = []
    for i, (block, _, mu) in enumerate(log):
        k = sum(s <= i for s in starts) - 1
        tol = f"{opt.opt_tol[phases[k] - 1]:.0e}".replace('e-0', 'e-')
        readouts.append(f"outer {k + 1:>2}  ·  sweep {(i - starts[k]) // 2 + 1:>2}  ·  phase {phases[k]} (opt tol {tol})"
                        f"  ·  max μ {mu.max():.1e}  ·  {'aerodynamics' if block == 0 else 'structures'} solve")
    return readouts


def make_video(opt, log, filename, fps=25):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.colors import LinearSegmentedColormap
    import pyvista as pv
    from vlm_jax import viz as vlm_viz

    states = [evaluate(x) for x in opt.x_history]  # before the first solve and after each solve
    readouts = loop_states(opt, log)
    blend = lambda a, b, t: {k: (1 - t) * a[k] + t * b[k] for k in a}
    bounds = [i for i in range(1, len(log)) if not np.array_equal(log[i][1], log[i - 1][1])]  # solves after a dual update

    # frames: (design, loop readout, sweeps completed); each dual update gets a short pause
    frames = [(states[0], readouts[0], 0)] * fps
    for i in range(len(log)):
        if i in bounds:
            frames += [(states[i], readouts[i], i // 2)] * 8
        frames += [(blend(states[i], states[i + 1], t), readouts[i], (i + 1) // 2) for t in (0.25, 0.5, 0.75, 1.0)]
    frames += [(states[-1], readouts[-1], len(log) // 2)] * (3 * fps)

    plt.rcParams.update({'font.size': 11, 'text.color': FG, 'axes.labelcolor': FG, 'axes.edgecolor': GRID,
                         'xtick.color': DIM, 'ytick.color': DIM, 'axes.facecolor': BG, 'figure.facecolor': BG})
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)

    fig.text(0.025, 0.95, 'Cessna 182 aerostructural optimization', fontsize=18, color=NEON, weight='bold', va='center')
    fig.text(0.025, 0.905, 'minimize drag  ·  lift = weight  ·  spar stress ≤ yield', color=DIM, fontsize=11,
             va='center')
    readout = fig.text(0.025, 0.862, '', color=CYAN, fontsize=10, family='monospace', va='center')

    # 3D view, rendered by PyVista into an image
    ax3d = fig.add_axes([0.0, 0.31, 0.6, 0.52])
    ax3d.set_axis_off()
    w, h = (np.array(ax3d.get_position().size) * fig.get_size_inches() * fig.dpi).astype(int)
    plotter = pv.Plotter(off_screen=True, window_size=(int(w), int(h)))
    plotter.set_background(BG)
    plotter.enable_anti_aliasing('ssaa')
    cmap = LinearSegmentedColormap.from_list('two_tone', [CYAN, NEON])
    wing = vlm_viz.to_pyvista_mesh(states[0]['points'].reshape(mesh0_jnp.shape), cell_data={'Cp': states[0]['cp']})
    clim = np.percentile(np.concatenate([s['cp'] for s in states]), [2, 98])
    plotter.add_mesh(wing, scalars='Cp', cmap=cmap, clim=clim, show_scalar_bar=False)
    cessna = pv.read(os.path.join(HERE, 'cessna182_no_wing.stl')).scale(0.3048, inplace=False)
    plotter.add_mesh(cessna.translate((-1.4, 0.0, -0.9), inplace=False), color='#5B6670', specular=0.6,
                     specular_power=20)
    plotter.camera_position = 'iso'
    center, b = np.array(plotter.center), plotter.bounds
    distance = 1.05 * np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]])

    def render(points, cp, azimuth):
        wing.points[:] = points
        wing.cell_data['Cp'][:] = cp
        az, el = np.radians(azimuth), np.radians(20)
        eye = center + distance * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        plotter.camera_position = [tuple(eye), tuple(center), (0.0, 0.0, 1.0)]
        plotter.render()
        return plotter.screenshot(return_img=True)

    image = ax3d.imshow(render(states[0]['points'], states[0]['cp'], 140))
    cax = fig.add_axes([0.04, 0.34, 0.16, 0.012])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(*clim), cmap), cax=cax, orientation='horizontal')
    cax.set_title('pressure coefficient', fontsize=9, color=DIM)
    cax.tick_params(labelsize=8, length=2)
    cbar.outline.set_edgecolor(GRID)

    def style(a):
        a.grid(axis='y', color=GRID)
        for s in a.spines.values():
            s.set_visible(False)

    # ALBCD's convergence measures after each BCD sweep, with the tolerances it stops on
    sweeps = np.arange(1, len(opt.feas_history) + 1)
    history = []
    for left, data, tol, label in ((0.06, opt.feas_history, opt.feas_tol, 'Feasibility'),
                                   (0.355, opt.opt_history, opt.opt_tol[-1], 'Optimality')):
        a = fig.add_axes([left, 0.08, 0.235, 0.17])
        a.set(yscale='log', xlim=(0.5, len(sweeps) + 0.5), xlabel='Iteration', ylabel=label,
              ylim=(10**np.floor(np.log10(min(np.min(data), tol))) / 2, 10**np.ceil(np.log10(np.max(data))) * 2))
        style(a)
        a.axhline(tol, color=DIM, ls=':', lw=1)
        history.append((a.plot([], [], color=NEON, lw=2)[0], np.asarray(data)))

    # spanwise distributions: the coupled lift, each discipline's own design variables, and the spar stress
    mono = np.load(os.path.join(HERE, 'monolithic_solution.npz'))
    y_elem = 0.5 * (y[:-1] + y[1:])
    lines = {}
    for key, label, bottom in (('lift', 'Lift (N/m)', 0.71), ('twist', 'Twist (deg)', 0.505),
                               ('thickness', 'Spar wall (mm)', 0.30), ('stress', 'Stress (MPa)', 0.095)):
        a = fig.add_axes([0.67, bottom, 0.31, 0.165])
        a.set_ylabel(label)
        a.set_xlim(y[0], y[-1])
        a.tick_params(labelbottom=key == 'stress')
        style(a)
        if key == 'lift':
            lines['lift'], = a.plot(y, states[0]['lift'], color=AERO, lw=2, label='computed by aerodynamics')
            lines['lift_copy'], = a.plot(y, states[0]['lift_copy'], color=STRUCT, lw=2, ls='--',
                                         label='copy used by structures')
            a.set_ylim(0, 1.15 * max(max(s['lift'].max(), s['lift_copy'].max()) for s in states))
            a.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9, ncol=2, labelcolor=FG)
        elif key == 'twist':
            a.plot(y, np.degrees(np.asarray(bspline_twist @ mono['twist_cp'])), color=DIM, lw=0.8,
                   label='monolithic optimum')
            lines['twist'], = a.plot(y, states[0]['twist'], color=AERO, lw=2)
            vals = np.concatenate([s['twist'] for s in states])
            a.set_ylim(vals.min() - 0.5, vals.max() + 0.5)
            a.legend(loc='upper right', frameon=False, fontsize=8, labelcolor=DIM)
        elif key == 'thickness':
            a.plot(y_elem, 1e3 * np.asarray(bspline_thickness @ mono['thickness_cp']), color=DIM, lw=0.8,
                   label='monolithic optimum')
            a.axhline(1e3 * min_gauge, color=DIM, ls=':', lw=1)
            lines['thickness'], = a.plot(y_elem, states[0]['thickness'], color=STRUCT, lw=2)
            a.set_ylim(0, 1.15 * max(s['thickness'].max() for s in states))
            a.legend(loc='upper right', frameon=False, fontsize=8, labelcolor=DIM)
        else:
            a.axhline(sigma_yield_mpa, color=DIM, ls=':', lw=1)
            a.text(y[-1], sigma_yield_mpa, 'yield', ha='right', va='bottom', color=DIM, fontsize=8)
            lines['stress'], = a.plot(y_elem, states[0]['stress'], color=STRUCT, lw=2)
            a.set_ylim(0, 1.2 * max(sigma_yield_mpa, max(s['stress'].max() for s in states)))
            a.set_xlabel('Spanwise location (m)')

    def update(k, frame):
        state, text, done = frame
        image.set_data(render(state['points'], state['cp'], 140 + 20 * k / len(frames)))
        for key, line in lines.items():
            line.set_ydata(state[key])
        for line, data in history:
            line.set_data(sweeps[:done], data[:done])
        readout.set_text(text)

    writer = FFMpegWriter(fps=fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-crf', '28', '-preset', 'veryslow',
                                                                  '-tune', 'animation', '-movflags', '+faststart'])
    with writer.saving(fig, filename, dpi=100):
        for k, frame in enumerate(frames):
            update(k, frame)
            writer.grab_frame(facecolor=BG)
    plotter.close()
    print(f'wrote {filename} ({len(frames)} frames, {len(frames) / fps:.1f} s)')


if __name__ == "__main__":
    import matplotlib as mpl
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('-o', '--output', default=os.path.join(HERE, 'cessna_video.mp4'))
    parser.add_argument('--ffmpeg', default=shutil.which('ffmpeg'), help='path to the ffmpeg executable')
    args = parser.parse_args()
    mpl.rcParams['animation.ffmpeg_path'] = args.ffmpeg or 'ffmpeg'

    opt, log = run_albcd()
    print(f'success: {opt.success}, CD = {opt.data["CD"]:.6f}')
    make_video(opt, log, args.output)
