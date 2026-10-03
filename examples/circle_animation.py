# Distribution Statement A. Approved for public release: distribution is unlimited. Approved AFRL-2026-1671 28-09-2026.

"""Animation of ALBCD solving the quick start problem:

    min x1^2 + x2^2 - 1.5 x1 x2  s.t.  x1^2 + x2^2 >= 0.25

The coupling constraint is written as phi = 0.25 - x1^2 - x2^2 + s = 0 with a slack
s >= 0. Block 1 owns (x1, s) and block 2 owns x2, so in the (x1, x2) plane block 1
moves the iterate horizontally and block 2 moves it vertically.

The left panel shows the iterates on contours of the augmented Lagrangian

    L(x1, x2) = min_{s >= 0} f + y phi + mu/2 phi^2,

which is what the blocks jointly minimize in the BCD inner loop. Between outer
iterations the multiplier y and penalty mu are updated, and the landscape morphs
until its minimum lies on the constraint boundary. The right panels show the
feasibility (max |phi|) and optimality (max block KKT residual) after each sweep.

The problem is the same as quadratic_global_circle.py, but the gradients are
written out by hand, so this needs only NumPy, modopt and matplotlib, plus ffmpeg
to write the video (pass --ffmpeg if it is not on the PATH; the binary from the
imageio-ffmpeg package is found automatically if that is installed).

    python examples/circle_animation.py                  # writes examples/circle_animation.mp4
    python examples/circle_animation.py -o out.mp4 --fps 24
"""

import argparse
import shutil
from pathlib import Path
import numpy as np
import modopt as mo
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from albcd import ALBCD, Subproblem
import warnings
warnings.filterwarnings("ignore")


R2 = 0.5**2 # squared radius of the constraint circle


def f(x1, x2):
    return x1**2 + x2**2 - 1.5 * x1 * x2


def phi(x1, s, x2):
    return R2 - (x1**2 + x2**2) + s


def augmented_lagrangian(x, y, mu):
    x1, s, x2 = x
    c = phi(x1, s, x2)
    return f(x1, x2) + y * c + 0.5 * mu * c**2


def augmented_lagrangian_grad(x, y, mu):
    """Gradient of the augmented Lagrangian with respect to x = (x1, s, x2)."""
    x1, s, x2 = x
    lam = y + mu * phi(x1, s, x2) # derivative of y phi + mu/2 phi^2 with respect to phi
    return np.array([2 * x1 - 1.5 * x2 - 2 * x1 * lam,
                     lam,
                     2 * x2 - 1.5 * x1 - 2 * x2 * lam])


def reduced_augmented_lagrangian(X1, X2, y, mu):
    """Augmented Lagrangian on a grid of (x1, x2), minimized over the slack s >= 0.

    The minimizing s makes phi = max(phi(s=0), -y/mu), so this is explicit.
    """
    c = np.maximum(R2 - (X1**2 + X2**2), -y / mu)
    return f(X1, X2) + y * c + 0.5 * mu * c**2


class BlockSubproblem(Subproblem):
    """Minimizes the augmented Lagrangian over this block's variables with SLSQP.

    Both blocks share this class; they differ only in index and bounds. Every solve
    also records the multiplier and penalty it was given, for the animation.
    """

    def __init__(self, index, xl, xu, log):
        super().__init__(index)
        self.xl = np.asarray(xl, dtype=float)
        self.xu = np.asarray(xu, dtype=float)
        self.log = log # shared list of (block, y, mu), one entry per solve, in x_history order

    def solve(self, x, y, mu, data, outputs) -> None:
        self.log.append((self, float(y[0]), float(mu[0])))

        obj = lambda v: float(augmented_lagrangian(self.recompose(x, v), y[0], mu[0]))
        grad = lambda v: augmented_lagrangian_grad(self.recompose(x, v), y[0], mu[0])[self.index]

        prob = mo.ProblemLite(x0=self.decompose(x), obj=obj, grad=grad,
                              xl=self.xl, xu=self.xu, name=f"block_{self.index}")
        optimizer = mo.SLSQP(prob, solver_options={'maxiter': 100, 'ftol': 1e-8}, turn_off_outputs=True)
        optimizer.solve()

        outputs["x"] = self.recompose(x, optimizer.results['x'])
        outputs["phi"] = np.array([phi(*outputs["x"])])

    def residual(self, x, y, mu, data) -> float:
        """Projected-gradient stationarity residual: zero iff v is a KKT point for xl <= v <= xu."""
        v = self.decompose(x)
        grad = augmented_lagrangian_grad(x, y[0], mu[0])[self.index]
        return float(np.max(np.abs(v - np.clip(v - grad, self.xl, self.xu))))


def run_albcd():
    log = []
    block1 = BlockSubproblem(slice(0, 2), xl=[-np.inf, 0.0], xu=[np.inf, np.inf], log=log) # owns x1 and s
    block2 = BlockSubproblem(slice(2, 3), xl=[-np.inf], xu=[np.inf], log=log) # owns x2

    opt = ALBCD(subproblems=[block1, block2],
                x0=np.array([-0.5, 0.0, 1.0]),
                mu0=np.array([1.0]),
                max_mu=1e3,
                rho=1.2,
                tau=0.5,
                feas_tol=1e-3,
                opt_tol=[1e-2, 1e-4],
                max_y=1e6,
                max_outer_iter=100,
                max_inner_iter=100,
                verbose=False)
    opt.solve()

    return opt, [(0 if b is block1 else 1, y, mu) for b, y, mu in log]


BLOCK_COLORS = ['#F39C12', '#E7298A'] # block 1 (x1, s): orange, block 2 (x2): magenta
BLOCK_LABELS = [r'block 1 solve: $(x_1, s)$', r'block 2 solve: $x_2$']
X_OPT = np.sqrt(2) / 4 # the solutions are x1 = x2 = +-sqrt(2)/4
LIMITS = (-1.0, 1.3) # axis limits of the main panel, in both x1 and x2
ZOOM = (0.30, 0.39)  # axis limits of the inset around the solution
# contour levels of the augmented Lagrangian above its minimum: dense in the valley, where
# the multiplier updates move the minimum, and sparse outside it
LEVELS = np.concatenate([np.linspace(0, 0.12, 13), np.geomspace(0.16, 3.0, 9)])


def build_frames(log, x_hist, fps):
    """Lists the animation frames as (completed solves, fraction of the next move, y, mu, banner).

    Each block solve gets a short move, scaled by its step length, and each outer
    iteration starts with a morph of the landscape from the old (y, mu) to the new.
    """
    frames = []
    y_prev, mu_prev = log[0][1], log[0][2]
    outer = 1

    frames += [(0, 0.0, y_prev, mu_prev, None)] * fps # hold on the starting point

    for i, (block, y, mu) in enumerate(log):

        if (y, mu) != (y_prev, mu_prev): # multiplier/penalty update: a new outer iteration
            outer += 1
            banner = rf'outer iteration {outer}:  $y \leftarrow y + \mu\phi = {y:.4f}$'
            if mu != mu_prev:
                banner += rf',  $\mu \leftarrow \rho\mu = {mu:.2f}$'
            n = int(1.2 * fps)
            for t in np.linspace(0, 1, n):
                t = 0.5 - 0.5 * np.cos(np.pi * t) # ease in and out
                frames.append((i, 0.0, y_prev + t * (y - y_prev), mu_prev * (mu / mu_prev)**t, banner))
            frames += [(i, 0.0, y, mu, banner)] * (fps // 2)
            y_prev, mu_prev = y, mu

        n = int(np.clip(30 * np.linalg.norm(x_hist[i + 1] - x_hist[i]), 3, 12)) # frames for this move
        for t in np.linspace(0, 1, n + 1)[1:]:
            frames.append((i, 0.5 - 0.5 * np.cos(np.pi * t), y, mu, None))

    frames += [(len(log), 0.0, y_prev, mu_prev, 'converged')] * (3 * fps) # hold on the solution

    return frames


def make_animation(opt, log, filename, fps, dpi):

    x_hist = np.array(opt.x_history)[:, [0, 2]] # (x1, x2) before the first solve and after each solve
    blocks = np.array([b for b, _, _ in log])
    feas = np.array(opt.feas_history)
    res = np.array(opt.opt_history)
    sweeps = np.arange(1, len(feas) + 1)
    outer_starts = [i // 2 + 1 for i in range(1, len(log)) if log[i][1:] != log[i - 1][1:]] # first sweep of each new outer iteration

    frames = build_frames(log, x_hist, fps)

    plt.rcParams.update({'font.size': 13, 'axes.titlesize': 14, 'mathtext.fontset': 'cm'})
    fig = plt.figure(figsize=(12.8, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], left=0.06, right=0.97, bottom=0.17, top=0.88,
                          wspace=0.22, hspace=0.12)
    ax = fig.add_subplot(gs[:, 0])
    ax_feas = fig.add_subplot(gs[0, 1])
    ax_opt = fig.add_subplot(gs[1, 1], sharex=ax_feas)
    fig.suptitle('ALBCD on  min $x_1^2 + x_2^2 - 1.5 x_1 x_2$  s.t.  $x_1^2 + x_2^2 \\geq 0.25$', fontsize=16)

    # landscape: contours of the augmented Lagrangian, minimized over s, relative to its minimum
    g = np.linspace(*LIMITS, 301)
    X1, X2 = np.meshgrid(g, g)
    gz = np.linspace(*ZOOM, 121)
    Z1, Z2 = np.meshgrid(gz, gz)
    cmap = plt.get_cmap('YlGnBu_r')
    norm = BoundaryNorm(LEVELS, cmap.N, extend='max') # one distinct color per band

    theta = np.linspace(0, 2 * np.pi, 400)
    circle = 0.5 * np.array([np.cos(theta), np.sin(theta)])

    inset = ax.inset_axes([0.6, 0.03, 0.37, 0.37])
    for a in (ax, inset):
        a.fill(*circle, facecolor='none', edgecolor='0.35', hatch='///', linewidth=0, zorder=2)
        a.plot(*circle, '--', color='k', linewidth=1.8, zorder=3)
        a.plot([X_OPT, -X_OPT], [X_OPT, -X_OPT], '*', color='gold', markersize=18 if a is ax else 22,
               mec='k', zorder=4)
        a.set_aspect('equal')
    ax.set(xlim=LIMITS, ylim=LIMITS, xlabel='$x_1$', ylabel='$x_2$', xticks=[-1, 0, 1], yticks=[-1, 0, 1])
    inset.set(xlim=ZOOM, ylim=ZOOM, xticks=[], yticks=[])
    ax.indicate_inset_zoom(inset, edgecolor='k', alpha=0.8)
    minimizers, = ax.plot([], [], 'X', color='w', mec='k', mew=1.2, markersize=13, zorder=5)
    ax.text(0, -0.12, 'infeasible', ha='center', va='center', fontsize=12, zorder=5,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))

    paths, heads = [], []
    for a, ms in ((ax, 5), (inset, 6)):
        lc = LineCollection([], linewidths=2.5, zorder=6)
        a.add_collection(lc)
        dots, = a.plot([], [], 'o', color='k', markersize=ms * 0.6, zorder=7)
        head, = a.plot([], [], 'o', markersize=ms + 5, mec='k', mew=1.5, zorder=8)
        paths.append((lc, dots))
        heads.append(head)
    ax.plot(*x_hist[0], 's', color='w', mec='k', markersize=9, zorder=7)

    status = ax.text(0.02, 0.02, '', transform=ax.transAxes, va='bottom', fontsize=12, zorder=10,
                     bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.92))
    banner = fig.text(0.5, 0.905, '', ha='center', va='bottom', fontsize=14, color='#B03A2E')

    # convergence history, revealed sweep by sweep
    for a, data, tols, label in ((ax_feas, feas, [opt.feas_tol], r'feasibility  max$|\phi|$'),
                                 (ax_opt, res, opt.opt_tol, 'optimality (KKT residual)')):
        a.set_yscale('log')
        a.set_xlim(0.5, len(feas) + 0.5)
        a.set_ylim(10**np.floor(np.log10(min(data.min(), min(tols)))) / 2, 10**np.ceil(np.log10(data.max())) * 2)
        a.set_ylabel(label)
        a.grid(True, which='major', alpha=0.3)
        for tol in tols:
            a.axhline(tol, color='0.4', linestyle=':', linewidth=1.5)
        for s in outer_starts:
            a.axvline(s - 0.5, color='#B03A2E', alpha=0.35, linewidth=1.2)
    ax_feas.text(0.7, opt.feas_tol, 'feas_tol', ha='left', va='bottom', fontsize=10, color='0.3')
    for k, tol in enumerate(opt.opt_tol, start=1):
        ax_opt.text(0.7, tol, f'opt_tol (phase {k})', ha='left', va='bottom', fontsize=10, color='0.3')
    ax_feas.text(outer_starts[0] - 0.3, ax_feas.get_ylim()[1] / 2, 'multiplier\nupdates', ha='right', va='top',
                 fontsize=10, color='#B03A2E')
    plt.setp(ax_feas.get_xticklabels(), visible=False)
    ax_opt.set_xlabel('BCD sweep (one solve of each block)')
    feas_line, = ax_feas.plot([], [], 'o-', color='#1F618D', markersize=4, linewidth=2)
    opt_line, = ax_opt.plot([], [], 'o-', color='#117A65', markersize=4, linewidth=2)

    handles = [plt.Line2D([], [], color=c, linewidth=3, label=l) for c, l in zip(BLOCK_COLORS, BLOCK_LABELS)]
    handles += [plt.Line2D([], [], color='k', linestyle='--', label='constraint boundary'),
                plt.Line2D([], [], marker='X', color='w', mec='k', markersize=11, linestyle='', label='minima of AL landscape'),
                plt.Line2D([], [], marker='*', color='gold', mec='k', markersize=14, linestyle='', label='optimal solutions'),
                plt.Line2D([], [], marker='s', color='w', mec='k', markersize=8, linestyle='', label='start')]
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, fontsize=11, handletextpad=0.4, columnspacing=1.2)

    contours = {}

    def draw_landscape(y, mu):
        for key in ('fill', 'line', 'zoom'):
            if key in contours:
                contours[key].remove()
        L = reduced_augmented_lagrangian(X1, X2, y, mu)
        L_min = L.min()
        contours['fill'] = ax.contourf(X1, X2, L - L_min, levels=LEVELS, cmap=cmap, norm=norm, alpha=0.8,
                                       extend='max', zorder=0)
        contours['line'] = ax.contour(X1, X2, L - L_min, levels=LEVELS, colors='w', linewidths=0.4, alpha=0.5, zorder=1)
        LZ = reduced_augmented_lagrangian(Z1, Z2, y, mu) - L_min
        contours['zoom'] = inset.contourf(Z1, Z2, LZ, levels=np.linspace(0, 0.01, 11), cmap=cmap, alpha=0.8,
                                          extend='max', zorder=0)
        # the landscape is symmetric under x -> -x, so its minima come in pairs
        x_min = np.array([X1.flat[L.argmin()], X2.flat[L.argmin()]])
        minimizers.set_data([x_min[0], -x_min[0]], [x_min[1], -x_min[1]])

    def update(frame):
        n, t, y, mu, msg = frame
        if (y, mu) != contours.get('state'):
            draw_landscape(y, mu)
            contours['state'] = (y, mu)

        # the path through the completed solves, plus the move in progress
        pts = list(x_hist[:n + 1])
        colors = [BLOCK_COLORS[b] for b in blocks[:n]]
        if n < len(log) and t > 0:
            pts.append(x_hist[n] + t * (x_hist[n + 1] - x_hist[n]))
            colors.append(BLOCK_COLORS[blocks[n]])
        pts = np.array(pts)
        segs = np.stack([pts[:-1], pts[1:]], axis=1) if len(pts) > 1 else np.zeros((0, 2, 2))
        for (lc, dots), head in zip(paths, heads):
            lc.set_segments(segs)
            lc.set_color(colors)
            dots.set_data(*x_hist[1:n + 1].T)
            head.set_data([pts[-1, 0]], [pts[-1, 1]])
            head.set_color(colors[-1] if colors else 'w')
            head.set_markeredgecolor('k')

        # sweeps completed so far: each sweep is one solve of block 1 then block 2
        done = n // 2
        feas_line.set_data(sweeps[:done], feas[:done])
        opt_line.set_data(sweeps[:done], res[:done])

        outer = 1 + sum(s <= done + 1 for s in outer_starts) if n < len(log) else 1 + len(outer_starts)
        if msg == 'converged':
            text = (f'converged: $x_1 = {opt.x[0]:.4f},\\ x_2 = {opt.x[2]:.4f}$\n'
                    f'exact: $x_1 = x_2 = \\sqrt{{2}}/4 = {X_OPT:.4f}$')
        else:
            active = f'block {blocks[n] + 1} solving' if n < len(log) and t > 0 else 'updating y, $\\mu$' if msg else ''
            text = (f'outer iteration {outer},  sweep {min(n // 2 + 1, len(feas))}\n'
                    f'$y = {y:.4f},\\ \\mu = {mu:.2f}$\n{active}')
        status.set_text(text)
        banner.set_text('' if msg in (None, 'converged') else msg)
        return []

    if not mpl.rcParams['animation.ffmpeg_path'] or not shutil.which(mpl.rcParams['animation.ffmpeg_path']):
        raise RuntimeError('ffmpeg not found: install it, or pass its path with --ffmpeg')
    writer = FFMpegWriter(fps=fps, codec='libx264',
                          extra_args=['-pix_fmt', 'yuv420p', '-crf', '23', '-preset', 'veryslow',
                                      '-tune', 'animation', '-movflags', '+faststart'])
    with writer.saving(fig, filename, dpi=dpi):
        for k, frame in enumerate(frames):
            update(frame)
            writer.grab_frame()
            if k % 50 == 0:
                print(f'frame {k}/{len(frames)}')
    plt.close(fig)
    print(f'wrote {filename} ({len(frames)} frames, {len(frames) / fps:.1f} s)')


def find_ffmpeg(path):
    """Returns the ffmpeg executable: the given path, ffmpeg on the PATH, or the imageio-ffmpeg binary."""
    if path:
        return path
    if shutil.which('ffmpeg'):
        return 'ffmpeg'
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('-o', '--output', default=str(Path(__file__).with_suffix('.mp4')),
                        help='video file to write (default: next to this script)')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--dpi', type=int, default=100, help='the figure is 12.8 x 6.4 in')
    parser.add_argument('--ffmpeg', help='path to the ffmpeg executable')
    args = parser.parse_args()

    mpl.rcParams['animation.ffmpeg_path'] = find_ffmpeg(args.ffmpeg) or 'ffmpeg'

    opt, log = run_albcd()
    print(f'success: {opt.success}, x = {opt.x}, {len(log)} block solves')
    make_animation(opt, log, args.output, args.fps, args.dpi)
