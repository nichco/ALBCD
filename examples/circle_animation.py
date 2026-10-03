# Distribution Statement A. Approved for public release: distribution is unlimited. Approved AFRL-2026-1671 28-09-2026.

"""Video of ALBCD solving the quick start problem (see quadratic_global_circle.py):

    min x1^2 + x2^2 - 1.5 x1 x2  s.t.  x1^2 + x2^2 >= 0.25

Block 1 owns (x1, s) and block 2 owns x2, so in the (x1, x2) plane block 1 moves the
iterate horizontally and block 2 vertically. The background is the augmented Lagrangian
minimized over the slack s; it morphs each time the multiplier y and penalty mu are
updated, until its minimum lies on the constraint circle. The right plots show the
feasibility and optimality after each BCD sweep.

Needs modopt, matplotlib and ffmpeg (on the PATH, or pass --ffmpeg):

    python examples/circle_animation.py [-o out.mp4] [--ffmpeg path/to/ffmpeg]
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

R2 = 0.5**2  # squared radius of the constraint circle
COLORS = ['tab:orange', 'tab:purple']  # block 1 (x1, s), block 2 (x2)
RED = '#C0392B'


def phi(x1, s, x2):
    return R2 - (x1**2 + x2**2) + s


def augmented_lagrangian_grad(x, y, mu):
    """Gradient of f + y phi + mu/2 phi^2 with respect to x = (x1, s, x2)."""
    x1, s, x2 = x
    lam = y + mu * phi(x1, s, x2)
    return np.array([2 * x1 - 1.5 * x2 - 2 * x1 * lam, lam, 2 * x2 - 1.5 * x1 - 2 * x2 * lam])


def landscape(X1, X2, y, mu):
    """Augmented Lagrangian on a grid of (x1, x2), minimized over s >= 0 (then phi = max(phi(s=0), -y/mu))."""
    c = np.maximum(R2 - (X1**2 + X2**2), -y / mu)
    return X1**2 + X2**2 - 1.5 * X1 * X2 + y * c + 0.5 * mu * c**2


class BlockSubproblem(Subproblem):
    """Minimizes the augmented Lagrangian over this block's variables with SLSQP."""

    def __init__(self, index, xl, xu, log):
        super().__init__(index)
        self.xl, self.xu = np.array(xl), np.array(xu)
        self.log = log  # shared across blocks: the (block index, y, mu) of every solve, in order

    def solve(self, x, y, mu, data, outputs) -> None:
        self.log.append((self.index.start > 0, float(y[0]), float(mu[0])))

        def obj(v):
            xv = self.recompose(x, v)
            c = phi(*xv)
            return float(xv[0]**2 + xv[2]**2 - 1.5 * xv[0] * xv[2] + y[0] * c + 0.5 * mu[0] * c**2)

        grad = lambda v: augmented_lagrangian_grad(self.recompose(x, v), y[0], mu[0])[self.index]
        prob = mo.ProblemLite(x0=self.decompose(x), obj=obj, grad=grad, xl=self.xl, xu=self.xu)
        optimizer = mo.SLSQP(prob, solver_options={'maxiter': 100, 'ftol': 1e-8}, turn_off_outputs=True)
        optimizer.solve()

        outputs["x"] = self.recompose(x, optimizer.results['x'])
        outputs["phi"] = np.array([phi(*outputs["x"])])

    def residual(self, x, y, mu, data) -> float:
        v = self.decompose(x)
        grad = augmented_lagrangian_grad(x, y[0], mu[0])[self.index]
        return float(np.max(np.abs(v - np.clip(v - grad, self.xl, self.xu))))


def run_albcd():
    log = []
    blocks = [BlockSubproblem(slice(0, 2), [-np.inf, 0.0], [np.inf, np.inf], log),  # x1 and s >= 0
              BlockSubproblem(slice(2, 3), [-np.inf], [np.inf], log)]  # x2
    opt = ALBCD(blocks, x0=np.array([-0.5, 0.0, 1.0]), mu0=np.array([1.0]), feas_tol=1e-3,
                opt_tol=[1e-2, 1e-4], max_inner_iter=100, verbose=False)
    opt.solve()
    return opt, log


def make_video(opt, log, filename, ffmpeg):
    fps = 25
    xy = np.array(opt.x_history)[:, [0, 2]]  # (x1, x2) before the first solve and after each solve
    feas, res = np.array(opt.feas_history), np.array(opt.opt_history)
    n_sweeps = len(feas)
    bounds = [i // 2 + 1 for i in range(1, len(log)) if log[i][1:] != log[i - 1][1:]]  # sweeps starting an outer iteration

    # frames: (solves completed, fraction of the next move, y, mu); the landscape morphs at each update
    frames = [(0, 0, *log[0][1:])] * fps
    for i, (_, y, mu) in enumerate(log):
        if i and (y, mu) != log[i - 1][1:]:
            y0, mu0 = log[i - 1][1:]
            frames += [(i, 0, y0 + t * (y - y0), mu0 * (mu / mu0)**t) for t in np.linspace(0, 1, 30)]
        frames += [(i, t, y, mu) for t in np.linspace(0, 1, 5)[1:]]
    frames += [(len(log), 0, *log[-1][1:])] * (3 * fps)

    plt.rcParams.update({'font.size': 13, 'mathtext.fontset': 'cm'})
    fig = plt.figure(figsize=(10.8, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[5.2, 3.3], left=0.07, right=0.98, bottom=0.2, top=0.97,
                          wspace=0.31, hspace=0.12)
    ax, ax_feas = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1])
    ax_opt = fig.add_subplot(gs[1, 1], sharex=ax_feas)

    # left plot: landscape, infeasible region, and the zoomed inset around one of the solutions
    lim, zoom = (-1, 1.3), (0.30, 0.39)
    inset = ax.inset_axes([0.6, 0.03, 0.37, 0.37])
    theta = np.linspace(0, 2 * np.pi, 400)
    for a in (ax, inset):
        a.fill(0.5 * np.cos(theta), 0.5 * np.sin(theta), color='tab:red', alpha=0.3, lw=0, zorder=2)
        a.plot(0.5 * np.cos(theta), 0.5 * np.sin(theta), '--', color='tab:red', lw=1.8, zorder=3)
        a.set_aspect('equal')
    ax.set(xlim=lim, ylim=lim, xlabel='$x_1$', ylabel='$x_2$', xticks=[-1, 0, 1], yticks=[-1, 0, 1])
    inset.set(xlim=zoom, ylim=zoom, xticks=[], yticks=[])
    ax.indicate_inset_zoom(inset, edgecolor='k', alpha=0.8)
    ax.text(0, -0.12, 'infeasible', ha='center', va='center', fontsize=12, zorder=5,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
    minima, = ax.plot([], [], 'X', color='w', mec='k', mew=1.2, ms=13, zorder=5)
    ax.plot(*xy[0], 's', color='w', mec='k', ms=9, zorder=7)

    paths = []  # the path and the current iterate, in the main plot and the inset
    for a in (ax, inset):
        lc = LineCollection([], lw=2.5, zorder=6)
        a.add_collection(lc)
        paths.append((lc, a.plot([], [], 'o', ms=10, mec='k', mew=1.5, zorder=8)[0]))
    counter = ax.text(0.03, 0.97, '', transform=ax.transAxes, va='top', fontsize=15, color=RED, weight='bold', zorder=10)
    status = ax.text(0.02, 0.02, '', transform=ax.transAxes, va='bottom', fontsize=12, zorder=10,
                     bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.92))
    fig.legend([plt.Line2D([], [], color=c, lw=3) for c in COLORS]
               + [plt.Line2D([], [], color='tab:red', ls='--'), plt.Line2D([], [], marker='X', color='w', mec='k', ls=''),
                  plt.Line2D([], [], marker='s', color='w', mec='k', ls='')],
               [r'block 1 solve: $(x_1, s)$', r'block 2 solve: $x_2$', 'constraint boundary',
                'minima of AL landscape', 'start'],
               loc='lower center', ncol=3, frameon=False, fontsize=11)

    # right plots: feasibility and optimality history, revealed one sweep at a time
    lines = []
    for a, data, tol, label, color, tol_label in ((ax_feas, feas, opt.feas_tol, 'Feasibility', '#1F618D', 'feas tol'),
                                                  (ax_opt, res, opt.opt_tol[-1], 'Optimality', '#117A65', 'opt tol')):
        a.set(yscale='log', xlim=(0.5, n_sweeps + 0.5), ylabel=label,
              ylim=(10**np.floor(np.log10(min(data.min(), tol))) / 2, 10**np.ceil(np.log10(data.max())) * 2))
        a.grid(axis='y', alpha=0.3)
        a.axhline(tol, color='0.4', ls=':', lw=1.5)
        a.text(0.7, tol, tol_label, va='bottom', fontsize=10, color='0.3')
        for b in bounds:
            a.axvline(b - 0.5, color=RED, alpha=0.35, lw=1.2)
        lines.append(a.plot([], [], color=color, lw=2)[0])
    ax_feas.text(bounds[0] - 0.7, ax_feas.get_ylim()[1] / 2, 'multiplier\nupdates', ha='right', va='top',
                 fontsize=10, color=RED)
    plt.setp(ax_feas.get_xticklabels(), visible=False)
    ax_opt.set_xlabel('Iteration')

    levels = np.concatenate([np.linspace(0, 0.12, 13), np.geomspace(0.16, 3, 9)])  # dense in the valley
    grid = [np.meshgrid(np.linspace(*lim, 301), np.linspace(*lim, 301)),
            np.meshgrid(np.linspace(*zoom, 121), np.linspace(*zoom, 121))]
    cmap = plt.get_cmap('YlGnBu_r')
    norm = BoundaryNorm(levels, cmap.N, extend='max')  # one color per band, so the valley stays visible
    drawn = {}

    def update(frame):
        n, t, y, mu = frame
        if (y, mu) != drawn.get('state'):  # redraw the landscape only when (y, mu) changed
            for artist in drawn.get('artists', []):
                artist.remove()
            drawn['state'] = (y, mu)
            drawn['artists'] = []
            L_min = landscape(*grid[0], y, mu).min()
            for a, (X1, X2) in zip((ax, inset), grid):
                lv = levels if a is ax else np.linspace(0, 0.01, 11)
                Z = landscape(X1, X2, y, mu) - L_min
                drawn['artists'] += [a.contourf(X1, X2, Z, levels=lv, cmap=cmap, norm=norm if a is ax else None, alpha=0.8, extend='max', zorder=0)]
            i = landscape(*grid[0], y, mu).argmin()
            minima.set_data([grid[0][0].flat[i], -grid[0][0].flat[i]], [grid[0][1].flat[i], -grid[0][1].flat[i]])

        pts = xy[:n + 1] if t == 0 else np.vstack([xy[:n + 1], xy[n] + t * (xy[n + 1] - xy[n])])
        colors = [COLORS[log[i][0]] for i in range(len(pts) - 1)]
        for lc, head in paths:
            lc.set_segments(np.stack([pts[:-1], pts[1:]], axis=1))
            lc.set_color(colors)
            head.set_data([pts[-1, 0]], [pts[-1, 1]])
            head.set_color(colors[-1] if colors else 'w')

        done = min(n // 2, n_sweeps)  # a sweep is one solve of each block
        for line, data in zip(lines, (feas, res)):
            line.set_data(np.arange(1, done + 1), data[:done])
        counter.set_text(f'Dual update {1 + sum(b <= done + 1 for b in bounds)}')
        status.set_text(f'$y = {y:.4f},\\ \\mu = {mu:.2f}$')

    writer = FFMpegWriter(fps=fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-crf', '23', '-preset', 'veryslow',
                                                                  '-tune', 'animation', '-movflags', '+faststart'])
    with writer.saving(fig, filename, dpi=100):
        for frame in frames:
            update(frame)
            writer.grab_frame()
    print(f'wrote {filename} ({len(frames)} frames, {len(frames) / fps:.1f} s)')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('-o', '--output', default=str(Path(__file__).with_suffix('.mp4')))
    parser.add_argument('--ffmpeg', default=shutil.which('ffmpeg'), help='path to the ffmpeg executable')
    args = parser.parse_args()
    mpl.rcParams['animation.ffmpeg_path'] = args.ffmpeg or 'ffmpeg'

    opt, log = run_albcd()
    print(f'success: {opt.success}, x = {opt.x}')
    make_video(opt, log, args.output, args.ffmpeg)

