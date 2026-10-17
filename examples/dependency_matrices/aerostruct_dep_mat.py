import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9})
C0, C1, C2 = "#6250d6", "#2a78d6", "#eb6834"   # coupling, subproblem 1 (theta, W^), subproblem 2 (t, F^)
IN_PER_UNIT = 0.638                            # same cell size (in inches) as dependency_matrices.pdf


def block_matrix(ax, x0, y0, cols, rows, filled, cell=0.125):
    """cols/rows: lists of (label, color, size in cells); filled: {(row, col): color}."""
    xs = [x0]
    for _, _, w in cols:
        xs.append(xs[-1] + w * cell)
    ys = [y0]
    for _, _, h in rows:
        ys.append(ys[-1] - h * cell)
    for (r, c), color in filled.items():
        ax.add_patch(Rectangle((xs[c], ys[r + 1]), xs[c + 1] - xs[c], ys[r] - ys[r + 1],
                               fc=color, ec="none"))
    for x in xs:
        ax.plot([x, x], [ys[0], ys[-1]], color="0.8", lw=0.5)
    for y in ys:
        ax.plot([xs[0], xs[-1]], [y, y], color="0.8", lw=0.5)
    ax.add_patch(Rectangle((xs[0], ys[-1]), xs[-1] - xs[0], ys[0] - ys[-1], fill=False, lw=0.8))
    for c, (lab, color, _) in enumerate(cols):
        ax.text((xs[c] + xs[c + 1]) / 2, ys[0] + 0.05, lab, ha="center", va="bottom", color=color)
    for r, (lab, color, _) in enumerate(rows):
        ax.text(xs[0] - 0.07, (ys[r] + ys[r + 1]) / 2, lab, ha="right", va="center", color=color)
    return xs, ys


# W, H = 3.5, 1.90                              # drawing extent in data units (the saved bbox always spans it)
fig = plt.figure(figsize=(2.5, 1.25))
ax = fig.add_axes((0, 0, 1, 1))
# ax.set(xlim=(0, W), ylim=(0, H), aspect="equal")
ax.set(aspect="equal")
ax.axis("off")
top = 1.35
sigma = r"$\sigma\leq\sigma_\mathrm{y}$"

# (a) monolithic: the weight W(t) and load F(theta) couple the two disciplines
cols = [(r"$\theta$", C1, 2), ("$t$", C2, 2)]
rows = [("$C_D$", "k", 2), ("$L=W$", C1, 2), (sigma, C2, 2)]
fill = {(r, c): cols[c][1] for r, c in [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)]}
xs, ys = block_matrix(ax, 0.62, top, cols, rows, fill)

# ax.text((xs[0] + xs[-1]) / 2, top + 0.30, "(a) Monolithic", ha="center", va="bottom")

# (b) decomposed: block-diagonal in (theta, W^) and (t, F^), coupled by the consistency rows
cols = [(r"$\theta$", C1, 2), (r"$\hat{W}$", C1, 2), ("$t$", C2, 2), (r"$\hat{F}$", C2, 2)]
rows = [("$C_D$", "k", 2), (r"$L=\hat{W}$", C1, 2), (sigma, C2, 2),
        (r"$W-\hat{W}$", C0, 2), (r"$F-\hat{F}$", C0, 2)]
deps = [(0, 0), (1, 0), (1, 1), (2, 2), (2, 3), (3, 1), (3, 2), (4, 0), (4, 3)]
fill = {(r, c): cols[c][1] for r, c in deps}
xs, ys = block_matrix(ax, 2.40, top, cols, rows, fill)
# ax.text((xs[0] + xs[-1]) / 2, top + 0.30, "(b) Decomposed", ha="center", va="bottom")

plt.savefig("aerostruct_dep_mat.pdf", bbox_inches="tight")
plt.show()