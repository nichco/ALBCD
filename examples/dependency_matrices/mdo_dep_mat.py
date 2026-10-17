import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9})
C0, C1, C2, C3 = "#6250d6", "#2a78d6", "#eb6834", "#1baf7a"   # global, subproblems 1-3
C4 = "#d4448f"                                                 # subproblem N
GREY = "0.88"                                                  # intermediate subproblems i = 4, ..., N-1
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


W, H = 2.10, 2.04                             # drawing extent in data units (the saved bbox always spans it)
fig = plt.figure(figsize=(W * IN_PER_UNIT, H * IN_PER_UNIT))
ax = fig.add_axes((0, 0, 1, 1))
ax.set(xlim=(0, W), ylim=(0, H), aspect="equal")
ax.axis("off")

# Subproblems shown explicitly: i -> (column of x_i, row of g_i/h_i, colour); "..." stands for i = 4, ..., N-1
sub = {"1": (1, 2, C1), "2": (2, 3, C2), "3": (3, 4, C3), "N": (5, 6, C4)}
ELL_C, ELL_R = 4, 5                            # ellipsis column / row

cols = [("$x_0$", C0, 2), ("$x_1$", C1, 2), ("$x_2$", C2, 2), ("$x_3$", C3, 2), (r"$\cdots$", "k", 2),
        ("$x_N$", C4, 2)]
rows = [("$f$", "k", 2), ("$g_0,h_0$", C0, 2), ("$g_1,h_1$", C1, 2), ("$g_2,h_2$", C2, 2),
        ("$g_3,h_3$", C3, 2), (r"$\vdots$", "k", 2), ("$g_N,h_N$", C4, 2)]

# f, g_0 and h_0 depend on every variable
fill = {(r, c): cols[c][1] for r in (0, 1) for c in range(len(cols))}
# g_i, h_i depend on the shared variable x_0 and on their own x_i only
for c, r, color in sub.values():
    fill.update({(r, 0): C0, (r, c): color})
# intermediate subproblems: same pattern, shaded light grey
fill.update({(0, ELL_C): GREY, (1, ELL_C): GREY, (ELL_R, 0): GREY, (ELL_R, ELL_C): GREY})
xs, ys = block_matrix(ax, 0.58, 1.77, cols, rows, fill)

plt.savefig("mdo_dep_mat.pdf", bbox_inches="tight")
plt.show()