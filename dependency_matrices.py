import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9})
C0, C1, C2, C3 = "#6250d6", "#2a78d6", "#eb6834", "#1baf7a"   # global, subproblems 1-3


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
        ax.plot([x, x], [ys[0], ys[-1]], color="0.6", lw=0.5)
    for y in ys:
        ax.plot([xs[0], xs[-1]], [y, y], color="0.6", lw=0.5)
    ax.add_patch(Rectangle((xs[0], ys[-1]), xs[-1] - xs[0], ys[0] - ys[-1], fill=False, lw=0.8))
    for c, (lab, color, _) in enumerate(cols):
        ax.text((xs[c] + xs[c + 1]) / 2, ys[0] + 0.05, lab, ha="center", va="bottom", color=color)
    for r, (lab, color, _) in enumerate(rows):
        ax.text(xs[0] - 0.07, (ys[r] + ys[r + 1]) / 2, lab, ha="right", va="center", color=color)
    return xs, ys


fig, ax = plt.subplots(figsize=(7.0, 3.0))
ax.set(xlim=(0, 7.0), ylim=(0, 3.0), aspect="equal")
ax.axis("off")
sub = [C1, C2, C3]

# (a) decomposed problem: x0 couples every row, g0/h0 couple every column
cols = [("$x_0$", C0, 2)] + [(f"$x_{i}$", sub[i - 1], 3) for i in (1, 2, 3)]
rows = [("$f$", "k", 1), ("$g_0,h_0$", C0, 2)] + [(f"$g_{i},h_{i}$", sub[i - 1], 2) for i in (1, 2, 3)]

fill = {(r, c): cols[c][1] for r in (0, 1) for c in range(4)}
fill.update({(1 + i, 0): C0 for i in (1, 2, 3)})
fill.update({(1 + i, i): sub[i - 1] for i in (1, 2, 3)})
block_matrix(ax, 0.70, 2.25, cols, rows, fill)

# (b) reformulated: u_i = (z_i, x_i, s_i), central variable z, coupling rows phi
cols = []
for i in (1, 2, 3):
    cols += [(f"$z_{i}$", "0.3", 2), (f"$x_{i}$", "0.3", 3), (f"$s_{i}$", "0.3", 1)]
cols += [("$z$", C0, 2)]
rows = [("$f$", "k", 1)] + [(f"$g_{i},h_{i}$", sub[i - 1], 2) for i in (1, 2, 3)] + \
       [(f"$z_{i}-z$", C0, 2) for i in (1, 2, 3)] + [("$h_0$", C0, 1), ("$g_0+s$", C0, 3)]
fill = {}
for i in (1, 2, 3):
    z, x, s = 3 * (i - 1), 3 * (i - 1) + 1, 3 * (i - 1) + 2
    col = sub[i - 1]
    fill.update({(0, z): col, (0, x): col})              # objective
    fill.update({(i, z): col, (i, x): col})              # local constraints (Omega_i)
    fill.update({(3 + i, z): col, (3 + i, 9): C0})       # consensus z_i - z
    fill.update({(7, z): col, (7, x): col})              # h0
    fill.update({(8, z): col, (8, x): col, (8, s): col})  # g0 + s
xs, ys = block_matrix(ax, 3.72, 2.25, cols, rows, fill)

for i in (1, 2, 3):                                       # u_i headers
    xa, xb = xs[3 * (i - 1)], xs[3 * i]
    ax.text((xa + xb) / 2, 2.5, f"$u_{i}$", ha="center", va="bottom", color=sub[i - 1])

ax.text(xs[-1] + 0.07, (ys[4] + ys[-1]) / 2, "$\\phi(u)$\nrelaxed\ninto $\\mathcal{L}$", va="center", color='k')

plt.savefig("option_b.pdf", bbox_inches="tight")
plt.show()
