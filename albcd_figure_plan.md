# ALBCD Paper — New Figure Plan

Task specs for adding high-impact figures to *"A Provably Convergent Distributed Architecture for Large-Scale Multidisciplinary Design Optimization"* (Orndorff & Hwang). Each option can be handed to Claude Code on its own.

**Audience:** aerospace MDO engineers and academics. Figures should be clean, academic, print-ready, and make one point each.

---

## 0. Instructions for Claude Code (read first)

- **Repo context:** the paper is `new-aiaa` LaTeX. The figures are PDFs included with `\includegraphics` (e.g., `fig_rho_sweep.pdf`, `fig_trajectories.pdf`, `ranger_problem.pdf`). The TikZ inputs are `gauss_seidel.tex` and `*_jac_dev.tex`. The ALBCD code is at `https://github.com/LSDOlab/ALBCD` (JAX/PyTorch examples).
- **Match the existing style:** before plotting, open an existing figure script (e.g., the ones producing `fig_bcd_demo.pdf`, `fig_albcd_demo.pdf`, `fig_trajectories.pdf`, `fig_convergence.pdf`). Reuse its fonts, rcParams, colormaps, line widths, and figure widths (`0.75–0.95\linewidth`). If you can't find any scripts, use matplotlib with a serif font (Times or newtx-compatible), font size 8–10 pt at final print size, no chart junk, and colorblind-safe colors.
- **Output:** a vector PDF named `fig_<name>.pdf`, the script that generates it (`scripts/fig_<name>.py`), and a LaTeX snippet (figure environment + caption + label) inserted at the location given below.
- **Never fabricate data.** If an option needs solver runs, run them or stop and report what's missing.
- **Verify:** after generating, open/render the PDF and check legibility at print width, label overlap, and that the caption's claims match the plot.

---

## Option 1 — Powell's cycling cube vs. proximal BCD  ⭐ top priority

**Section:** 4 (ALBCD convergence theory), placed right after the paragraph following Proposition 1 that cites Powell [`powell1973search`], before Proposition 2.

**Point:** without the proximal term (τ = 0), BCD can cycle forever through non-stationary points (Proposition 1 gives only partial stationarity). With any τ > 0, the iterates converge to a stationary point (Proposition 2). This is the paper's main theoretical contribution in one picture, and Section 4 currently has no figures.

**What to plot:**
- **Main panel (3D):** semi-transparent wireframe of the cube [−1, 1]³ inside the box [−2, 2]³.
  - τ = 0 path: partial-update iterates in one color with arrowheads. They converge onto a closed six-sided loop through the six vertices (1,1,−1) → (1,−1,−1) → (1,−1,1) → (−1,−1,1) → (−1,1,1) → (−1,1,−1). The loop goes around the long diagonal and avoids (1,1,1) and (−1,−1,−1).
  - τ > 0 path(s): accent color(s), breaking away from the loop to the stationary corner (−2,−2,−2).
  - Mark the start point. Use an isometric-ish view angle where the hexagon reads clearly.
- **Inset or side panel:** projected-gradient norm vs. BCD sweep (log y-axis). The τ = 0 line stays flat at 2.0; τ = 0.01, 0.1, 1 drop to 0 within about 3 sweeps. Points that are exactly 0 need a floor value (e.g., 1e-16) or a "→ 0" marker on a log axis.

**Problem definition:**
```
f(x) = -(x1*x2 + x2*x3 + x1*x3) + Σ_i max(|x_i| - 1, 0)^2,   x ∈ [-2, 2]^3
x0   = (-1 - e, 1 + e/2, -1 - e/4),  e = 1/10
```
Cyclic exact coordinate minimization of `f + (τ/2)(x_i - x_i^k)^2` over each coordinate.

**Critical implementation details (verified):**
1. **The box is required.** Without bounds, f is unbounded below along (1,1,1), which violates Assumption 1 (bounded level set). The box also matches the paper's Ω_i setting.
2. **Use exact rational arithmetic** (`fractions.Fraction`). The deviation from the vertices shrinks by 64× per sweep. In float64 it vanishes into round-off after about 6 sweeps, the coordinate step sees a tie, and the cycle breaks by accident, which would wrongly suggest that τ = 0 converges.
3. The τ > 0 limit (−2,−2,−2) is a KKT point of the box-constrained problem (normal-cone stationarity with active bounds), consistent with Eq. (stationary).

**Verified reference results:**
| τ | Behavior | Final f | Projected-gradient norm |
|---|---|---|---|
| 0 | cycles on hexagon | 1.0000 | 2.000 (constant) |
| 0.01 | reaches (−2,−2,−2) by sweep 3 | −9.0 | 0 |
| 0.1 | reaches (−2,−2,−2) by sweep 2 | −9.0 | 0 |
| 1.0 | reaches (−2,−2,−2) by sweep 3 | −9.0 | 0 |

**Reference implementation (core logic, verified):**
```python
from fractions import Fraction as F
B = F(2)
def clip(x, lo, hi): return max(lo, min(hi, x))
def pos(t): return max(abs(t) - 1, F(0))
def fval(x): return -(x[0]*x[1] + x[1]*x[2] + x[0]*x[2]) + sum(pos(t)**2 for t in x)

def coord_min(s, xk, tau):
    # argmin_x  -x*s + (|x|-1)_+^2 + tau/2 (x - xk)^2   on [-B, B]
    obj = lambda x: -x*s + pos(x)**2 + tau/2*(x - xk)**2
    c = [clip((s + 2 + tau*xk)/(2 + tau), F(1), B),
         clip((s - 2 + tau*xk)/(2 + tau), -B, F(-1))]
    c += [clip(xk + s/tau, F(-1), F(1))] if tau > 0 else [F(-1), F(1)]
    return min(c, key=obj)

def grad(x):
    g = [-(x[1]+x[2]), -(x[0]+x[2]), -(x[0]+x[1])]
    return [g[i] + 2*(1 if x[i] > 0 else -1)*pos(x[i]) for i in range(3)]

def proj_grad_norm(x):
    g = grad(x)
    return max(abs(x[i] - clip(x[i] - g[i], -B, B)) for i in range(3))

def run(tau, sweeps=12, e=F(1, 10)):
    x = [-1 - e, 1 + e/2, -1 - e/4]
    path = [list(x)]
    for _ in range(sweeps):
        for i in range(3):
            s = sum(x[j] for j in range(3) if j != i)
            x[i] = coord_min(s, x[i], tau)
            path.append(list(x))          # partial updates u^{k,i}
    return path
```
Convert to float only for plotting.

**Caption draft:** *Powell's example [ref] on the box [−2,2]³. Without a proximal term (τ = 0), cyclic BCD converges to a six-vertex cycle at which the projected gradient remains 2, so no limit point is stationary (cf. Proposition 1). With any τ > 0, the iterates converge to a stationary point within three sweeps (Proposition 2).*

**Acceptance criteria:** exact arithmetic is used; the hexagon is clearly visible; the inset shows a flat line vs. a drop; the caption values match the computed values.

---

## Option 2 — "The price of time" (fuel vs. flight-time trade-off)

**Section:** 5.2 (Multi-point airliner), after Fig. `fig:fleet` / Table 2.

**Point:** the single scalar multiplier λ* is the marginal fuel cost of one unit of flight time. At the solution, every flight sits where the slope of its fuel-vs-time curve equals −λ*, so all the tangent lines are parallel. This is the KKT condition made visible, and it shows why the problem decomposes so well: one price coordinates everything.

**What to plot:**
- For each of the N = 8 flights (same instance as Fig. `fig:fleet`): the curve fuel_i*(t_i) = minimum fuel for fixed range r_i and fixed flight time t_i. Expect a U-shape.
- On each curve:
  - ○ the unconstrained minimum-fuel point (slope 0, the dashed trajectories in Fig. `fig:fleet`)
  - ● the ALBCD time-limited solution
  - a short tangent segment with slope −λ* through ● (all segments parallel)
- Optional: faded markers of (t_i, fuel_i) at each outer iteration showing points sliding along the curves.
- Inset: λ^j vs. outer iteration j, converging to λ*.
- Colors: the same range colormap as `fig_trajectories.pdf` (2,200–5,000 km), with a shared colorbar.

**Data needed (new runs):**
- For each flight, sweep the fixed flight time over about 10–15 values bracketing both the unconstrained optimum and the time-limited solution, using the existing single-flight trajectory optimizer (implicit shooting, SLSQP via modOpt) with t_i fixed. That's about 100–120 small solves.
- Log λ, μ, and (t_i, fuel_i) at each outer iteration of the ALBCD run.

**Sanity checks:**
- The numerical slope d fuel_i / d t_i at the solution ≈ −λ* for all i (report the spread).
- The time limit is active (Σ t_i = T_limit, slack ≈ 0).
- Watch units (time in hours vs. days; the caption of Fig. `fig:fleet` uses a 1.6-day limit).

**Caption draft:** *Minimum fuel burn versus flight time for each of the N = 8 flights, colored by range. Open circles are each flight's unconstrained optimum; filled circles are the ALBCD solution under the time limit. At the solution, all slopes equal −λ\*, the marginal fuel cost of flight time; the inset shows the multiplier converging.*

---

## Option 3 — Disciplines negotiating (spanwise distributions)

**Section:** 5.1 (Distributed aerostructural), after Table 1 or after `fig:comp`.

**Point:** ALBCD recovers the monolithic design in the quantities wing designers actually look at, not just ‖x − x*‖. It also makes IDF consistency concrete: early on, the structure is sized for loads the wing doesn't produce yet.

**What to plot (3 or 4 panels vs. span y):**
1. Spanwise load: F(θ) (produced by aero) as solid lines and F̂ (assumed by structures) as dashed lines, at selected outer iterations j (sequential colormap from light = early to dark = late). Monolithic solution in black. An elliptical distribution with the same total lift in gray for reference.
2. Twist θ(y) at the same iterations plus monolithic.
3. Thickness t(y) at the same iterations plus monolithic.
4. Optional small panel: W(t) vs. Ŵ over outer iterations.

**Data needed:** existing ALBCD runs (baseline parameters: ρ = 1.2, γ = 0.5, μ₀ = 1, τ = 1e-4, η phases 1e-2 → 1e-4, ε = 1e-4). Add logging of θ, t, F̂, F(θ), Ŵ, W(t) at each outer iteration. No new optimization formulations.

**Premium variant:** reuse the rendering pipeline of `ranger_problem.pdf` to show a 3D wing filmstrip (3–4 iterations) with two colors of load arrows (aero-produced vs. structure-assumed) that merge.

**Caption draft:** *Spanwise loads, twist, and thickness at selected ALBCD outer iterations (light to dark) compared with the monolithic solution (black). Solid and dashed load curves are the aerodynamic loads F(θ) and the structural copy F̂; their agreement is the IDF consistency enforced by the relaxed constraints.*

---

## Option 4 — XDSM diagram of ALBCD

**Section:** 3.3 (Coordination scheme), beside Algorithm 1. Optionally a second one for the aerostructural decomposition in 5.1.

**Point:** XDSM (Lambe & Martins, 2012) is the standard notation for MDO architectures; the Martins & Lambe 2013 survey draws every architecture with it. An ALBCD XDSM lets readers compare it directly with ALC, CO, ATC, etc., and shows that only λ, μ and the variable copies pass between subproblems. That supports the proprietary/export-control argument in the Introduction.

**Content:**
- Drivers: outer AL loop (λ update Eq. multiplier_update, μ update Eq. penalty_update, stopping test ‖φ‖∞ ≤ ε) and inner BCD loop (stopping test r ≤ η).
- Optimizer blocks: subproblems 1…N (Eq. albcd_subproblem), each with its own local constraints g_i, h_i, s_i ≥ 0.
- Function block: central-variable update z (Eq. explicit_z_update), marked "central-variable formulation only".
- Data lines: u_i flowing between subproblems (most up-to-date values, Gauss–Seidel order); λ, μ to all subproblems; φ(u) back to the outer driver.
- Process line (thin black) showing the cyclic order.

**Tooling:** `pyXDSM` (github.com/mdolab/pyXDSM), which outputs TikZ and fits the existing `\input{...}.tex` workflow (the paper already loads `tikz`). Check its requirements (it may need extra TikZ libraries in the preamble) and keep notation consistent with the Nomenclature.

**Caption draft:** *Extended design structure matrix (XDSM) of the ALBCD architecture with N subproblems and central-variable consensus.*

---

## Option 5 — Consensus spreading across the graph

**Section:** 3.2.3 (Comparison of spanning-tree vs. central-variable consensus).

**Point:** the claim that "for some spanning trees, discrepancies propagate edge by edge, while the central variable propagates them immediately" is currently prose only. Showing it makes the consensus-layout choice a visible design decision.

**What to plot:**
- Grid of small multiples. **Rows:** chain spanning tree (z_i − z_{i+1}), star spanning tree (z_i − z_1), central variable (z_i − z). **Columns:** BCD sweeps 1, 2, 4, 8.
- Each panel is a graph of N = 8–12 nodes (subproblems), plus the hub node z for the central-variable row. Node fill = disagreement |z_i − z̄| on a shared sequential colormap with a colorbar.
- Right column: consensus error max_i |z_i − z̄| vs. sweep for all three layouts (log y).

**Toy problem:** one scalar global variable, N subproblems with local objectives f_i(z_i, x_i) that pull z_i toward different targets (e.g., (z_i − a_i)² + a weak coupling), run through the ALBCD package with identical λ, μ, τ settings across the three layouts. Report the settings in the caption or text.

**Note:** this needs a small experiment, so make sure the result actually supports the prose claim. If the chain doesn't come out slower, report that instead of tuning until it does.

---

## Option 6 — Quadratic penalty vs. augmented Lagrangian conditioning

**Section:** 3.1 (Constraint relaxation), after Fig. `global_failure`.

**Point:** a quadratic penalty only reaches the constraint as μ → ∞, and the subproblems become ill-conditioned along the way. The augmented Lagrangian reaches it with bounded μ by updating λ. This shows Section 3.1's main argument and anticipates the μ₀ result in Fig. `fig:cessna_mu_sweep`.

**What to plot (2 × 4 small multiples):**
- Problem: f(x, y) = x² + y² − 1.5xy with the equality version of the Fig. 3 constraint, x − 0.5y = 0 (state clearly that it's the equality version).
- **Top row:** quadratic penalty f + (μ/2)φ² for μ = 1, 10, 100, 1000. The contours squeeze into a narrow valley and the minimizer creeps toward the constraint line without reaching it.
- **Bottom row:** augmented Lagrangian with μ = 1 fixed, λ updated by λ ← λ + μφ for j = 0, 1, 2, 3. The contours stay round and the minimizer moves onto the constraint.
- Annotate each panel with the Hessian condition number κ and the distance of the minimizer from the constraint.

**Caveat:** Sections 2–3 already have three contour-plot figures (Figs. 2–4). Consider 3D surfaces or a more compact layout so this doesn't read as repetitive.

---

## Smaller upgrades to existing figures

### 7a. Phase portrait for `fig:comp` (ALBCD vs. ALC vs. ADMM)
Plot each method's path in the (feasibility ‖φ‖∞, optimality residual r) log-log plane, with markers at outer iterations and the origin direction marked "converged". You see the methods head for the origin, and ALC's divergence becomes a visible escape. It can replace or sit beside the current figure. It uses existing run data.

### 7b. Extend `fig:fleet_scaling` to the memory wall
- The dense SLSQP Jacobian alone is about 600N × 100N × 8 bytes ≈ 4.8 GB at N = 100 and ≈ 30 GB at N = 250, which hits the 32 GB laptop limit.
- Extend N until monolithic fails or exceeds a time cap, and shade a "monolithic infeasible" region.
- **Strongly recommended:** add IPOPT (sparse) as a third curve. The Discussion already says sparse optimizers could narrow the gap, so reviewers will ask for this.

### 7c. Free up space
Figs. `fig:cessna_rho_sweep`, `fig:cessna_mu_sweep`, and `fig:cessna_prox_sweep` share the same 3-panel layout. Merge them into one 3-row sensitivity figure (rows = ρ, μ₀, τ; columns = the same metrics) and update the text references.

---

## Suggested order of execution
1. **Option 1** (low effort, verified, fills Section 4)
2. **Option 2** (highest appeal for aerospace readers; needs about 100 small solves)
3. **Option 3** (only logging needed)
4. **Option 4** (low effort; expected by MDO reviewers)
5. 7c → 7a → 7b, then Options 5 and 6 as space allows
