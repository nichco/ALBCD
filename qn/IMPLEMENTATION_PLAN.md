# Implementation plan: `QuasiNewtonALBCD`

This plan is for an agent implementing it in the ALBCD repository. Read §0–§2 before writing code. Work through the tasks in §3 in order; each task has a definition of done. Stop and report if a definition of done can't be met. Don't work around one silently.

---

## 0. Context

ALBCD (`src/albcd/albcd.py`) solves

```
min Σᵢ fᵢ(xᵢ)   s.t.  φ(x) = 0  (coupling),  local constraints and bounds in each block
```

It alternates block coordinate descent (BCD) sweeps on the augmented Lagrangian (AL) `f + yᵀφ + ½ Σ μ φ²` with the first-order update `y ← y + μφ`. Both loops converge linearly, and the final accuracy plateaus at about 1e-4.

**Your task:** add a new solver class, `QuasiNewtonALBCD`. At every multiplier update it takes a **coupled quasi-Newton step**, computed block by block. It must:

1. converge in fewer sweeps and less wall time than ALBCD, and
2. reach a KKT measure of 1e-8,

on two problems: the **airliner fleet** (`examples/jax/airliner_fleet`) and **ranger** (`examples/jax/ranger`).

**Hard constraints:**

- **Every existing file in the repository is off-limits.** Don't edit, move, rename or delete any file that exists when you start. This includes `src/`, `tests/`, `examples/`, `hybrid/`, `old/`, `pyproject.toml`, the README, the `.npz` results, and the cache and `__pycache__` files (see "Not writing to existing files" below).
- **All new work goes in one new top-level folder, `qn/`** (layout in §0.1). Don't create new files in existing folders either.
- **If an existing file would need a change, copy it instead.** Copy it into `qn/` with a `_qn` suffix (e.g. `qn/subproblem_qn.py`), make the change in the copy, and import only the copy from your new code. Record every copy in the report (Task 7): the source file, what changed, and why. Prefer subclassing or wrapping over copying wherever it works. The plan below is written so that no copies should be needed.
- **No second derivatives anywhere** in the method: no `jax.hessian`, no `jax.jacfwd(jax.grad(...))`, no Hessian-vector products. Curvature comes only from damped BFGS on gradient differences. (Finite-difference Hessians may appear in *tests*, as a check, and nowhere else.)
- The solver code (`qn/qn_core.py`, `qn/qn_subproblem.py`, `qn/qn_albcd.py`) must depend only on **numpy** and the installed `albcd` package. Don't import scipy, jax or modopt there.
- Don't install new packages. The environment already has numpy, scipy, jax (with x64), modopt, matplotlib and the `albcd` package.
- Don't commit or push unless the user asks.

**Not writing to existing files.** Running Python and pytest normally rewrites existing bytecode and cache files in the repository. Avoid this:

- Set `PYTHONDONTWRITEBYTECODE=1` in the environment of every Python and pytest command.
- Run pytest with `-p no:cacheprovider`, so it doesn't write to `.pytest_cache`.
- Every output (`.npz`, `.txt`, `.png`, `.md`) goes under `qn/`.
- **Before starting**, save the output of `git --no-optional-locks status --porcelain` to `qn/git_status_before.txt`. **At the end**, run it again and save it to `qn/git_status_after.txt`. The only difference allowed is the new untracked `qn/` folder. (One file, `examples/jax/ranger/monolithic_viz_2.py`, is already modified before you start; leave it alone.)

### 0.1 Layout of the new folder

```
qn/
  IMPLEMENTATION_PLAN.md   this plan (already present when you start; read only)
  qn_subproblem.py         Task 1: QNSubproblem, a subclass of albcd.Subproblem
  qn_core.py               Task 2: pure numerical functions
  qn_albcd.py              Task 3: QuasiNewtonALBCD, a subclass of albcd.ALBCD
  airliner_blocks.py       Task 4: airliner blocks with grad_phi()
  ranger_blocks.py         Task 4: ranger blocks with grad_phi()
  monolithic_tight_ranger.py, monolithic_tight_airliner.py     Task 5
  experiments_airliner.py, experiments_ranger.py               Task 6
  conftest.py              registers the "slow" marker, puts qn/ on sys.path
  tests/test_qn_core.py, tests/test_qn_albcd.py, tests/test_qn_examples.py
  results/                 every .npz, .txt and .png written by Tasks 5–7
  QN_REPORT.md             Task 7
  git_status_before.txt, git_status_after.txt
```

**Imports:**
- New modules import the solver with `from albcd import ALBCD, Subproblem`, as the examples do.
- The example blocks need each example's `models.py`. Insert the example folder at the front of `sys.path` before importing it, e.g. `sys.path.insert(0, str(REPO / "examples" / "jax" / "airliner_fleet"))`. Both examples' model files resolve their data files relative to their own `__file__` (e.g. `ranger.stp`), so this works from any working directory.
- Run all tests from the repository root with `pytest -p no:cacheprovider qn/tests`. `qn/conftest.py` puts `qn/` on `sys.path`, and registers the marker with `config.addinivalue_line("markers", "slow: ...")` in `pytest_configure`, so `pyproject.toml` stays untouched.

**Read these files first (read only):**

- `src/albcd/albcd.py`, `src/albcd/subproblem.py`, `tests/test_albcd.py`
- `src/albcd/second_order_albcd.py`: shows the subclassing pattern (overriding `_update_y` and `_update_mu`)
- `examples/jax/airliner_fleet/airliner_fleet.py`, `models.py`, `monolithic.py`
- `examples/jax/ranger/ranger.py`, `models.py`, `monolithic.py`

**Note on the example scripts:** they run their solve and plotting at module level, so you **cannot import** their `Subproblem` classes. §3 Task 4 says how to handle this.

---

## 1. The math, with every convention fixed

### 1.1 Problem structure (required, and checked at runtime)

- **Separable objective:** f = Σᵢ fᵢ(xᵢ).
- **Separable coupling:** φ(x) = Σᵢ cᵢ(xᵢ). Each block supplies its own term cᵢ (length m_c) and its Jacobian Jᵢ = ∂cᵢ/∂xᵢ (m_c × nᵢ).
  - **Airliner** (m_c = 1): cᵢ = [τᵢ], with T_budget subtracted in block 0 only: c₀ = [τ₀ − T_budget].
  - **Ranger** (m_c = 83), from `coupling()` in `ranger.py`:
    - c_aero(v) = concat(1e-3·loads(twist), [−1e-4·weight_copy])
    - c_struct(v) = concat(−1e-3·loads_copy, [1e-4·weight(thickness, loads_copy)])
    - Their sum must equal `coupling(aero_loads, aero_loads_copy, weight, weight_copy)`.

### 1.2 Sign conventions

These are ALBCD's. Use them everywhere in the new code.

- **Lagrangian:** L = Σᵢ fᵢ + λᵀφ + Σᵢ κᵢᵀ(h_{i,A} − b_{i,A}).
  - h_{i,A} are block i's **active** local constraints. b_{i,A} is the bound each one is active at (cl or cu; equalities have cl = cu).
- **Stationarity residual of block i**, over its free variables:

  ```
  rᵢ = ∇fᵢ + Jᵢᵀλ + Gᵢᵀκᵢ,      Gᵢ = ∂h_{i,A}/∂xᵢ
  ```

- **Local constraint residual of block i:** gᵢ = h_{i,A}(xᵢ) − b_{i,A}.

### 1.3 Converting the block solvers' multipliers to κ

In both examples, `gradient()` computes `∇(subproblem objective) − (rows)ᵀ m`. Convert as follows:

- **Airliner (SLSQP).** `m = optimizer.results["multipliers"]` uses the row order of `kkt_rows(J)`: equalities, then lower-bounded inequalities, then upper-bounded inequalities, and the upper rows are negated. Write full-length κ (603 entries, one per local constraint) with:
  - κ_eq = −m_eq
  - κ_lower = −m_lower
  - κ_upper = +m_upper
  - Equality rows are always active. An inequality row is active if its |m| > `act_tol`·max(1, ‖m‖∞), with `act_tol` = 1e-8.
  - Non-active entries of κ are 0.
- **Ranger (OpenSQP), aero block.** The local constraint is `lift − weight_copy = 0`, always active. Set κ = −`self.multipliers`.
- **Ranger (OpenSQP), structures block.** The local constraint is `max_sigma − yield ≤ 0`, an upper bound with cu = 0. `solve()` already stores `self.multipliers = −pi` and `gradient()` uses `grad − jacᵀ self.multipliers`. So set κ = −`self.multipliers`. The constraint is active if |κ| > `act_tol`·max(1, |κ|) or |h| ≤ 1e-8.

Check every conversion with Task 4's sign test. Don't trust the derivation above without it.

### 1.4 Active bounds

- Variable j of block i is on a bound if xⱼ ≤ xlⱼ + `bnd_tol`·max(1, |xlⱼ|), or the same at xu, with `bnd_tol` = 1e-8.
- It is **fixed** (active) if it is on a lower bound with (rᵢ)ⱼ ≥ 0, or on an upper bound with (rᵢ)ⱼ ≤ 0.
- Fixed variables are removed from the step (Δxⱼ = 0) and from the stationarity measure.
- All other variables are **free**, and F denotes the free set.

### 1.5 Scaling

Every block has a scale vector D (the airliner's `x_scaler`; ranger's `X_SCALER`). All linear algebra and BFGS work in the scaled variables x̂ = D·x:

- ĝ = g / D (gradients, elementwise)
- Ĵ = J / D (Jacobians, columnwise)
- Δx = Δx̂ / D

### 1.6 KKT measure

```
θ = max( maxᵢ ‖(rᵢ)_F‖∞ ,  ‖φ‖∞ ,  maxᵢ localviolᵢ )
localviolᵢ = max over all local constraints of the violation of cl ≤ h ≤ cu, and of the bounds (0 if satisfied)
```

These are the unscaled residuals, consistent with ALBCD's `optimality` and `feasibility`. **The stopping test is θ ≤ `kkt_tol`, with default 1e-8.**

### 1.7 Block curvature: damped BFGS

The solver (not the block) keeps, for each block i, a dense SPD matrix Bᵢ (nᵢ × nᵢ, scaled variables). It approximates the block's **AL Hessian** ∇²ᵢ[fᵢ + λᵀcᵢ + ½μ‖φ‖² + κᵢᵀhᵢ] ≈ Hᵢ + μ Ĵᵢᵀ diag(1) Ĵᵢ.

Use the **vector μ** throughout: `μ` is an array of length m_c, so the penalty term is Ĵᵢᵀ diag(μ) Ĵᵢ.

**Initialization:**
- Bᵢ = I until the first pair with sᵀv > 0.
- On that pair, set Bᵢ = σI with σ = clip(vᵀv / sᵀv, 1e-6, 1e6), then apply the update.

**Powell-damped update** for a pair (s, v), with s = Δx̂ and v = Δ(scaled AL gradient):

```
skip if ‖s‖ ≤ 1e-12·max(1, ‖x̂‖)
Bs = B s;  sBs = sᵀBs;  sv = sᵀv
if sv < 0.2·sBs:  ω = 0.8·sBs/(sBs − sv);  v ← ω v + (1 − ω) Bs
B ← B − (Bs)(Bs)ᵀ/sBs + v vᵀ/(sᵀv)
symmetrize: B ← ½(B + Bᵀ)
```

**When μ changes** by Δμ (a vector): B ← B + Ĵᵢᵀ diag(Δμ) Ĵᵢ, using the current Ĵᵢ.

**Pair sources.** All gradients in pairs are scaled AL gradients of block i, with λ, μ, κᵢ and the other blocks held fixed:

```
∇ᵢAL(xᵢ) = ∇fᵢ(xᵢ) + Jᵢ(xᵢ)ᵀ(λ + μ ⊙ φ̃(xᵢ)) + ∇hᵢ(xᵢ)ᵀκᵢ,
φ̃(xᵢ) = cᵢ(xᵢ) + Σ_{j≠i} cⱼ(current xⱼ)
```

- **Outer pairs (free).** These are taken between two consecutive linearization points of block i, `old` and `new`. The solver stores each linearization's raw pieces (∇fᵢ, cᵢ, Jᵢ, hᵢ, ∇hᵢ). It assembles both gradients with the **new** λ, μ and κᵢ, and with φ̃ built from the new values of the other blocks' cⱼ. So no extra function evaluations are needed.
- **Harvested pairs (opt-in, free).** During `solve()`, a block may record each point its optimizer evaluates as `(v, ∇(subproblem objective)(v), ∇h(v))`. The subproblem objective is the block's AL at the solve's (y, μ). The solver forms the Lagrangian gradient `∇obj + ∇hᵀκ_final`, where κ_final comes from that solve (§1.3). It takes pairs between consecutive distinct recorded points in evaluation order.
- **Skip a pair** if block i's active set (the set of active local constraints and fixed variables) differs between its two points. For harvested pairs, use the final active set for all points; don't skip them.

### 1.8 The coupled step (the condensed system)

Given the current x, multipliers λ (= ŷ, see §1.9) and κᵢ, each block's linearization (raw pieces), free set Fᵢ, active rows Aᵢ and Bᵢ:

**Per block i** (independent, so parallelizable):

```
B_F = Bᵢ[F, F]                 (n_F × n_F)
Ĝ  = ∇ĥᵢ[A, F]                 (n_A × n_F)    (scaled: columns divided by D[F])
Ĵ  = Ĵᵢ[:, F]                  (m_c × n_F)
r̂  = (r̂ᵢ)_F = (∇f̂ᵢ + Ĵᵢᵀλ + ∇ĥᵢ[A]ᵀκᵢ[A])_F
g  = hᵢ[A] − b[A]
K  = [[B_F, Ĝᵀ], [Ĝ, 0]]
factor K once (np.linalg.solve on stacked right-hand sides is fine; sizes ≤ ~110)
W  = K⁻¹ [Ĵᵀ; 0]               ((n_F + n_A) × m_c)
u  = K⁻¹ [r̂; g]                ((n_F + n_A))
Sᵢ = Ĵ W[:n_F]                 (m_c × m_c)
sᵢ = Ĵ u[:n_F]                 (m_c)
```

**Coordinator:**

```
S = Σᵢ Sᵢ ;   symmetrize
Δλ = solve(S + ε·tr(S)/m_c·I, φ − Σᵢ sᵢ),   ε = 1e-14  (guards against redundant coupling rows)
```

**Per block i, back-substitution:**

```
[Δx̂_F; Δκ_A] = −(u + W Δλ)
Δxᵢ = 0 on fixed variables;  Δxᵢ[F] = Δx̂_F / D[F]
```

**Refinement** (option `refine`: 0 or 1, default 1). Bᵢ includes the block-diagonal penalty curvature Ĵᵢᵀdiag(μ)Ĵᵢ but not the cross-block terms, so the step above solves a slightly different system. One correction round:

```
rerun the condensed system with the SAME factorizations, replacing:
    r̂ ← −Ĵᵀ diag(μ) (Ĵ Δx̂_F)     (per block)
    g ← 0
    φ ← 0
add the resulting (Δx̂, Δκ, Δλ) to the first step
```

**Trial point:**

```
x_trial = clip(x + Δx, xl, xu)
κ_trial = κ + Δκ   (on active rows; 0 elsewhere)
λ_trial = λ + Δλ
```

### 1.9 Multiplier estimate at a BCD point

After a sweep, the step is taken at λ = ŷ = y + μ ⊙ φ (ALBCD's first-order estimate), not at y.

---

## 2. Algorithm

### 2.1 Options

| Option | Default | Meaning |
|---|---|---|
| `mode` | `"adaptive"` | `"dual"` (V1), `"primal_dual"` (V2) or `"adaptive"` (V3) |
| `sweeps` | `1` | sweeps per outer iteration in V2/V3. V1 always uses ALBCD's own inner loop (`max_inner_iter`, inexact tolerance), because it needs x close to an AL minimizer |
| `kkt_tol` | `1e-8` | stopping test θ ≤ `kkt_tol` |
| `accept` | `0.9` | accept a trial if θ_trial ≤ `accept`·θ_current |
| `skip_factor` | `5.0` | V3: skip the next sweep if θ_trial ≤ θ_current / `skip_factor` |
| `refine` | `1` | refinement rounds (0 or 1) |
| `pairs` | `"both"` | `"outer"`, `"harvest"` or `"both"` |
| `dual_cap` | `10.0` | fallback multiplier step allowed if ‖Δλ‖∞ ≤ `dual_cap`·max(1, ‖ŷ‖∞) |
| `mu_growth_after` | `2` | consecutive rejections before ALBCD's μ growth rule is applied |
| `act_tol`, `bnd_tol` | `1e-8` | §1.3, §1.4 |

All other ALBCD constructor arguments pass through unchanged (`mu0`, `max_mu`, `rho`, `tau`, `max_outer_iter`, `max_inner_iter`, `max_y`, `kappa`, `verbose`, …).

### 2.2 Outer iteration (V2 and V3)

```
state: x, data, y, mu, kappa_i (per block), B_i (per block), lin_i (last linearization per block)
skip = False
for k in 1..max_outer_iter:
    if not skip:
        run `sweeps` sweeps (same loop body as ALBCD._block_coordinate_descent, without its stopping test)
        for each block: kappa_i ← converted solver multipliers (§1.3); harvest pairs if enabled
        lam ← y + mu ⊙ phi
    else:
        lam ← y   (y was set from the last accepted step)
    lin_i ← block.grad_phi(x, data) for every block; update B_i with outer pairs from the previous lin_i
    theta ← θ(x, lam, kappa)  (§1.6);  record it
    if theta ≤ kkt_tol: success; break
    compute the step (§1.8), the trial point, then lin_trial ← grad_phi at the trial point
    theta_trial ← θ(x_trial, lam_trial, kappa_trial)
    accepted ← theta_trial ≤ accept·theta
               and no active inequality's κ changed sign
               (sign rule: κ ≤ 0 for rows active at cl, κ ≥ 0 for rows active at cu)
    outer pair: update B_i between lin_i and lin_trial (whether or not the step is accepted)
    if accepted:
        x, data, phi ← trial values (data from lin_trial outputs; phi = Σ c_i at trial)
        y ← lam_trial;  kappa_i ← kappa_trial;  lin_i ← lin_trial;  rejections ← 0
        skip ← (mode == "adaptive") and theta_trial ≤ theta / skip_factor
        if theta_trial ≤ kkt_tol: success; break
    else:
        rejections += 1;  skip ← False
        if ‖Δλ‖∞ ≤ dual_cap·max(1, ‖lam‖∞):  y ← lam + Δλ   (event "dual_fallback")
        else:                                 y ← lam        (event "first_order")
        if rejections ≥ mu_growth_after:
            mu_new ← ALBCD._update_mu(mu, phi, phi_old);  B_i += Ĵᵢᵀ diag(mu_new − mu) Ĵᵢ;  mu ← mu_new
    phi_old ← phi
```

### 2.3 V1 (`mode="dual"`)

1. Run ALBCD's inner loop unchanged.
2. Set λ = ŷ, linearize, compute the step with `refine`.
3. Apply **only** y ← ŷ + Δλ if ‖Δλ‖∞ ≤ `dual_cap`·max(1, ‖ŷ‖∞). Otherwise use y ← ŷ.
4. x is never changed by the step. μ is fixed unless the feasibility after the next inner loop is worse than before. In that case, use the first-order update for the next iteration and apply ALBCD's μ rule.
5. Stop on θ ≤ `kkt_tol`, or on ALBCD's own test if θ can't reach it within `max_outer_iter`.

### 2.4 History

Add to `self.history`, one entry per **event**:

- `"event"`: one of `sweep`, `qn_accept`, `qn_reject`, `dual_fallback`, `first_order`, `mu_growth`
- `"theta"`: θ after the event (NaN if not evaluated)
- `"time"`: seconds since the start of `solve()`
- `"t_parallel"`: cumulative critical-path time. For a sweep, add the sum of block solve times (sweeps are sequential). For a step, add the maximum over blocks of (linearize + factorization time), plus the coordinator time.
- counters `"n_solves"`, `"n_linearize"`, `"n_pairs"` (cumulative)

Keep ALBCD's existing `"x"`, `"optimality"` and `"feasibility"` entries per sweep, unchanged.

---

## 3. Tasks

### Task 1: Block interface

Create `qn/qn_subproblem.py` with `class QNSubproblem(Subproblem)`, where `Subproblem` comes from the installed `albcd` package. Don't touch `src/albcd/subproblem.py`. The subclass adds only these:

```python
x_scale = None   # class attribute: scale vector D for this block's variables (None → ones)

def grad_phi(self, x, data) -> dict:
    """First-order information of this block at the global x, for QuasiNewtonALBCD. Optional.

    Returns a dict with, for this block's variables v = x[self.index] (unscaled):
        grad_f : (n,)        gradient of this block's objective term f_i(v) (no multiplier or penalty terms,
                             no proximal term)
        c      : (m_c,)      this block's coupling term c_i(v); sum over blocks = phi(x)
        J      : (m_c, n)    dc_i/dv
        h      : (n_h,)      local constraint values (n_h may be 0)
        Jh     : (n_h, n)    dh/dv
        cl, cu : (n_h,)      local constraint bounds (cl == cu for equalities)
        xl, xu : (n,)        variable bounds (+-inf where unbounded)
        outputs: dict        the shared data this block writes in solve(), recomputed at x without
                             solving (may be empty)
    """
    raise NotImplementedError

def local_multipliers(self) -> np.ndarray:
    """Multipliers kappa of the local constraints from the latest solve(), length n_h, in the convention
    grad_f + J^T lam + Jh^T kappa = 0 on the active rows, and 0 on inactive rows. Optional."""
    raise NotImplementedError
```

Optional harvesting contract: if a block sets `self.curvature = [(v, grad_obj, Jh), ...]` during `solve()`, the solver consumes and clears it after each solve.

Don't add anything to the `albcd` package's exports. New code imports the new classes from the `qn/` modules directly.

**Done when:** `qn/qn_subproblem.py` imports cleanly, `qn/git_status_before.txt` exists, and `PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider tests/test_albcd.py` still passes. That run is read-only: it confirms the existing package works in your environment.

### Task 2: Numerical core (numpy only)

Create `qn/qn_core.py` with pure functions, no solver state:

- `damped_bfgs_update(B, s, v) -> B` and `initial_scale(s, v) -> sigma` (§1.7)
- `active_sets(lin, kappa, r, act_tol, bnd_tol) -> (free_mask, active_rows, bound_at)`. Here `bound_at` holds the b values of the active rows (§1.3–1.4).
- `block_condense(B, lin, D, lam, kappa, free, rows, bvals) -> dict(K_factors, W, u, S, s, ...)` (§1.8)
- `coordinator_solve(S_list, s_list, phi) -> dlam`
- `back_substitute(block_data, dlam) -> (dx, dkappa)`
- `kkt_measure(lins, lam, kappas, free_masks, phi) -> theta` (§1.6)

Create `qn/tests/test_qn_core.py` with:

1. **BFGS:** the secant condition B⁺s = v holds when sᵀv ≥ 0.2·sᵀBs. B⁺ stays SPD under random pairs with negative sᵀv. A Δμ update equals adding Ĵᵀdiag(Δμ)Ĵ.
2. **Condensed step = assembled step:** for random problems (3 blocks, nᵢ ∈ {5, 8, 12}, m_c ∈ {1, 4}, 0–3 active rows per block, 0–2 fixed variables per block, random SPD Bᵢ), the step from §1.8 with `refine=0` matches the solution of the full bordered system built from the same Bᵢ, to 1e-10. Build the full system **only in the test**.
3. **Refinement:** with Hᵢ := Bᵢ − Ĵᵢᵀdiag(μ)Ĵᵢ chosen so that the reduced Hessian is positive definite, repeating the refinement many times converges to the solution of the true system (H blocks plus the full penalty-free coupling). One round reduces the error by at least 10× on these tests.
4. **Active sets:** hand-built cases for equality rows, lower and upper rows, and bounds with each sign of r.

**Done when:** `pytest -p no:cacheprovider qn/tests/test_qn_core.py` passes.

### Task 3: Solver class

Create `qn/qn_albcd.py` with `class QuasiNewtonALBCD(ALBCD)`, where `ALBCD` comes from the installed `albcd` package, implementing §2 using `qn_core`. Reuse `ALBCD._update_mu` by calling it; don't copy it. Write the sweep helper in the subclass, mirroring the body of `ALBCD._block_coordinate_descent` and appending to the same history keys. At the start of `solve()`, raise a clear error if any subproblem isn't a `QNSubproblem` that implements `grad_phi` and `local_multipliers`.

Add tests to `qn/tests/test_qn_albcd.py`, using the closed-form problem from `tests/test_albcd.py`. Write a **new** `Block` class in the test file: a `QNSubproblem` with the same `solve` and `gradient` as the existing test's `Block`, plus `grad_phi` and `local_multipliers`. Don't import from the existing test file. The problem is f = (x0 − 1)² + (x1 − 3)², φ = x0 − x1, with solution x = [2, 2], y = −2. There are no local constraints, so h is empty and κ has length 0.

1. With Bᵢ initialized to the true Hessians (2 for each block; add a test-only hook `B0={i: matrix}`), `mode="primal_dual"` reaches θ ≤ 1e-12 after its first step.
2. With default initialization, every mode reaches θ ≤ 1e-10 and x = [2, 2] to 1e-9, y = [−2] to 1e-8.
3. `mode="adaptive"` performs fewer sweeps than `mode="primal_dual"` on the same problem.
4. With every step rejected (`accept=0.0`), the solver behaves like ALBCD with fallbacks and still converges, and the history events show `qn_reject`.
5. **A separable problem with copies and a local constraint**, to exercise the path ranger needs. Use 2 blocks:
   - Block A owns (a, w) with f = (a − 1)², local constraint a² − w = 0, and c_A = [a³; −w].
   - Block B owns (l, t) with f = (t − 2)², local constraint l + t ≤ 10, bounds t ≥ 0, and c_B = [−l; t²].

   Build a reference solution with `scipy.optimize.minimize(method="SLSQP")` on the joint problem, with tight tolerances (tests may use scipy). Check that all modes reach θ ≤ 1e-9 and match the reference to 1e-6.

**Done when:** `pytest -p no:cacheprovider qn/tests/test_qn_albcd.py qn/tests/test_qn_core.py` passes.

### Task 4: Example blocks with `grad_phi`

The example scripts can't be imported (they run at module level), and they are off-limits anyway. Create **new** modules in `qn/` that copy the subproblem classes (each one subclassing `QNSubproblem`, not `Subproblem`) and add the new methods. Keep the copied code identical apart from the additions and the constructor changes listed below, and put a comment at the top naming the source file and lines. Import each example's `models.py` through `sys.path` (§0.1); don't copy `models.py`.

**`qn/airliner_blocks.py`:**

- `make_fleet(N)`: same ranges, masses, flights and T_budget as `airliner_fleet.py` (LatinHypercube seed 0; `T_budget = 0.9 * np.sum(0.276 + 5.468e-6 * ranges)`).
- `class FlightSubproblem(Subproblem)`: copy from `airliner_fleet.py`, with constructor `(i, flights, T_budget)` in place of the globals. Add:
  - `x_scale = x_scaler`
  - `grad_phi`:
    - grad_f = `outputs_jac(v)[-1]` (fuel gradient)
    - c = [τᵢ − (T_budget if i == 0 else 0)]
    - J = unit row at the last variable
    - h = `outputs(v)[:-1]`, Jh = `outputs_jac(v)[:-1]`
    - cl, cu, xl, xu from `models`
    - outputs = {}
  - `local_multipliers`: §1.3.
  - Harvesting: wrap the `grad` and `jac` lambdas passed to `mo.ProblemLite` so they append `(v.copy(), grad, jac)` to `self.curvature`. The gradient recorded is `objective_gradient`, which is the subproblem AL gradient.
- `make_solver(N, cls=ALBCD, **options)`: builds the solver with the example's settings: `mu0=[0.01]`, `max_mu=1e3`, `rho=2.0`, `tau=0.5`, `feas_tol=1e-4`, `opt_tol=1e-4`, `max_y=1e6`, `max_outer_iter=100`, `max_inner_iter=20`. Options override these.
- `compile_models(opt)`: calls each jitted function once.

**`qn/ranger_blocks.py`:**

- Copy `coupling`, `SQPSubproblem`, `AeroSubproblem` and `StructSubproblem` from `ranger.py`, with the proximal coefficient `tau = 0.0`.
- Add `x_scale = X_SCALER` to both blocks.
- `AeroSubproblem.grad_phi`, with v = [twist_cp, weight_copy]:
  - grad_f = gradient of 1e2·CD(twist), with 0 for weight_copy (`jax.grad`)
  - c = c_aero (§1.1), J = `jax.jacrev(c_aero)`
  - h = [lift − weight_copy], Jh via `self.jac`; cl = cu = [0]; bounds ±inf
  - outputs = {"aero_loads": …, "CD": …}
- `StructSubproblem.grad_phi`, with v = [thickness_cp, loads_copy]:
  - grad_f = 0
  - c = c_struct, J = `jax.jacrev(c_struct)`
  - h = [max_sigma − yield], cl = [−inf], cu = [0]
  - xl, xu = `XL`, `XU`
  - outputs = {"weight": …}
- `local_multipliers`: §1.3.
- Harvesting: wrap the `grad` and `jac` lambdas in `SQPSubproblem.sqp` the same way as in the airliner. Record gradients **without** the proximal term.
- Jit every new function once in `setup()`.
- `make_solver(cls=ALBCD, **options)` with the example's settings: `mu0 = full(83, 10.0)`, `max_mu=1e6`, `rho=1.2`, `tau=0.5`, `feas_tol=3e-4`, `opt_tol=1e-4`, `max_outer_iter=40`, `max_inner_iter=20`. Also the same `x0` and `data0` construction as `ranger.py`.

**Tests**, in `qn/tests/test_qn_examples.py`, skipped if jax or modopt is missing. Mark them slow with `pytest.mark.slow`. The marker is registered in `qn/conftest.py` (§0.1), not in `pyproject.toml`.

1. **Separability:** after one ALBCD sweep, Σᵢ grad_phi(x)["c"] equals `opt.phi` to 1e-12, on both problems (airliner with N = 2).
2. **Derivatives:** central finite differences of grad_f, J and Jh at the post-sweep point agree to 1e-5 relative.
3. **Multiplier signs:** after a block solve, the residual `grad_obj + Jhᵀκ` at the solved point (with grad_obj the subproblem AL gradient), restricted to free variables, has max-norm ≤ 10 × the block solver's tolerance. This check validates §1.3.
4. **Outputs:** grad_phi's `outputs` match what `solve()` wrote, when evaluated at the same x.

**Done when:** all tests under `qn/tests` pass.

### Task 5: Reference solutions

Create new scripts in `qn/`. They read nothing from the existing reference files except for the comparison below, and write only to `qn/results/`.

- `qn/monolithic_tight_ranger.py`: the same problem as `examples/jax/ranger/monolithic.py` (copy its objective and constraint definitions; import `models` through `sys.path`), solved with SLSQP `ftol=1e-14`, `maxiter=1000`. Saves `qn/results/monolithic_solution_tight_ranger.npz` (`twist_cp`, `thickness_cp`, `CD`). Skip the plotting and pyvista imports.
- `qn/monolithic_tight_airliner.py N`: the same as `examples/jax/airliner_fleet/monolithic.py` with `ftol=1e-12`. Saves `qn/results/monolithic_solution_tight_airliner_N{N}.npz`. Run it for N = 4. Run N = 8 only if N = 4 takes under 15 minutes; otherwise report the time and skip N = 8.

**Done when:** the files exist, and the objective matches the existing reference to its accuracy.

### Task 6: Experiments

Create `qn/experiments_airliner.py` and `qn/experiments_ranger.py`. Each runs a list of named configurations and saves `qn/results/qn_results_<problem>.npz`, holding every configuration's history (§2.4) plus ‖x − x\*‖ after each event against the tight reference from Task 5.

- Distance is measured in scaled variables.
- For ranger, use only the twist and thickness entries, as `ranger.py` does.
- Call `compile_models` before any timed solve.
- Set `OPENBLAS_NUM_THREADS=1` before importing numpy, as the examples do.

**Configurations** (the ALBCD settings are the example's unless stated):

| Name | Solver | Settings |
|---|---|---|
| `albcd` | `ALBCD` | the example's |
| `albcd_tight` | `ALBCD` | `opt_tol = feas_tol = 1e-9`, `max_outer_iter = 200` |
| `v1` | QN | `mode="dual"` |
| `v2` | QN | `mode="primal_dual"` |
| `v3` | QN | `mode="adaptive"` |
| `v3_mu_lo`, `v3_mu_hi` | QN | `mode="adaptive"`, with `mu0` divided and multiplied by 10 |
| `v3_outer`, `v3_harvest` | QN | `mode="adaptive"`, with `pairs` = `"outer"` and `"harvest"` |
| `v3_norefine` | QN | `mode="adaptive"`, `refine=0` |

All QN runs use `kkt_tol=1e-8` and `max_outer_iter=100`.

**Problem sizes:** airliner N = 4 for every configuration; N = 8 for `albcd`, `v2` and `v3`.

**Summary:** print a table and also write it to `qn/results/qn_summary_<problem>.txt`. Columns:
- configuration
- success
- final θ
- final ‖x − x\*‖
- sweeps, block solves, QN steps accepted and rejected
- wall time and critical-path time to θ ≤ 1e-4
- wall time and critical-path time to θ ≤ 1e-8
- total wall time

**Runtime budget:** ALBCD on ranger takes about 2 minutes and on the airliner with N = 8 about 1.5 minutes per run. If one configuration takes more than 3× the `albcd` time, stop it at `max_outer_iter` and report it. Don't tune.

**Done when:** both summaries exist and §4 can be filled in.

### Task 7: Report

Write `qn/QN_REPORT.md` containing:

- both summary tables
- the success criteria (§4), each marked met or not met
- per problem: the θ history of `albcd`, `v2` and `v3` against cumulative sweeps and against wall time, as matplotlib PNGs in `qn/results/`
- observations: acceptance rate, when skipping started, which curvature source helped, and the failure modes seen
- every copy of an existing file you made (source, change, reason), or the line "No existing files were copied."
- the comparison of `qn/git_status_before.txt` and `qn/git_status_after.txt`, confirming that only `qn/` is new
- every place where you deviated from this plan, and why

---

## 4. Success criteria

Evaluate these per problem: airliner (N = 4 and N = 8) and ranger.

1. **Accuracy:** `v2` or `v3` reaches θ ≤ 1e-8, and ‖x − x\*‖ is at the level of the tight reference's accuracy.
2. **Speed:** `v3`, or the best QN configuration, reaches θ ≤ 1e-4 in fewer sweeps **and** less wall time than `albcd` takes to meet its own tolerances.
3. **Rate:** over the last 5 accepted steps of `v3`, the ratio θₖ₊₁/θₖ decreases, which indicates superlinear convergence. If it doesn't, the report must say what limited it (curvature pairs, active-set changes, rejections).

If a criterion isn't met, **report it. Don't change the method to meet it.** Suggestions for changes go in the report.

---

## 5. Pitfalls specific to this code

- **SLSQP (airliner)** returns no bound multipliers. Fixed variables come from §1.4 only.
- **modopt scaling:** with `x_scaler`, user callbacks receive *unscaled* v. Record unscaled v in harvested curvature and scale it in the solver.
- **`last_call_cache`** in the airliner caches the last evaluation per point. Don't mutate arrays it returns, and `.copy()` before storing.
- **Ranger's `coupling()`** scales loads by 1e-3 and weight by 1e-4, and the `X_SCALER` of the load copies is 1e-2. Don't add further scaling.
- **φ consistency after a step.** After a QN step moves all blocks at once, ranger's `data["weight"]` and `data["aero_loads"]` must come from `grad_phi(...)["outputs"]` at the trial point. Otherwise the next block solve uses stale coupling data.
- **Stale `gradient()` state.** `gradient()` in both examples uses multipliers (and, in the airliner, Jacobian rows) cached during the last `solve()`. After a QN step this cache is stale. The QN solver must use θ from §1.6 and must not call `gradient()` for its decisions. ALBCD's per-sweep `optimality` history, computed after solves, stays valid.
- **The ranger structures block has no objective term**, and its load copies enter only linearly. Its Bᵢ gets curvature in those directions only through the penalty term, μ·(1e-3)² per copy in unscaled terms. Initial scaling and the Δμ updates matter there. If K is badly conditioned, report the condition numbers per step in the history (`"cond_K"`), but don't add regularization unless Task 3's tests require it.
- **The airliner has 602 inequality path constraints per flight.** Expect many active rows: the active rows must have full row rank on the free variables. If `np.linalg.solve` on K raises or produces a residual above 1e-8 relative, treat the step as rejected (event `qn_reject`, reason `"singular_K"`) and continue.
- **Avoid timing JAX compilation:** call `compile_models` and jit every new function in `setup()` before timed solves.
