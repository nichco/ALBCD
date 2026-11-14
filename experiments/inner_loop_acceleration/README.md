# Accelerating the ALBCD2 inner (BCD) loop: experiments

Prototype: `albcd2x.py` (`ALBCD2X`, a subclass of `albcd.albcd2.ALBCD2`; every option off gives
ALBCD2 exactly). Benchmarks:

    # toy problems (seconds each); TOY_SOLVER = lbfgsb (default) | bfgs | bfgs_warm
    python bench.py quad_copy_stiff,nonlin_copy,shared3,rosen_consensus baseline,sor1.3,AAreuse10,eta0.1,sor1.3+eta
    # real aerostructural example (JAX VLM + beam, SLSQP blocks); ~1 min per config
    python aero_bench.py baseline,sor1.3

Use the `jaxenv` interpreter. Cost is counted as block-solver function evaluations (`nfev`), since
wall time on this machine is noisy when several JAX processes run at once.

| option | idea | basis |
|---|---|---|
| `accel='sor', omega` | over-relax each block right after its solve (and the `data` it produced), projected onto the block's bounds | nonlinear SOR (Ortega–Rheinboldt); Ostrowski–Reich: converges for 0 < ω < 2 when the AL Hessian is SPD; Young: any 2-block split is consistently ordered, ω_opt = 2/(1+√(1-ρ_GS)) |
| `accel='sor_adaptive'` | ω from Young's formula with ρ estimated from successive sweep displacements | Hageman–Young adaptive SOR |
| `accel='anderson', m, reuse` | type-II Anderson on the sweep map u → G(u), u = (x, numeric data); `reuse` keeps secant columns across outer iterations | Anderson = GMRES on linear maps (Walker–Ni); = IQN-ILS in FSI coupling (Degroote), reuse = IQN-ILS(q) across time steps |
| `accel='aitken'` | Irons–Tuck dynamic relaxation | OpenMDAO NLBGS, FSI (Küttler–Wall) |
| `block_eta` | ask each block to solve only to `eta * (current inner tolerance)` via `data['_tol']` | inexact BCD (Tseng–Yun), inexact AL (Rockafellar, Eckstein–Silva) |
| `warm='secant'` | multi-secant predictor of the AL minimizer x(y, log mu) at the start of each outer iteration | predictor–corrector continuation (Allgower–Georg) |
| `inner_rule='relative'` | stop the sweeps when stationarity ≤ σ‖mu·phi‖ | Eckstein–Silva relative-error AL |
| `TOY_SOLVER=bfgs_warm` | each block's BFGS starts from its previous inverse-Hessian estimate | Hessian recycling across sequences of related problems (Morales–Nocedal) |

## Results (block-solver function evaluations, relative to baseline ALBCD2)

Settings for the toys: mu0 = 1, rho = 1.5, tau = 0.5, feas_tol = opt_tol = 1e-6. Aerostruct: the
settings of `examples/jax/aerostruct/aerostruct_albcd3.py`.

| config | quad_copy_stiff | nonlin_copy | shared3 (3 blocks) | rosen_consensus | aerostruct |
|---|---|---|---|---|---|
| SOR ω=1.3 | 0.82 | 0.44 | 0.70 | 1.28 | **0.75** (sweeps 84 → 43, converged) |
| adaptive SOR (Young) | 0.86 | 0.53 | 0.78 | 0.99 | – |
| Anderson m=8 | 0.86 | 0.67 | 0.48 | 1.00 | 1.65 (AA m=5) |
| Anderson + reuse (IQN-ILS) m=10 | 0.81 | 0.58 | 0.44 | 1.00 | did not converge in 60 outer iterations |
| block_eta = 0.1 | 0.43 | 0.69 | 0.43 | 0.87 | did not converge in 60 (μ grew to 3e3 vs 285) |
| SOR 1.3 + block_eta | 0.31 | 0.27 | 0.31 | 1.09 | did not converge in 60 |
| Anderson-reuse + block_eta | 0.31 | 0.38 | 0.21 | 0.82 | – |
| relative inner rule | 1.31 | 0.65 | 0.65 | 0.48 | 1.59 |
| secant warm start | 0.83 | 1.35 | 1.05 | 0.90 | – |

Stiff regime (mu0 = 100, so the inner loop takes hundreds of sweeps per outer iteration):

| config | quad_copy_stiff | nonlin_copy |
|---|---|---|
| SOR ω=1.3 | 0.59 | 0.57 |
| Anderson m=8 | 0.23 | 0.16 |
| Anderson + reuse m=10 | 0.13 | 0.52 (6 safeguard restarts) |
| SOR 1.3 + block_eta | 0.24 | 0.25 |

Hessian recycling (BFGS block solver, unbounded toys; relative to cold-start BFGS baseline):

| config | quad_copy_stiff | shared3 | rosen_consensus |
|---|---|---|---|
| warm Hessian only | 0.55 | 0.35 | 0.82 |
| warm Hessian + block_eta | 0.05 | 0.06 | 0.35 |
| warm Hessian + SOR 1.3 + block_eta | **0.05** | **0.04** | 0.53 |
