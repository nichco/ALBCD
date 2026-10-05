# Airliner fleet trajectory optimization

An ALBCD example, with gradients from JAX, built so that ALBCD beats a monolithic optimizer
beyond a modest problem size. N flights of the same regional airliner, with ranges of
2,000-5,000 km and takeoff masses of 22-28 t sampled with a Latin hypercube, minimize the
fleet's total fuel subject to a total block-time budget 3 % tighter than the sum of their
minimum-fuel times. Each flight is the inverse-dynamics trajectory problem of CoTest's
`airliner/inverse_dynamics_bsplines.py`: altitude and airspeed are cubic B-splines in range,
the point-mass equations are solved for thrust and angle of attack, and mass and time are
integrated with RK4, with throttle and lift-coefficient limits along the flight.

Each flight is an ALBCD block that owns its trajectory and a block-time allocation, tied to
its flight time by a local constraint. The only coupling constraint is the budget,
`sum(tau_i) = T_budget`, which is linear in the allocations. Every flight couples to the
others through that one scalar, and once its multiplier is fixed the flights are
independent; the multiplier is the fleet's cost index (kg of fuel per second of block
time), about 0.07 kg/s here. The penalty follows the usual augmented Lagrangian rule (it
doubles whenever the budget violation doesn't halve), starting from `mu0 = 0.01`. The start
matters: the flights interact only through the penalty term, while one flight's fuel changes
with its own allocation with a curvature of only 0.1-0.2 t/ks^2, so a penalty much stiffer
than that (e.g. `mu0 = 1` at N = 4) makes the block sweeps crawl.

N is 4 by default; pass another as an argument, e.g. `python airliner_fleet.py 8` (and
likewise for `monolithic.py`).

| File | Contents |
| --- | --- |
| `airliner_fleet.py` | The fleet, the budget and the ALBCD solve. Run this. |
| `monolithic.py` | The same fleet and budget solved with a single SLSQP; writes the reference solution `monolithic_solution_N{N}.npz` |
| `models.py` | The aircraft and trajectory model of one flight and its constraints, shared by both scripts |
| `scaling.py` | Runs both for N = 1 to 16 and plots the solve times (`scaling.pdf`) |
| `atmos1976_jax.py`, `rk4_jax.py` | The standard atmosphere and RK4 integrator, copied from CoTest's `airliner` folder |

```bash
python examples/jax/airliner_fleet/airliner_fleet.py
```

## Scaling

Solve times on one Windows desktop, excluding JAX compilation, from `scaling.py`, computed
with `n_int = 50` spline intervals (102 variables per flight; `models.py` now uses 60):

| Flights N | Variables | Monolithic SLSQP | ALBCD | ALBCD block solves |
| --- | --- | --- | --- | --- |
| 1 | 102 | 2.0 s | 12.7 s | 14 |
| 2 | 204 | 14.1 s | 30.2 s | 38 |
| 4 | 408 | 69.6 s | 96.8 s | 80 |
| 8 | 816 | 1,075 s | 171 s | 152 |
| 16 | 1,632 | > 1,800 s (stopped) | 327 s | 320 |

ALBCD reaches the monolithic solution at every N where both finish: the fleet fuel agrees
to within 0.01 kg, the cost index to within 0.1 %, and the relative error in the variables
is 7e-5 to 3e-4. ALBCD needs about 20 block solves per flight at every N, so its cost grows
linearly; the two cost the same between N = 4 and 8, and at N = 8, ALBCD is about 6 times
faster.

The monolithic cost grows quickly because SLSQP works with dense matrices, with about N x 100
variables and about N x 900 inequality rows, while each ALBCD block solve has the size of a
single flight. Both use the same derivatives (one forward-mode Jacobian per flight gives the
constraint Jacobian and the fuel gradient), so the difference is in the optimizer, not the
model.

Some caveats:

- Timings on this machine vary noticeably between runs (ALBCD at N = 4 has taken from 46 s
  to 97 s with about the same number of block solves), so the block-solve counts are the more
  reliable measure for ALBCD.
- The monolithic N = 16 solve was stopped at the 30-minute limit of `scaling.py`; an earlier
  attempt ran for about 90 minutes without finishing.
- The baseline is dense SLSQP. A sparse optimizer such as IPOPT could exploit the same
  structure (independent flights plus one budget row) and would narrow the gap.
- ALBCD solves the blocks one after another. Since the flights are independent given the
  multiplier and the others' allocations, they could also be solved in parallel.
