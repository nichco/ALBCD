"""QuasiNewtonALBCD: ALBCD with a coupled quasi-Newton step, computed block by block, at every multiplier update.

See IMPLEMENTATION_PLAN.md §1-§2. After the block sweeps, every block linearizes itself at the current x
(QNSubproblem.grad_phi). With the multiplier estimate lam = y + mu * phi, the local-constraint multipliers kappa_i
from the block solves and a damped-BFGS model B_i of each block's augmented-Lagrangian Hessian, the solver takes
one Newton step on the KKT conditions of the whole problem. The step is computed by condensing every block onto
the coupling multipliers (one small factorization per block, independent of the others), solving an m_c x m_c
system for the multiplier step, and back-substituting. The trial point is accepted if it reduces the KKT measure
theta (qn_core.kkt_measure) by the factor ``accept``. Otherwise the solver falls back to ALBCD's multiplier update.

Modes
    "dual"         (V1) ALBCD's inner loop, then only the multiplier part of the step: y <- lam + dlam.
    "primal_dual"  (V2) ``sweeps`` sweeps, then the full step (x, kappa, y).
    "adaptive"     (V3) V2, but the next sweep is skipped after a step that reduced theta by ``skip_factor``.
    "aladin"       V2's step, applied without evaluating it, ALADIN-style: x + dx and lam + dlam become the starting
                   point and multipliers of the next sweep. Blocks are reached only through solve() and grad_phi at
                   the points their own solves produced (no trial-point grad_phi, no theta_trial, no skipping).
                   Shared outputs at the new point come from QNSubproblem.predict_outputs (linear predictions).
                   With ``guard`` (default), a step whose next sweep ends with theta > guard_accept * theta before
                   the step is undone (the solver returns to the pre-step point) and V2's fallback multiplier
                   update is applied instead.

The solver depends only on numpy and albcd; blocks must be QNSubproblem instances that implement grad_phi and
local_multipliers.
"""

import copy
import time
import numpy as np
from albcd import ALBCD

import qn_core as qc
from qn_subproblem import QNSubproblem

MODES = ("dual", "primal_dual", "adaptive", "aladin")
PAIRS = ("outer", "harvest", "both")


def linearization(sp, x, data):
    """Block sp's grad_phi at x, as float arrays of consistent shapes, plus lin["v"] = x[sp.index]."""
    lin = dict(sp.grad_phi(x, data))
    lin["v"] = np.array(x[sp.index], dtype=float)
    n = len(lin["v"])
    for key in ("grad_f", "c", "h", "cl", "cu", "xl", "xu"):
        lin[key] = np.atleast_1d(np.asarray(lin[key], dtype=float))
    lin["J"] = np.asarray(lin["J"], dtype=float).reshape(len(lin["c"]), n)
    lin["Jh"] = np.asarray(lin["Jh"], dtype=float).reshape(len(lin["h"]), n)
    lin.setdefault("outputs", {})
    return lin


def point(sp, lin, lam, kappa, act_tol=1e-8, bnd_tol=1e-8):
    """A linearization of block sp with its active sets at (lam, kappa)."""
    kappa = qc.kappa_or_zeros(kappa, len(lin["h"]))
    r = qc.stationarity(lin, lam, kappa)
    free, rows, bvals = qc.active_sets(lin, kappa, r, act_tol, bnd_tol, getattr(sp, "tight_tol", None))
    return dict(lin=lin, D=sp.scale(len(lin["v"])), free=free, rows=rows, bvals=bvals,
                key=(tuple(rows), tuple(np.flatnonzero(~free))))


class QuasiNewtonALBCD(ALBCD):

    def __init__(self, subproblems, x0, *args,
                 mode: str = "adaptive",
                 sweeps: int = 1,
                 kkt_tol: float = 1e-8,
                 accept: float = 0.9,
                 skip_factor: float = 5.0,
                 refine: int = 1,
                 pairs: str = "both",
                 dual_cap: float = 10.0,
                 mu_growth_after: int = 2,
                 act_tol: float = 1e-8,
                 bnd_tol: float = 1e-8,
                 guard: bool = True,
                 guard_accept: float = 1.0,
                 B0: dict = None,
                 **kwargs):

        super().__init__(subproblems, x0, *args, **kwargs)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        if pairs not in PAIRS:
            raise ValueError(f"pairs must be one of {PAIRS}, not {pairs!r}")
        self.mu = np.array(self.mu, dtype=float)  # a copy: ALBCD keeps the caller's mu0 array
        self.y = np.zeros_like(self.mu)
        self.mode = mode
        self.sweeps = int(sweeps)
        self.kkt_tol = kkt_tol
        self.accept = accept
        self.skip_factor = skip_factor
        self.refine = int(refine)
        self.pairs = pairs
        self.dual_cap = dual_cap
        self.mu_growth_after = mu_growth_after
        self.act_tol = act_tol
        self.bnd_tol = bnd_tol
        self.guard = guard                # "aladin" mode only
        self.guard_accept = guard_accept  # "aladin" mode only
        self.B0 = {} if B0 is None else dict(B0)

        nb = len(self.subproblems)
        self.B = [None] * nb            # damped-BFGS model of each block's AL Hessian (scaled variables)
        self._B_init = [False] * nb     # True once B_i has been initialized by a pair (or by B0)
        self.kloc = [None] * nb        # local-constraint multipliers of each block
        self.points = [None] * nb       # latest linearization of each block, with its active sets
        self.theta = np.nan
        self.status = ""

    # ------------------------------------------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------------------------------------------

    def _check_blocks(self):
        for i, sp in enumerate(self.subproblems):
            ok = isinstance(sp, QNSubproblem)
            ok = ok and type(sp).grad_phi is not QNSubproblem.grad_phi
            ok = ok and type(sp).local_multipliers is not QNSubproblem.local_multipliers
            if not ok:
                raise TypeError(f"QuasiNewtonALBCD: subproblem {i} ({type(sp).__name__}) must be a QNSubproblem that "
                                "implements grad_phi() and local_multipliers()")
            sp.harvest = self.pairs in ("harvest", "both")
            sp.act_tol = self.act_tol

    def _event(self, name, theta=np.nan, dt_par=0.0, cond_K=np.nan, reason=""):
        self._t_par += dt_par
        record = dict(event=name, theta=float(theta), time=time.perf_counter() - self._t0, t_parallel=self._t_par,
                      n_solves=self._n_solves, n_linearize=self._n_lin, n_pairs=self._n_pairs, cond_K=float(cond_K),
                      reason=reason, x_event=self.x.copy(), y_event=self.y.copy(), mu_event=self.mu.copy())
        for key, value in record.items():
            self.history[key].append(value)

    def _patch_last_event(self, theta, dt_par=0.0):
        """Set theta of the latest event (a sweep, before the linearization that evaluates theta) and add the
        linearization's critical-path time to it."""
        self._t_par += dt_par
        record = dict(theta=float(theta), t_parallel=self._t_par, time=time.perf_counter() - self._t0,
                      n_linearize=self._n_lin, n_pairs=self._n_pairs)
        for key, value in record.items():
            self.history[key][-1] = value

    def _ensure_B(self, i, n):
        if self.B[i] is None:
            if i in self.B0:
                self.B[i] = np.array(self.B0[i], dtype=float).reshape(n, n)
                self._B_init[i] = True
            else:
                self.B[i] = np.eye(n)

    def _pair(self, i, s, v, xhat_norm):
        self.B[i], self._B_init[i], used = qc.apply_pair(self.B[i], s, v, self._B_init[i], xhat_norm)
        self._n_pairs += int(used)

    # ------------------------------------------------------------------------------------------------------
    # sweeps (mirrors ALBCD._block_coordinate_descent)
    # ------------------------------------------------------------------------------------------------------

    def _solve_block(self, i):
        sp = self.subproblems[i]
        outputs = {}
        sp.curvature = []
        t = time.perf_counter()
        sp.solve(self.x, self.y, self.mu, self.data, outputs)
        dt = time.perf_counter() - t
        self._n_solves += 1
        self.x = outputs.pop("x")
        self.phi = np.array(outputs.pop("phi", []), dtype=float).reshape(self.mu.shape)
        self.data.update(outputs)
        self.history["x"].append(self.x.copy())

        kappa = np.asarray(sp.local_multipliers(), dtype=float).copy()
        self.kloc[i] = kappa
        records, sp.curvature = getattr(sp, "curvature", []) or [], []
        if self.pairs in ("harvest", "both") and len(records) > 1:
            self._harvest(i, records, kappa)
        return dt

    def _harvest(self, i, records, kappa):
        """BFGS pairs between consecutive distinct points a block solve evaluated: the gradients are the
        subproblem's AL gradients plus Jh^T kappa_final, scaled. All points share the final active set."""
        sp = self.subproblems[i]
        v0 = np.asarray(records[0][0], dtype=float)
        D = sp.scale(len(v0))
        self._ensure_B(i, len(v0))
        prev = None
        for v, g, Jh in records:
            v = np.asarray(v, dtype=float)
            gL = np.asarray(g, dtype=float)
            if len(kappa):
                gL = gL + np.asarray(Jh, dtype=float).reshape(len(kappa), -1).T @ kappa
            if prev is not None and not np.array_equal(v, prev[0]):
                self._pair(i, D * (v - prev[0]), (gL - prev[1]) / D, np.linalg.norm(D * v))
            if prev is None or not np.array_equal(v, prev[0]):
                prev = (v, gL)

    def _inner_loop(self, n_sweeps, opt_tol=None):
        """n_sweeps sweeps, appending to ALBCD's history keys; with opt_tol, stops on ALBCD's inexact test."""
        for it in range(n_sweeps):
            t_sweep = sum(self._solve_block(i) for i in range(len(self.subproblems)))

            optimality = max(np.linalg.norm(sp.gradient(self.x, self.y, self.mu, self.data), ord=np.inf)
                             for sp in self.subproblems)
            self.history["optimality"].append(optimality)
            feas = np.max(np.abs(self.phi), initial=0.0)
            self.history["feasibility"].append(feas)
            self._event("sweep", np.nan, t_sweep)

            if opt_tol is not None:
                tol = opt_tol * max(1.0, self.kappa * feas / self.feas_tol)
                if optimality <= tol:
                    self._log(f"inner loop converged with optimality {optimality:.3e} (tolerance {tol:.3e}) in {it + 1} iterations")
                    break
        else:
            if opt_tol is not None:
                self._log(f"inner loop failed with optimality {optimality:.3e} in {it + 1} iterations")
        return optimality

    # ------------------------------------------------------------------------------------------------------
    # linearizations, active sets, theta and pairs
    # ------------------------------------------------------------------------------------------------------

    def _point(self, i, lin, lam, kappa):
        return point(self.subproblems[i], lin, lam, kappa, self.act_tol, self.bnd_tol)

    def _linearize(self, i, x):
        t = time.perf_counter()
        lin = linearization(self.subproblems[i], x, self.data)
        dt = time.perf_counter() - t
        self._n_lin += 1
        self._ensure_B(i, len(lin["v"]))
        return lin, dt

    def _theta(self, points, lam, kappas, phi, parts=False):
        return qc.kkt_measure([p["lin"] for p in points], lam, kappas, [p["free"] for p in points], phi, parts)

    def _outer_pair(self, i, old, new, lam, kappa, phi_new):
        """BFGS pair between two linearizations of block i, with both AL gradients assembled at (lam, mu, kappa)
        and the other blocks' coupling terms at their new values. Skipped if the active set changed."""
        if old is None or old["key"] != new["key"]:
            return
        lo, ln = old["lin"], new["lin"]
        others = np.asarray(phi_new) - ln["c"]
        k = qc.kappa_or_zeros(kappa, len(ln["h"]))
        g_new = qc.al_gradient(ln, lam, self.mu, ln["c"] + others, k)
        g_old = qc.al_gradient(lo, lam, self.mu, lo["c"] + others, k)
        D = new["D"]
        self._pair(i, D * (ln["v"] - lo["v"]), (g_new - g_old) / D, np.linalg.norm(D * ln["v"]))

    def _linearize_all(self, lam):
        """Linearize every block at self.x, update B_i with outer pairs from the previous linearizations, and
        return (points, theta, critical-path time)."""
        lins, dts = zip(*(self._linearize(i, self.x) for i in range(len(self.subproblems))))
        phi_sum = sum(lin["c"] for lin in lins)
        if not np.allclose(phi_sum, self.phi, rtol=1e-8, atol=1e-8 * max(1.0, np.max(np.abs(self.phi), initial=0.0))):
            raise ValueError("QuasiNewtonALBCD: the sum of the blocks' coupling terms c_i differs from phi "
                             f"(max difference {np.max(np.abs(phi_sum - self.phi)):.3e}); the coupling must be separable")
        points = [self._point(i, lin, lam, self.kloc[i]) for i, lin in enumerate(lins)]
        if self.pairs in ("outer", "both"):
            for i, (old, new) in enumerate(zip(self.points, points)):
                self._outer_pair(i, old, new, lam, self.kloc[i], self.phi)
        self.points = points
        self.kloc = [qc.kappa_or_zeros(k, len(p["lin"]["h"])) for k, p in zip(self.kloc, points)]
        theta = self._theta(points, lam, self.kloc, self.phi)
        return theta, max(dts)

    def _grow_mu(self, phi, phi_old):
        mu_new = self._update_mu(self.mu, phi, phi_old)
        dmu = mu_new - self.mu
        if np.any(dmu != 0):
            for i, p in enumerate(self.points):
                if p is not None and self.B[i] is not None:
                    self.B[i] = qc.penalty_update(self.B[i], p["lin"]["J"], p["D"], dmu)
            self.mu = mu_new
            self._event("mu_growth")

    # ------------------------------------------------------------------------------------------------------
    # the coupled step
    # ------------------------------------------------------------------------------------------------------

    def _step(self, lam):
        """The coupled step at self.points. Returns (dx_list, dkappa_list, dlam, max factorization time, coordinator
        time, max cond(K)), or None if some K is singular."""
        bds, t_fac, conds = [], [], []
        for i, p in enumerate(self.points):
            t = time.perf_counter()
            try:
                bd = qc.block_condense(self.B[i], p["lin"], p["D"], lam, self.kloc[i], p["free"], p["rows"], p["bvals"])
            except np.linalg.LinAlgError:
                return None
            t_fac.append(time.perf_counter() - t)
            conds.append(np.linalg.cond(bd["K"]) if bd["K"].size else 1.0)
            bds.append(bd)
        t = time.perf_counter()
        try:
            dx, dk, dlam = qc.coupled_step(bds, self.phi, self.mu, self.refine)
        except np.linalg.LinAlgError:
            return None
        t_coord = time.perf_counter() - t
        return dx, dk, dlam, max(t_fac), t_coord, max(conds)

    def _sign_ok(self, point, kappa):
        """No active inequality's multiplier has the wrong sign (kappa <= 0 at cl, kappa >= 0 at cu)."""
        lin = point["lin"]
        tol = self.act_tol * max(1.0, np.max(np.abs(kappa), initial=0.0))
        for j, b in zip(point["rows"], point["bvals"]):
            if lin["cl"][j] == lin["cu"][j]:
                continue
            if b == lin["cl"][j] and kappa[j] > tol:
                return False
            if b == lin["cu"][j] and kappa[j] < -tol:
                return False
        return True

    # ------------------------------------------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------------------------------------------

    def solve(self) -> "QuasiNewtonALBCD":
        self._check_blocks()
        self._t0 = time.perf_counter()
        self._t_par = 0.0
        self._n_solves = self._n_lin = self._n_pairs = 0
        # self.kappa is ALBCD's inexact-inner-loop factor; the local-constraint multipliers are self.kloc
        self.kloc = [None] * len(self.subproblems)
        if self.mode == "dual":
            self._solve_dual()
        elif self.mode == "aladin":
            self._solve_aladin()
        else:
            self._solve_primal_dual()
        self.tf = time.perf_counter() - self._t0
        self._log(f"{'QuasiNewtonALBCD converged' if self.success else 'QuasiNewtonALBCD failed'} "
                  f"(theta {self.theta:.3e}, {self.status}, {self.tf:.2f} s)")
        return self

    def _clip_y(self, y):
        return np.clip(y, -self.max_y, self.max_y)

    def _dual_ok(self, dlam, lam):
        """dlam is finite and within dual_cap of the multiplier scale."""
        return (dlam is not None and np.all(np.isfinite(dlam))
                and np.max(np.abs(dlam)) <= self.dual_cap * max(1.0, np.max(np.abs(lam))))

    def _solve_primal_dual(self):
        skip = False
        rejections = 0
        phi_old = np.full_like(self.mu, np.inf)
        for k in range(1, self.max_outer_iter + 1):
            if not skip:
                self._inner_loop(self.sweeps)
                lam = self.y + self.mu * self.phi
                theta, t_lin = self._linearize_all(lam)
                self._patch_last_event(theta, t_lin)
            else:
                lam = self.y
                # self.points are the accepted trial linearizations: recompute their active sets at (lam, kappa)
                self.points = [self._point(i, p["lin"], lam, self.kloc[i]) for i, p in enumerate(self.points)]
                theta = self._theta(self.points, lam, self.kloc, self.phi)
            self.theta = theta
            self._log(f"outer {k}: theta {theta:.3e}{' (sweep skipped)' if skip else ''}")
            if theta <= self.kkt_tol:
                self.success, self.status = True, "kkt_tol"
                return

            step = self._step(lam)
            accepted = False
            dlam = None
            if step is None:
                self._event("qn_reject", np.nan, reason="singular_K")
            else:
                dx, dk, dlam, t_fac, t_coord, cond = step
                x_trial = self.x.copy()
                kappa_trial = []
                for i, (sp, p) in enumerate(zip(self.subproblems, self.points)):
                    lin = p["lin"]
                    x_trial[sp.index] = np.clip(lin["v"] + dx[i], lin["xl"], lin["xu"])
                    kt = np.zeros(len(lin["h"]))
                    kt[p["rows"]] = self.kloc[i][p["rows"]] + dk[i][p["rows"]]
                    kappa_trial.append(kt)
                lam_trial = lam + dlam
                lins_t, dts = zip(*(self._linearize(i, x_trial) for i in range(len(self.subproblems))))
                phi_trial = sum(lin["c"] for lin in lins_t)
                points_t = [self._point(i, lin, lam_trial, kt) for i, (lin, kt) in enumerate(zip(lins_t, kappa_trial))]
                theta_trial = self._theta(points_t, lam_trial, kappa_trial, phi_trial)
                sign_ok = all(self._sign_ok(p, kt) for p, kt in zip(self.points, kappa_trial))
                accepted = (self.accept > 0 and np.isfinite(theta_trial) and theta_trial <= self.accept * theta
                            and sign_ok)

                # outer pair between the current and the trial linearizations, accepted or not
                if self.pairs in ("outer", "both"):
                    lam_p = lam_trial if accepted else lam
                    for i in range(len(self.subproblems)):
                        kap = kappa_trial[i] if accepted else self.kloc[i]
                        self._outer_pair(i, self.points[i], points_t[i], lam_p, kap, phi_trial)

                t_step = max(dts) + t_fac + t_coord
                if accepted:
                    self.x = x_trial
                    for lin in lins_t:
                        self.data.update(lin["outputs"])
                    self.phi = phi_trial
                    self.y = self._clip_y(lam_trial)
                    self.kloc = kappa_trial
                    self.points = points_t
                    rejections = 0
                    self.theta = theta_trial
                    self._event("qn_accept", theta_trial, t_step, cond)
                    skip = self.mode == "adaptive" and theta_trial <= theta / self.skip_factor
                    if theta_trial <= self.kkt_tol:
                        self.success, self.status = True, "kkt_tol"
                        return
                else:
                    reason = "sign" if not sign_ok else ("theta" if np.isfinite(theta_trial) else "nonfinite")
                    self._event("qn_reject", theta_trial, t_step, cond, reason=reason)

            if not accepted:
                rejections += 1
                skip = False
                if self._dual_ok(dlam, lam):
                    self.y = self._clip_y(lam + dlam)
                    self._event("dual_fallback")
                else:
                    self.y = self._clip_y(lam)
                    self._event("first_order")
                if rejections >= self.mu_growth_after:
                    self._grow_mu(self.phi, phi_old)
            phi_old = self.phi.copy()
        self.status = "max_outer_iter"

    def _solve_aladin(self):
        """ALADIN-style iteration: sweep, grad_phi at the post-sweep point, coupled step applied unevaluated.

        Events: "qn_step" when a step is applied (theta NaN: it is not evaluated), then "qn_accept" or, with the guard,
        "qn_reject" (reason "next_sweep") with the post-sweep theta the step led to.
        """
        rejections = 0
        phi_old = np.full_like(self.mu, np.inf)
        saved = None  # the post-sweep state at which the latest, not yet judged, step was taken
        for k in range(1, self.max_outer_iter + 1):
            self._inner_loop(self.sweeps)
            lam = self.y + self.mu * self.phi
            theta, t_lin = self._linearize_all(lam)
            self._patch_last_event(theta, t_lin)

            if saved is not None:  # judge the previous step by where the sweep from it ended
                if self.guard and not theta <= self.guard_accept * saved["theta"]:
                    self._event("qn_reject", theta, reason="next_sweep")
                    self.x, self.data, self.phi = saved["x"], saved["data"], saved["phi"]
                    self.kloc, self.points, self.theta = saved["kloc"], saved["points"], saved["theta"]
                    lam, dlam = saved["lam"], saved["dlam"]
                    saved = None
                    rejections += 1
                    if self._dual_ok(dlam, lam):
                        self.y = self._clip_y(lam + dlam)
                        self._event("dual_fallback")
                    else:
                        self.y = self._clip_y(lam)
                        self._event("first_order")
                    if rejections >= self.mu_growth_after:
                        self._grow_mu(self.phi, phi_old)
                    phi_old = self.phi.copy()
                    continue
                self._event("qn_accept", theta)
                rejections = 0
                saved = None

            self.theta = theta
            self._log(f"outer {k}: theta {theta:.3e}")
            if theta <= self.kkt_tol:
                self.success, self.status = True, "kkt_tol"
                return

            step = self._step(lam)
            if step is None:
                self._event("qn_reject", np.nan, reason="singular_K")
                self.y = self._clip_y(lam)
                self._event("first_order")
                phi_old = self.phi.copy()
                continue
            dx, _, dlam, t_fac, t_coord, cond = step
            saved = dict(x=self.x.copy(), data=copy.deepcopy(self.data), phi=self.phi.copy(),
                         kloc=[kl.copy() for kl in self.kloc], points=list(self.points), lam=lam, dlam=dlam,
                         theta=theta)
            x_new = self.x.copy()
            for i, (sp, p) in enumerate(zip(self.subproblems, self.points)):
                lin = p["lin"]
                v = np.clip(lin["v"] + dx[i], lin["xl"], lin["xu"])
                dv = v - lin["v"]
                x_new[sp.index] = v
                self.data.update(sp.predict_outputs(lin, dv))
                self.phi = self.phi + lin["J"] @ dv  # linear prediction; the next sweep recomputes phi
            self.x = x_new
            self.y = self._clip_y(lam + dlam)
            self._event("qn_step", np.nan, t_fac + t_coord, cond)
            phi_old = saved["phi"]
        self.status = "max_outer_iter"

    def _solve_dual(self):
        phi_old = np.full_like(self.mu, np.inf)
        feas_prev = np.inf
        for k in range(1, self.max_outer_iter + 1):
            optimality = self._inner_loop(self.max_inner_iter, self.opt_tol)
            lam = self.y + self.mu * self.phi
            theta, t_lin = self._linearize_all(lam)
            self._patch_last_event(theta, t_lin)
            self.theta = theta
            feas = np.max(np.abs(self.phi), initial=0.0)
            self._log(f"outer {k}: theta {theta:.3e}")
            if theta <= self.kkt_tol:
                self.success, self.status = True, "kkt_tol"
                return
            if feas > feas_prev:
                # the latest multiplier step made feasibility worse: first-order update and ALBCD's mu rule
                self.y = self._clip_y(lam)
                self._event("first_order")
                self._grow_mu(self.phi, phi_old)
            else:
                step = self._step(lam)
                if step is None:
                    self.y = self._clip_y(lam)
                    self._event("first_order", reason="singular_K")
                else:
                    dlam, t_fac, t_coord, cond = step[2:]
                    if self._dual_ok(dlam, lam):
                        self.y = self._clip_y(lam + dlam)
                        self._event("dual_fallback", np.nan, t_fac + t_coord, cond)
                    else:
                        self.y = self._clip_y(lam)
                        self._event("first_order", np.nan, t_fac + t_coord, cond, reason="dual_cap")
            feas_prev = feas
            phi_old = self.phi.copy()
        self.albcd_converged = bool(self.history["feasibility"][-1] <= self.feas_tol and optimality <= self.opt_tol)
        self.status = "max_outer_iter" + (" (ALBCD tolerances met)" if self.albcd_converged else "")
