"""Observations for QN_REPORT.md (Task 7), from qn/results/qn_results_*.npz. Writes qn/results/qn_observations.txt.

Per QN configuration: acceptance rate, rejection reasons, the first skipped sweep (V3), cond(K), mu growth, and the
rate over the last 5 accepted steps: the step contraction theta_trial / theta_before for each accepted step, and
the ratio of successive accepted thetas.
"""

from pathlib import Path
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"


def analyze(d, prefix, name):
    ev = d[f"{prefix}{name}__event"]
    th = d[f"{prefix}{name}__theta"]
    reason = d[f"{prefix}{name}__reason"]
    cond = d[f"{prefix}{name}__cond_K"]
    mu = d[f"{prefix}{name}__mu_max"]
    lines = []
    acc = np.flatnonzero(ev == "qn_accept")
    rej = np.flatnonzero(ev == "qn_reject")
    n_steps = len(acc) + len(rej)
    if n_steps:
        reasons = {r: int(np.sum(reason[rej] == r)) for r in sorted(set(reason[rej]))}
        lines.append(f"  steps {n_steps}: accepted {len(acc)} ({100 * len(acc) / n_steps:.0f}%), rejected {len(rej)} {reasons}")
        # theta before each step: the latest finite theta before the step event
        before = []
        for k in acc:
            prev = [j for j in range(k) if np.isfinite(th[j]) and ev[j] in ("sweep", "qn_accept")]
            before.append(th[prev[-1]] if prev else np.nan)
        contraction = th[acc] / np.array(before)
        last = acc[-5:]
        lines.append("  last 5 accepted: theta " + ", ".join(f"{t:.2e}" for t in th[last]))
        lines.append("    step contraction theta_trial/theta_before: " + ", ".join(f"{c:.3f}" for c in contraction[-5:]))
        succ = th[last][1:] / th[last][:-1]
        lines.append("    successive accepted theta ratio: " + ", ".join(f"{c:.3f}" for c in succ))
        # first skip: an accepted step directly followed by another step (no sweep in between)
        skips = [k for k in acc if k + 1 < len(ev) and ev[k + 1] in ("qn_accept", "qn_reject")]
        if skips:
            first = skips[0]
            lines.append(f"  first skipped sweep after event {first} (sweep {int(np.sum(ev[:first] == 'sweep'))}, theta "
                         f"{th[first]:.2e}); skipped sweeps: {len(skips)}")
        c = cond[np.isfinite(cond)]
        if len(c):
            lines.append(f"  cond(K) max over blocks: {c.min():.1e} .. {c.max():.1e}")
        rej_last = [k for k in rej if k > (acc[0] if len(acc) else -1)]
        lines.append(f"  rejections after the first acceptance: {len(rej_last)} "
                     f"({', '.join(f'{reason[k] or "?"}@{th[k]:.1e}' for k in rej_last[:8])})")
    lines.append(f"  mu max: {mu[0]:.3g} -> {mu[-1]:.3g}; mu_growth events {int(np.sum(ev == 'mu_growth'))}; "
                 f"dual_fallback {int(np.sum(ev == 'dual_fallback'))}, first_order {int(np.sum(ev == 'first_order'))}")
    # sweeps raising theta
    sw = np.flatnonzero((ev == "sweep") & np.isfinite(th))
    ups = [k for k in sw if k > 0 and np.isfinite(th[k - 1]) and th[k] > th[k - 1]]
    if ups:
        lines.append(f"  sweeps that raised theta: {len(ups)} of {len(sw)} (largest factor "
                     f"{max(th[k] / th[k - 1] for k in ups):.2g})")
    return lines


def main():
    out = []
    files = [("Airliner N = 4", RESULTS / "qn_results_airliner.npz", "N4/"),
             ("Airliner N = 8", RESULTS / "qn_results_airliner.npz", "N8/"),
             ("Ranger", RESULTS / "qn_results_ranger.npz", "")]
    for title, path, prefix in files:
        if not path.exists():
            continue
        d = np.load(path)
        if f"{prefix}names" not in d.files:
            continue
        out.append(f"== {title} ==")
        for name in d[f"{prefix}names"]:
            if str(d[f"{prefix}{name}__solver"]) != "QN":
                continue
            out.append(f"{name}: {d[f'{prefix}{name}__status']}")
            out.extend(analyze(d, prefix, name))
        out.append("")
    text = "\n".join(out)
    print(text)
    (RESULTS / "qn_observations.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
