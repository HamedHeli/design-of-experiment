"""Split-plot and repeated-measures experiments: two sizes of experimental unit, two error terms
(STAT 404 ch. 12).

Design
    b blocks; in each block the w levels of the whole-plot factor W are randomized to w whole plots;
    each whole plot is split into s sub-plots and the s levels of the sub-plot factor S are randomized
    to them. W is hard to change (oven temperature, irrigation, a field treatment); S is easy.

Model
    Y_ijk = mu + beta_k + alpha_i + eta_ik + gamma_j + (alpha gamma)_ij + E_ijk
        block beta_k, whole-plot error eta_ik ~ N(0, sigma_W^2), sub-plot error E ~ N(0, sigma^2).

ANOVA: two strata
    Whole-plot stratum: Blocks (b-1), W (w-1), whole-plot error = Block x W ((b-1)(w-1))
        E[MS(WP error)] = sigma^2 + s sigma_W^2;   F_W = MS(W) / MS(WP error)
    Sub-plot stratum:  S (s-1), W x S ((w-1)(s-1)), sub-plot error (w(b-1)(s-1))
        E[MS(SP error)] = sigma^2;                 F_S = MS(S) / MS(SP error), same for W x S

Consequences
    S and W x S are compared within whole plots, so they are estimated more precisely than W.
    Analysing a split plot as if it were a fully randomized factorial uses the small sub-plot error for W,
    which inflates the type I error for W, often badly.

Repeated measures
    A subject is a whole plot (the between-subject treatment is W) and the repeated times are sub-plots.
    The split-plot analysis then assumes compound symmetry (sphericity) across times; with
    autocorrelated measurements use a Greenhouse-Geisser correction or a mixed model.

Run:  python 11-split-plot-and-repeated-measures/split_plot.py
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def simulate(rng, b, w, s, alpha, gamma, sigma_w, sigma, inter=None):
    """Returns Y with shape (b, w, s)."""
    alpha = np.asarray(alpha, float)
    gamma = np.asarray(gamma, float)
    inter = np.zeros((w, s)) if inter is None else inter
    Y = (rng.normal(0, 1, b)[:, None, None] + alpha[None, :, None] + gamma[None, None, :]
         + inter[None] + rng.normal(0, sigma_w, (b, w))[..., None]
         + rng.normal(0, sigma, (b, w, s)))
    return 10 + Y


def split_plot_anova(Y):
    Y = np.asarray(Y, float)
    b, w, s = Y.shape
    g = Y.mean()
    mB = Y.mean((1, 2)) - g
    mW = Y.mean((0, 2)) - g
    mS = Y.mean((0, 1)) - g
    mBW = Y.mean(2) - g - mB[:, None] - mW[None, :]
    mWS = Y.mean(0) - g - mW[:, None] - mS[None, :]
    ss = dict(B=w * s * (mB ** 2).sum(), W=b * s * (mW ** 2).sum(), WPerr=s * (mBW ** 2).sum(),
              S=b * w * (mS ** 2).sum(), WS=b * (mWS ** 2).sum())
    ss["SPerr"] = ((Y - g) ** 2).sum() - sum(ss.values())
    df = dict(B=b - 1, W=w - 1, WPerr=(b - 1) * (w - 1), S=s - 1, WS=(w - 1) * (s - 1),
              SPerr=w * (b - 1) * (s - 1))
    ms = {k: ss[k] / df[k] for k in ss}
    out = {k: dict(df=df[k], ss=ss[k], ms=ms[k]) for k in ss}
    for k, err in (("W", "WPerr"), ("S", "SPerr"), ("WS", "SPerr")):
        F = ms[k] / ms[err]
        out[k].update(F=F, p=stats.f.sf(F, df[k], df[err]), err=err)
    return out


def naive_anova_p_W(Y):
    """Wrong analysis: blocks + W*S factorial with one pooled residual."""
    an = split_plot_anova(Y)
    ss_res = an["WPerr"]["ss"] + an["SPerr"]["ss"]
    df_res = an["WPerr"]["df"] + an["SPerr"]["df"]
    F = an["W"]["ms"] / (ss_res / df_res)
    return stats.f.sf(F, an["W"]["df"], df_res)


def demo():
    rng = np.random.default_rng(13)
    b, w, s = 4, 3, 4
    Y = simulate(rng, b, w, s, alpha=[0, 1.0, 2.0], gamma=[0, 0.5, 1.0, 1.5], sigma_w=1.5, sigma=0.5)
    an = split_plot_anova(Y)
    print(f"split plot: {b} blocks x {w} whole-plot levels x {s} sub-plot levels")
    print("source        DF      SS      MS       F        p   (tested against)")
    for k in ("B", "W", "WPerr", "S", "WS", "SPerr"):
        r = an[k]
        extra = f"{r['F']:8.2f} {r['p']:9.2g}   {r['err']}" if "F" in r else ""
        print(f"{k:10s} {r['df']:4d} {r['ss']:8.2f} {r['ms']:7.2f} {extra}")

    # type I error for W under H0 (no W effect), correct vs naive analysis
    rej_ok = rej_naive = 0
    reps = 4000
    for _ in range(reps):
        Y0 = simulate(rng, b, w, s, alpha=[0, 0, 0], gamma=[0, 0.5, 1.0, 1.5], sigma_w=1.5, sigma=0.5)
        rej_ok += split_plot_anova(Y0)["W"]["p"] < 0.05
        rej_naive += naive_anova_p_W(Y0) < 0.05
    print(f"type I error for W at 5% (no W effect): split-plot analysis {rej_ok/reps:.3f}, "
          f"naive single-error analysis {rej_naive/reps:.3f}")


if __name__ == "__main__":
    demo()
