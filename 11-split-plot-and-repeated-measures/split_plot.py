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

import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01-linear-models"))
from robust_se import cluster_cov, ols_resid  # noqa: E402


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


def regression_w_contrast(Y, cluster_robust=True):
    """Regression at the sub-plot level: y ~ block + W + S, coefficient for W level 2 vs level 1.
    With cluster_robust, the SE is CR2 clustered on whole plots and the t test uses G - 1 DF;
    otherwise the classic OLS SE (which treats every sub-plot as independent)."""
    b, w, s = Y.shape
    blk, wl, sl = np.meshgrid(np.arange(b), np.arange(w), np.arange(s), indexing="ij")
    blk, wl, sl, y = blk.ravel(), wl.ravel(), sl.ravel(), Y.ravel()
    X = np.column_stack([np.ones_like(y)] + [(blk == k) * 1.0 for k in range(1, b)]
                        + [(wl == k) * 1.0 for k in range(1, w)] + [(sl == k) * 1.0 for k in range(1, s)])
    j = b                                                       # first W dummy
    beta, e = ols_resid(X, y)
    if cluster_robust:
        V, G = cluster_cov(X, e, blk * w + wl, "CR2")       # cluster = whole plot
        df = G - 1
    else:
        n, k = X.shape
        V = np.linalg.inv(X.T @ X) * (e @ e) / (n - k)
        df = n - k
    t = beta[j] / np.sqrt(V[j, j])
    return float(beta[j]), float(np.sqrt(V[j, j])), float(2 * stats.t.sf(abs(t), df))


def whole_plot_t(Y):
    """The split-plot answer for the same contrast: SE from MS(whole-plot error) on (b-1)(w-1) DF."""
    b, w, s = Y.shape
    an = split_plot_anova(Y)
    m = Y.mean((0, 2))
    se = np.sqrt(2 * an["WPerr"]["ms"] / (b * s))
    t = (m[1] - m[0]) / se
    return float(m[1] - m[0]), float(se), float(2 * stats.t.sf(abs(t), an["WPerr"]["df"]))


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

    # the regression view: one W contrast, classic OLS SE vs cluster-robust SE clustered on whole plots
    rej = {"split-plot t (WP error)": 0, "OLS, classic SE": 0, "OLS, CR2 SE by whole plot": 0}
    for _ in range(reps):
        Y0 = simulate(rng, b, w, s, alpha=[0, 0, 0], gamma=[0, 0.5, 1.0, 1.5], sigma_w=1.5, sigma=0.5)
        rej["split-plot t (WP error)"] += whole_plot_t(Y0)[2] < 0.05
        rej["OLS, classic SE"] += regression_w_contrast(Y0, cluster_robust=False)[2] < 0.05
        rej["OLS, CR2 SE by whole plot"] += regression_w_contrast(Y0, cluster_robust=True)[2] < 0.05
    print("type I error for the W2 - W1 contrast: " + ", ".join(f"{k} {v/reps:.3f}" for k, v in rej.items()))


if __name__ == "__main__":
    demo()
