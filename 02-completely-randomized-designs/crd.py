"""Completely randomized design (CRD): one-way ANOVA, three regression codings, contrasts,
orthogonal polynomial contrasts (STAT 404 ch. 3).

Model
    Y_ij = mu + tau_i + E_ij,  i = 1..t treatments, j = 1..n_i replicates, E ~ N(0, sigma^2).
    Treatments are allocated to the n = sum n_i units completely at random.

ANOVA
    SS(Treatments) = sum_i n_i (ybar_i - ybar)^2       on t - 1 DF
    SS(Residual)   = sum_ij (y_ij - ybar_i)^2          on n - t DF
    F = MS(Trt) / MS(Res) ~ F(t-1, n-t) under H0: tau_1 = ... = tau_t.

Three equivalent regressions (same fitted values, same ANOVA, different betas)
    cell means:   t zero/one dummies, no intercept      beta_i = mu_i
    reference:    intercept + t-1 dummies               beta_0 = mu_1, beta_i = mu_i - mu_1
    contrasts:    intercept + t-1 contrast columns      beta_0 = mean of the mu_i (sum-to-zero)

Contrasts
    kappa = sum c_i mu_i with sum c_i = 0. Estimate sum c_i ybar_i, variance sigma^2 sum c_i^2 / n_i.
    SS(contrast) = (sum c_i ybar_i)^2 / sum(c_i^2 / n_i), 1 DF.
    Two contrasts are orthogonal if sum c_i d_i / n_i = 0; t-1 mutually orthogonal contrasts
    split SS(Treatments) exactly.

Orthogonal polynomial contrasts
    For a quantitative factor at equally spaced levels (equal n), the linear, quadratic, cubic, ...
    contrasts are the orthogonalized columns of the Vandermonde matrix in the levels.

Run:  python 02-completely-randomized-designs/crd.py
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def randomize_crd(treatments, n_rep, rng):
    """Return a random allocation of treatments to n = t * n_rep units (unit order)."""
    labels = np.repeat(np.asarray(treatments), n_rep)
    return rng.permutation(labels)


def one_way_anova(y, g):
    y = np.asarray(y, float)
    levels, gi = np.unique(g, return_inverse=True)
    t, n = len(levels), len(y)
    ni = np.bincount(gi)
    means = np.bincount(gi, y) / ni
    ss_trt = float((ni * (means - y.mean()) ** 2).sum())
    ss_res = float(((y - means[gi]) ** 2).sum())
    ms_trt, ms_res = ss_trt / (t - 1), ss_res / (n - t)
    F = ms_trt / ms_res
    return dict(levels=levels, n=ni, means=means, ss_trt=ss_trt, ss_res=ss_res, df_trt=t - 1,
                df_res=n - t, ms_trt=ms_trt, ms_res=ms_res, F=F, p=stats.f.sf(F, t - 1, n - t))


def design_matrices(g):
    """The three codings of a t-level factor."""
    levels, gi = np.unique(g, return_inverse=True)
    t = len(levels)
    D = np.eye(t)[gi]                                       # cell-means dummies
    ref = np.column_stack([np.ones(len(gi)), D[:, 1:]])     # intercept + t-1 dummies
    S = np.vstack([np.eye(t - 1), -np.ones(t - 1)])         # sum-to-zero contrast coding
    con = np.column_stack([np.ones(len(gi)), S[gi]])
    return D, ref, con


def contrast(an, c, alpha=0.05):
    """Estimate, se, t, p, CI and 1-DF sum of squares for a contrast c of the treatment means."""
    c = np.asarray(c, float)
    assert abs(c.sum()) < 1e-9, "contrast coefficients must sum to zero"
    est = float(c @ an["means"])
    v = float((c ** 2 / an["n"]).sum())
    se = np.sqrt(an["ms_res"] * v)
    t = est / se
    q = stats.t.ppf(1 - alpha / 2, an["df_res"])
    return dict(est=est, se=se, t=t, p=2 * stats.t.sf(abs(t), an["df_res"]),
                ci=(est - q * se, est + q * se), ss=est ** 2 / v)


def orthogonal_poly_contrasts(x, degree=None):
    """Columns are linear, quadratic, ... contrasts for levels x (equal replication)."""
    x = np.asarray(x, float)
    degree = len(x) - 1 if degree is None else degree
    V = np.vander(x - x.mean(), degree + 1, increasing=True)
    Q, _ = np.linalg.qr(V)
    C = Q[:, 1:] * np.sign(Q[-1, 1:])                       # drop the constant; last level positive
    return C                                                # unit-length columns; any rescaling is fine


def demo():
    rng = np.random.default_rng(3)
    # A quantitative treatment at 5 equally spaced levels, response curved in the level
    levels = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    n_rep = 6
    alloc = randomize_crd(levels, n_rep, rng)               # what the experimenter would run
    mu = {1.0: 20.0, 2.0: 23.5, 3.0: 25.0, 4.0: 24.5, 5.0: 22.0}
    y = np.array([mu[a] for a in alloc]) + 1.2 * rng.standard_normal(len(alloc))

    an = one_way_anova(y, alloc)
    print("ANOVA (CRD)       DF      SS      MS       F        p")
    print(f"Treatments       {an['df_trt']:3d} {an['ss_trt']:8.2f} {an['ms_trt']:7.2f} {an['F']:7.2f} {an['p']:9.2g}")
    print(f"Residual         {an['df_res']:3d} {an['ss_res']:8.2f} {an['ms_res']:7.2f}")

    D, ref, con = design_matrices(alloc)
    fits = [np.linalg.lstsq(X, y, rcond=None)[0] for X in (D, ref, con)]
    same = max(np.abs(X @ b - D @ fits[0]).max() for X, b in zip((D, ref, con), fits))
    print(f"three codings give identical fitted values (max diff {same:.1e})")
    print(f"  cell means beta[0] = {fits[0][0]:.2f}, reference beta[0] = {fits[1][0]:.2f}, "
          f"sum-to-zero beta[0] = {fits[2][0]:.2f} (mean of the 5 means)")

    C = orthogonal_poly_contrasts(levels)
    names = ["linear", "quadratic", "cubic", "quartic"]
    total = 0.0
    print("orthogonal polynomial contrasts:")
    for nm, c in zip(names, C.T):
        r = contrast(an, c)
        total += r["ss"]
        print(f"  {nm:9s} SS = {r['ss']:7.2f}  p = {r['p']:.2g}")
    print(f"  sum of the four 1-DF SS = {total:.2f} = SS(Treatments) {an['ss_trt']:.2f}")


if __name__ == "__main__":
    demo()
