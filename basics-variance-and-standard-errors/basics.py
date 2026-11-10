"""The basics every design rests on: variance, standard deviation, standard error, the variance of a
difference, pooled vs Welch standard errors, paired differences, contrasts, proportions, and the
standard error of a percentage lift.

Variance and standard deviation (describe the spread of individual observations)
    Var(Y) = sigma^2 = E[(Y - mu)^2],  SD = sigma (same units as Y).
    Sample: s^2 = sum (y_i - ybar)^2 / (n - 1). Dividing by n - 1 makes s^2 unbiased, because the n
    deviations from ybar sum to 0, so only n - 1 of them are free (degrees of freedom).

Standard error (describes the spread of an *estimate* across repeated experiments)
    SE(ybar) = sigma / sqrt(n), estimated by s / sqrt(n). More data shrinks the SE, never the SD.

Rules for variances (a, b constants)
    Var(aX + b) = a^2 Var(X)
    Var(X + Y) = Var(X) + Var(Y) + 2 Cov(X, Y)
    Var(X - Y) = Var(X) + Var(Y) - 2 Cov(X, Y)        <- the minus sign does NOT carry to the variances
    independent X, Y:  Var(X - Y) = Var(X) + Var(Y)
    Var(sum c_i ybar_i) = sum c_i^2 sigma_i^2 / n_i  for independent groups (any contrast)

Difference of two independent means
    Var(ybar_1 - ybar_2) = sigma_1^2 / n_1 + sigma_2^2 / n_2
    Welch (unpooled) SE:  sqrt(s_1^2/n_1 + s_2^2/n_2), Welch-Satterthwaite DF
    Pooled (assumes sigma_1 = sigma_2 = sigma):
        s_p^2 = [(n_1 - 1) s_1^2 + (n_2 - 1) s_2^2] / (n_1 + n_2 - 2)   (DF-weighted average of the s^2)
        SE = s_p sqrt(1/n_1 + 1/n_2),  t on n_1 + n_2 - 2 DF
    With t groups, s_p^2 is exactly MS(Residual) of the one-way ANOVA.

Paired differences
    d_i = y_i,B - y_i,A on the same unit: Var(dbar) = (sigma_A^2 + sigma_B^2 - 2 Cov) / m.
    Positive within-pair correlation is what makes pairing (blocking) pay off.

Proportions (Bernoulli outcomes, e.g. conversion)
    Var(Y) = p(1 - p), SE(phat) = sqrt(p(1 - p)/n).
    Difference: SE = sqrt(p1(1-p1)/n1 + p2(1-p2)/n2) for the CI; under H0 p1 = p2 the z test uses the
    pooled phat = (x1 + x2)/(n1 + n2): SE0 = sqrt(phat(1 - phat)(1/n1 + 1/n2)).

Clustered observations (design effect)
    n observations in clusters of size m with intra-cluster correlation rho: the variance of a mean (or of a
    cluster-level treatment effect) is inflated by 1 + (m - 1) rho, so the effective sample size is
    n / (1 + (m - 1) rho). Fix the SE with cluster-robust standard errors (01-linear-models/robust_se.py).

Relative (percentage) lift, delta method
    L = ybar_T / ybar_C - 1.  Var(L) ~ (1/mu_C^2) [Var(ybar_T) + (mu_T/mu_C)^2 Var(ybar_C)].

Run:  python basics-variance-and-standard-errors/basics.py
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def sample_var(y, ddof=1):
    y = np.asarray(y, float)
    return ((y - y.mean()) ** 2).sum() / (len(y) - ddof)


def se_mean(y):
    return np.sqrt(sample_var(y) / len(y))


def pooled_two_sample(y1, y2):
    """Estimate, pooled SD, SE, t, DF, two-sided p."""
    n1, n2 = len(y1), len(y2)
    sp2 = ((n1 - 1) * sample_var(y1) + (n2 - 1) * sample_var(y2)) / (n1 + n2 - 2)
    se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
    d = np.mean(y1) - np.mean(y2)
    df = n1 + n2 - 2
    t = d / se
    return dict(diff=d, sp=np.sqrt(sp2), se=se, t=t, df=df, p=2 * stats.t.sf(abs(t), df))


def welch_two_sample(y1, y2):
    n1, n2 = len(y1), len(y2)
    v1, v2 = sample_var(y1) / n1, sample_var(y2) / n2
    se = np.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    d = np.mean(y1) - np.mean(y2)
    t = d / se
    return dict(diff=d, se=se, t=t, df=df, p=2 * stats.t.sf(abs(t), df))


def paired(yA, yB):
    d = np.asarray(yB, float) - np.asarray(yA, float)
    se = se_mean(d)
    t = d.mean() / se
    return dict(diff=d.mean(), se=se, t=t, df=len(d) - 1, p=2 * stats.t.sf(abs(t), len(d) - 1))


def contrast_se(means_n_s2, c, pooled=True):
    """SE of sum c_i ybar_i from (n_i, s_i^2) per group; pooled uses the DF-weighted s^2."""
    n = np.array([g[0] for g in means_n_s2], float)
    s2 = np.array([g[1] for g in means_n_s2], float)
    c = np.asarray(c, float)
    if pooled:
        sp2 = ((n - 1) * s2).sum() / (n - 1).sum()
        return np.sqrt(sp2 * (c ** 2 / n).sum())
    return np.sqrt((c ** 2 * s2 / n).sum())


def two_proportions(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    se_ci = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    p = (x1 + x2) / (n1 + n2)
    se0 = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se0
    return dict(diff=p1 - p2, se_ci=se_ci, se_test=se0, z=z, p=2 * stats.norm.sf(abs(z)),
                ci=(p1 - p2 - 1.96 * se_ci, p1 - p2 + 1.96 * se_ci))


def relative_lift(yT, yC):
    """Percentage lift ybar_T / ybar_C - 1 and its delta-method SE."""
    mT, mC = np.mean(yT), np.mean(yC)
    vT, vC = sample_var(yT) / len(yT), sample_var(yC) / len(yC)
    lift = mT / mC - 1
    se = np.sqrt(vT / mC ** 2 + mT ** 2 * vC / mC ** 4)
    return lift, se


def design_effect(m, rho):
    return 1 + (m - 1) * rho


def demo():
    rng = np.random.default_rng(20)
    sigma, n = 2.0, 25

    # 1. SD vs SE, and why n - 1
    means, s2_n1, s2_n = [], [], []
    for _ in range(20000):
        y = rng.normal(10, sigma, n)
        means.append(y.mean())
        s2_n1.append(sample_var(y, 1))
        s2_n.append(sample_var(y, 0))
    print(f"n = {n}, sigma = {sigma}:")
    print(f"  SD of one observation {sigma:.2f};  SD of ybar over 20000 samples {np.std(means):.3f} "
          f"= SE = sigma/sqrt(n) = {sigma/np.sqrt(n):.3f}")
    print(f"  mean of s^2 with n-1: {np.mean(s2_n1):.3f}, with n: {np.mean(s2_n):.3f}  (sigma^2 = 4, "
          f"(n-1)/n sigma^2 = {4*(n-1)/n:.3f})")
    for m in (25, 100, 400):
        y = rng.normal(10, sigma, m)
        print(f"  one sample of n = {m:3d}: SD = {np.std(y, ddof=1):.2f}, SE = {se_mean(y):.3f}")

    # 2. variance of a difference: variances add
    x = rng.normal(0, 3, 200000)
    yv = rng.normal(0, 4, 200000)
    print(f"Var(X - Y), independent, sd 3 and 4: {np.var(x - yv):.2f} (= 9 + 16 = 25, SD 5, not 4 - 3)")
    z = 0.8 * (x / 3) * 4 + np.sqrt(1 - 0.64) * rng.normal(0, 4, 200000)       # corr 0.8 with x
    print(f"Var(X - Y), corr 0.8: {np.var(x - z):.2f} (= 9 + 16 - 2*0.8*3*4 = {25 - 2*0.8*12:.1f})")

    # 3. pooled vs Welch
    y1, y2 = rng.normal(11, 2, 15), rng.normal(10, 2, 15)
    pw, ww = pooled_two_sample(y1, y2), welch_two_sample(y1, y2)
    print(f"equal SDs, 15 + 15: pooled SE {pw['se']:.3f} ({pw['df']} DF, p = {pw['p']:.3f}); "
          f"Welch SE {ww['se']:.3f} ({ww['df']:.1f} DF, p = {ww['p']:.3f})")
    # type I error when SDs and sizes differ: small group has the big SD
    rej_p = rej_w = 0
    for _ in range(10000):
        a, b = rng.normal(0, 4, 8), rng.normal(0, 1, 40)
        rej_p += pooled_two_sample(a, b)["p"] < 0.05
        rej_w += welch_two_sample(a, b)["p"] < 0.05
    print(f"no true difference, n = 8 (SD 4) vs 40 (SD 1): type I error pooled {rej_p/1e4:.3f}, "
          f"Welch {rej_w/1e4:.3f}")

    # 4. paired vs unpaired on the same data
    unit = rng.normal(50, 5, 12)
    yA = unit + rng.normal(0, 1, 12)
    yB = unit + 1.5 + rng.normal(0, 1, 12)
    pr, up = paired(yA, yB), welch_two_sample(yB, yA)
    print(f"12 units measured twice, effect 1.5: estimate {pr['diff']:.2f}; paired SE {pr['se']:.2f} (p = {pr['p']:.4f}); "
          f"treated as independent SE {up['se']:.2f} (p = {up['p']:.2f})")

    # 5. contrast SE
    groups = [(6, 1.2), (6, 0.9), (6, 1.4), (6, 1.1)]
    c = [1, 1, -1, -1]
    print(f"contrast (1, 1, -1, -1) with n = 6 per group: SE = {contrast_se(groups, c):.3f} "
          f"= s_p sqrt(4/6)")

    # 6. clustered data: the naive SE of a mean ignores the design effect
    G, mm, rho = 40, 25, 0.1
    means, naive = [], []
    for _ in range(5000):
        y = (rng.normal(0, np.sqrt(rho), G)[:, None] + rng.normal(0, np.sqrt(1 - rho), (G, mm))).ravel()
        means.append(y.mean())
        naive.append(se_mean(y))
    print(f"{G} clusters of {mm}, ICC {rho}: true SE of the mean {np.std(means):.4f}, naive s/sqrt(n) "
          f"{np.mean(naive):.4f}; ratio^2 {(np.std(means)/np.mean(naive))**2:.2f} vs design effect "
          f"{design_effect(mm, rho):.2f}")

    # 7. proportions and lift
    pr2 = two_proportions(560, 10000, 500, 10000)
    print(f"conversion 5.6% vs 5.0%, 10000 each: diff {pr2['diff']:.4f}, SE(CI) {pr2['se_ci']:.4f}, "
          f"SE(test, pooled p) {pr2['se_test']:.4f}, z = {pr2['z']:.2f}, p = {pr2['p']:.3f}")
    yT, yC = rng.exponential(10.4, 5000), rng.exponential(10.0, 5000)
    lift, se = relative_lift(yT, yC)
    boot = [relative_lift(rng.choice(yT, 5000), rng.choice(yC, 5000))[0] for _ in range(2000)]
    print(f"revenue lift {100*lift:.1f}%: delta-method SE {100*se:.2f} pts, bootstrap SE {100*np.std(boot):.2f} pts")


if __name__ == "__main__":
    demo()
