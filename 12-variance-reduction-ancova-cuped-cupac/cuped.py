"""CUPED: Controlled-experiment Using Pre-Experiment Data (Deng, Xu, Kohavi & Walker, WSDM 2013).
Not in the STAT 404 notes.

Idea
    In an online A/B test the main enemy is variance: metrics are heavy-tailed and effects are ~1%.
    For each user, take a covariate X measured *before* the experiment started (usually the same metric
    in the pre-period). It cannot be affected by the treatment, so it can be used to remove noise.

Estimator (control variates from Monte Carlo)
    theta = Cov(X, Y) / Var(X), estimated on the pooled data (both arms)
    Y_cv  = Y - theta (X - mean(X))
    Delta_cuped = mean(Y_cv | T) - mean(Y_cv | C) = Delta_DiM - theta (Xbar_T - Xbar_C)
    Unbiased for any fixed theta because E[Xbar_T - Xbar_C] = 0 under randomization.
    Var(Y_cv) = Var(Y) (1 - rho^2), rho = corr(X, Y): the variance reduction is rho^2, so the
    sample size needed for the same power falls by the factor (1 - rho^2).

Relation to ANCOVA
    CUPED with a pooled theta is ANCOVA (Y ~ T + X) with the slope estimated ignoring T; the two agree to
    O(1/n). CUPED is popular because it is just a per-user transformation of the metric, so the existing
    t-test pipeline, dashboards and ratio-metric delta methods keep working.

Practical details
    Users with no pre-period data: set X = 0 and add an indicator for "has pre-period data" (or use the
    multi-covariate version: Y - theta^T (X - Xbar) with theta = Sigma_XX^{-1} Sigma_XY).
    Pre-period length: longer windows usually raise rho, with diminishing returns (the paper uses 1-2 weeks).
    Never use data from after the treatment started as X.

Run:  python 12-variance-reduction-ancova-cuped-cupac/cuped.py
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def cuped_theta(y, x):
    xc = x - x.mean()
    return float(xc @ (y - y.mean()) / (xc @ xc))


def cuped_adjust(y, x, theta=None):
    theta = cuped_theta(y, x) if theta is None else theta
    return y - theta * (x - x.mean()), theta


def ttest_delta(y, t):
    """Absolute difference in means, its se (Welch), and the 95% CI."""
    y1, y0 = y[t == 1], y[t == 0]
    d = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return d, se, (d - 1.96 * se, d + 1.96 * se)


def cuped_test(y, x, t):
    y_cv, theta = cuped_adjust(y, x)
    return (*ttest_delta(y_cv, t), theta)


def sample_size_per_arm(sd, mde, alpha=0.05, power=0.8, var_reduction=0.0):
    """Two-sided z-test sample size per arm for an absolute minimum detectable effect."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(2 * (z * sd) ** 2 * (1 - var_reduction) / mde ** 2))


def simulate_users(rng, n, lift=0.02):
    """Sessions-like metric: a user propensity drives both the pre-period X and the in-experiment Y."""
    propensity = rng.lognormal(mean=1.0, sigma=0.8, size=n)
    x = rng.poisson(propensity * 2.0).astype(float)            # 2 pre-period weeks
    t = (rng.random(n) < 0.5).astype(float)
    y = rng.poisson(propensity * 2.0 * (1 + lift * t)).astype(float)
    return y, x, t


def demo():
    rng = np.random.default_rng(15)
    n = 20000
    y, x, t = simulate_users(rng, n)
    rho = np.corrcoef(x, y)[0, 1]
    d, se, ci = ttest_delta(y, t)
    dc, sec, cic, theta = cuped_test(y, x, t)
    print(f"one A/B test, n = {n}: corr(pre, post) = {rho:.2f}, theta = {theta:.2f}")
    print(f"  plain t-test : delta = {d:+.3f}, se = {se:.4f}, 95% CI ({ci[0]:+.3f}, {ci[1]:+.3f})")
    print(f"  CUPED        : delta = {dc:+.3f}, se = {sec:.4f}, 95% CI ({cic[0]:+.3f}, {cic[1]:+.3f})")
    print(f"  variance ratio {(sec/se)**2:.3f}  vs  1 - rho^2 = {1 - rho**2:.3f}")

    # ANCOVA equivalence
    X = np.column_stack([np.ones(n), t, x - x.mean()])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"  ANCOVA coefficient on T = {b[1]:+.4f} (CUPED {dc:+.4f}); slope {b[2]:.4f} (theta {theta:.4f})")

    # many experiments: unbiasedness, coverage, power
    reps = 1000
    truth = 2.0 * np.exp(1.0 + 0.8 ** 2 / 2) * 0.02              # E[Y_T] - E[Y_C] in simulate_users
    est_p, est_c, pow_p, pow_c = [], [], 0, 0
    for _ in range(reps):
        y, x, t = simulate_users(rng, n)
        d, se, _ = ttest_delta(y, t)
        dc, sec, _, _ = cuped_test(y, x, t)
        est_p.append(d)
        est_c.append(dc)
        pow_p += abs(d / se) > 1.96
        pow_c += abs(dc / sec) > 1.96
    print(f"1000 experiments (true delta {truth:.4f}): mean delta plain {np.mean(est_p):.4f}, CUPED {np.mean(est_c):.4f}; "
          f"sd plain {np.std(est_p):.4f}, CUPED {np.std(est_c):.4f}")
    print(f"  power at a 2% lift: plain t-test {pow_p/reps:.2f}, CUPED {pow_c/reps:.2f}")
    sd = y.std()
    print(f"  users per arm for 80% power at this effect: plain {sample_size_per_arm(sd, truth):,}, "
          f"CUPED {sample_size_per_arm(sd, truth, var_reduction=rho**2):,}")


if __name__ == "__main__":
    demo()
