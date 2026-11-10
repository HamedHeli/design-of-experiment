"""Cluster-robust standard errors in A/B tests: where they are needed, and how they combine with CUPED.

Case 1: cluster-randomized experiment
    Treatment is assigned to whole clusters (cities, stores, schools, ad campaigns) because of interference
    or logistics, but the outcome is measured per user. Users in the same city share shocks, so the
    user-level t-test treats n users as independent when there are really only G clusters. Fixes:
    cluster-robust SE clustered on the randomization unit (CR2, t with G - 1 DF), or analyse cluster means.
    CUPED still works: adjust each user's outcome with their pre-period metric, then cluster the SE.

Case 2: user-randomized, analysed per session / page view
    Randomization is by user, but a metric like "clicks per session" is computed over sessions, and a user
    contributes many correlated sessions. The analysis unit (session) is finer than the randomization unit
    (user), so session-level SEs are too small. Clustering by user gives the right SE; for a ratio of means it
    reproduces the delta-method SE that experimentation platforms use (Deng, Knoblich & Lu 2018).

Rule: cluster at the level of randomization (or coarser if errors are correlated at a coarser level).
When analysis unit = randomization unit (plain user-level A/B test with one row per user), there is nothing to
cluster and the ordinary Welch / HC SE is correct.

Run:  python 12-variance-reduction-ancova-cuped-cupac/cluster_randomized.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01-linear-models"))
from robust_se import classic_cov, cluster_cov, ols_resid  # noqa: E402


def simulate_cities(rng, G=30, mean_users=60, tau=0.0, icc=0.05):
    sizes = rng.poisson(mean_users, G) + 5
    city = np.repeat(np.arange(G), sizes)
    treated_city = (rng.permutation(G) < G // 2).astype(float)
    t = treated_city[city]
    city_shock = rng.normal(0, np.sqrt(icc), G)[city]
    user = rng.normal(0, np.sqrt(1 - icc), len(city))
    pre = city_shock + user + rng.normal(0, 0.5, len(city))      # pre-period metric, same user and city
    y = tau * t + city_shock + user + rng.normal(0, 0.5, len(city))
    return y, pre, t, city


def analyse_cities(y, pre, t, city, cuped=False):
    """Returns dict of (estimate, se, p) for naive user-level, CR2-by-city, and city-means analyses."""
    if cuped:
        xc = pre - pre.mean()
        y = y - (xc @ (y - y.mean()) / (xc @ xc)) * xc
    X = np.column_stack([np.ones_like(y), t])
    beta, e = ols_resid(X, y)
    out = {}
    se = np.sqrt(classic_cov(X, e)[1, 1])
    out["naive"] = (beta[1], se, 2 * stats.t.sf(abs(beta[1] / se), len(y) - 2))
    V, G = cluster_cov(X, e, city, "CR2")
    se = np.sqrt(V[1, 1])
    out["CR2"] = (beta[1], se, 2 * stats.t.sf(abs(beta[1] / se), G - 1))
    cm = np.bincount(city, y) / np.bincount(city)
    ct = np.bincount(city, t) / np.bincount(city)
    r = stats.ttest_ind(cm[ct == 1], cm[ct == 0])
    out["city means"] = (cm[ct == 1].mean() - cm[ct == 0].mean(), np.nan, r.pvalue)
    return out


def simulate_sessions(rng, n_users=2000, tau=0.0):
    t = (rng.random(n_users) < 0.5).astype(float)
    k = rng.poisson(4, n_users) + 1                              # sessions per user
    uid = np.repeat(np.arange(n_users), k)
    user_rate = rng.beta(2, 8, n_users)                          # each user's own click propensity
    p = np.clip(user_rate[uid] * (1 + tau * t[uid]), 0, 1)
    clicks = rng.binomial(3, p).astype(float)                    # clicks in a session
    return clicks, t[uid], uid


def delta_method_ratio_se(clicks, uid, t):
    """SE of (clicks per session)_T - (clicks per session)_C, treating users as the independent units."""
    out = []
    for arm in (1.0, 0.0):
        users = np.unique(uid[t == arm])
        num = np.bincount(uid, clicks)[users]
        den = np.bincount(uid)[users].astype(float)
        n = len(users)
        mx, my = num.mean(), den.mean()
        vx, vy, cxy = num.var(ddof=1), den.var(ddof=1), np.cov(num, den)[0, 1]
        r = mx / my
        out.append((r, (vx / my ** 2 - 2 * r * cxy / my ** 2 + r ** 2 * vy / my ** 2) / n))
    return out[0][0] - out[1][0], np.sqrt(out[0][1] + out[1][1])


def demo():
    rng = np.random.default_rng(25)
    reps = 1000

    print("Case 1: 30 cities randomized, ~65 users each, ICC 0.05, no true effect")
    for cuped in (False, True):
        rej = {"naive": 0, "CR2": 0, "city means": 0}
        sd_est, se_cr = [], []
        for _ in range(reps):
            res = analyse_cities(*simulate_cities(rng), cuped=cuped)
            for k in rej:
                rej[k] += res[k][2] < 0.05
            sd_est.append(res["CR2"][0])
            se_cr.append(res["CR2"][1])
        lab = "with CUPED   " if cuped else "without CUPED"
        print(f"  {lab}: type I error naive {rej['naive']/reps:.3f}, CR2 by city {rej['CR2']/reps:.3f}, "
              f"city-means t {rej['city means']/reps:.3f};  sd of estimate {np.std(sd_est):.4f}, "
              f"mean CR2 SE {np.mean(se_cr):.4f}")

    print("Case 2: 2000 users randomized, clicks per session analysed over ~10000 sessions, no true effect")
    rej = {"session-level naive": 0, "CR1 by user": 0, "delta method": 0}
    est, se_n, se_c, se_d = [], [], [], []
    for _ in range(reps // 2):
        clicks, t, uid = simulate_sessions(rng)
        X = np.column_stack([np.ones_like(clicks), t])
        beta, e = ols_resid(X, clicks)
        sn = np.sqrt(classic_cov(X, e)[1, 1])
        V, G = cluster_cov(X, e, uid, "CR1")
        sc = np.sqrt(V[1, 1])
        d, sd = delta_method_ratio_se(clicks, uid, t)
        est.append(beta[1])
        se_n.append(sn)
        se_c.append(sc)
        se_d.append(sd)
        rej["session-level naive"] += abs(beta[1] / sn) > 1.96
        rej["CR1 by user"] += abs(beta[1] / sc) > 1.96
        rej["delta method"] += abs(d / sd) > 1.96
    n = reps // 2
    print(f"  sd of estimate {np.std(est):.4f}; mean SE: naive {np.mean(se_n):.4f}, CR1 by user "
          f"{np.mean(se_c):.4f}, delta method {np.mean(se_d):.4f}")
    print("  type I error: " + ", ".join(f"{k} {v/n:.3f}" for k, v in rej.items()))


if __name__ == "__main__":
    demo()
