"""CUPAC: Control Using Predictions As Covariates (Tang, Huang, Grinstead, DoorDash Engineering 2020).
Reading "CUPA" in the request as CUPAC, CUPED with a machine-learned covariate. Not in the STAT 404 notes.

Idea
    CUPED removes the part of Y that is *linearly* explained by one pre-period metric: variance
    reduction rho^2. If many pre-experiment features W (history, geography, device, tenure, ...) predict Y,
    and the relation is non-linear, a better single covariate is an ML prediction Yhat = f(W).
    CUPAC = CUPED with X := f(W). The variance reduction becomes corr(f(W), Y)^2.

Why it stays unbiased
    Any function of pre-treatment information is independent of the random assignment, so
    E[Xbar_T - Xbar_C] = 0 still holds. Two rules keep it that way:
    1. Features must be pre-treatment (no post-assignment data, nothing the treatment could move).
    2. f must not use the experiment's outcomes in a way that depends on T. Either train f on
       historical data (a period before the experiment, the DoorDash setup), or cross-fit on the experiment
       itself without the treatment label: split users into K folds, predict fold k with a model fitted on
       the other folds (the same idea as double/debiased ML; it is also how "MLRATE" works).

Relation to other methods
    CUPAC is ANCOVA with a learned covariate; with the linear adjustment on top (theta) a bad model cannot make
    things worse than plain DiM asymptotically: theta shrinks toward 0 when f is useless.

Run:  python 12-variance-reduction-ancova-cuped-cupac/cupac.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cuped import cuped_test, ttest_delta  # noqa: E402


def make_users(rng, n):
    """Pre-treatment features W (5 columns incl. a noisy pre-period metric) and the baseline mean mu(W)."""
    w = rng.normal(size=(n, 4))
    mu = (6 + 2.0 * np.sin(2 * w[:, 0]) + 1.5 * w[:, 1] ** 2 + 1.5 * w[:, 2] * w[:, 3]
          + 1.0 * (w[:, 0] > 0.5))
    pre = mu + rng.normal(0, 2.0, n)                         # the CUPED covariate: last period's metric
    return np.column_stack([w, pre]), mu


def outcome(rng, mu, t, tau):
    return mu + tau * t + rng.normal(0, 1.5, len(mu))


def fit_model(W, y):
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=0).fit(W, y)


def crossfit_predictions(W, y, k, rng):
    """Out-of-fold predictions of y from W, ignoring the treatment label."""
    pred = np.empty(len(y))
    for tr, te in KFold(k, shuffle=True, random_state=int(rng.integers(1 << 31))).split(W):
        pred[te] = fit_model(W[tr], y[tr]).predict(W[te])
    return pred


def demo():
    rng = np.random.default_rng(16)
    tau, n = 0.15, 8000

    # historical period: same population, no experiment; the label is that period's outcome
    W_hist, mu_hist = make_users(rng, 20000)
    model = fit_model(W_hist, outcome(rng, mu_hist, np.zeros(len(mu_hist)), 0.0))

    W, mu = make_users(rng, n)
    t = (rng.random(n) < 0.5).astype(float)
    y = outcome(rng, mu, t, tau)
    pre = W[:, -1]
    f_hist = model.predict(W)
    f_cf = crossfit_predictions(W, y, 5, rng)
    print(f"one experiment, n = {n}, true tau = {tau}")
    for name, cov in (("plain t-test", None), ("CUPED (pre-period metric)", pre),
                      ("CUPAC (model trained on history)", f_hist), ("CUPAC (5-fold cross-fit)", f_cf)):
        if cov is None:
            d, se, _ = ttest_delta(y, t)
            r2 = 0.0
        else:
            d, se, _, _ = cuped_test(y, cov, t)
            r2 = np.corrcoef(cov, y)[0, 1] ** 2
        print(f"  {name:33s} delta = {d:+.3f}  se = {se:.4f}  corr^2 = {r2:.2f}")

    # repeated experiments with the historical model: bias, sd, coverage, power
    reps = 1000
    res = {k: [] for k in ("plain", "cuped", "cupac")}
    cover = dict.fromkeys(res, 0)
    power = dict.fromkeys(res, 0)
    for _ in range(reps):
        W, mu = make_users(rng, n)
        t = (rng.random(n) < 0.5).astype(float)
        y = outcome(rng, mu, t, tau)
        for k, (d, se) in {"plain": ttest_delta(y, t)[:2],
                           "cuped": cuped_test(y, W[:, -1], t)[:2],
                           "cupac": cuped_test(y, model.predict(W), t)[:2]}.items():
            res[k].append(d)
            cover[k] += abs(d - tau) <= 1.96 * se
            power[k] += abs(d / se) > 1.96
    print(f"{reps} experiments (historical model):")
    for k in res:
        print(f"  {k:6s} mean {np.mean(res[k]):.4f}  sd {np.std(res[k]):.4f}  "
              f"coverage {cover[k]/reps:.3f}  power {power[k]/reps:.2f}")
    v = {k: np.var(res[k]) for k in res}
    print(f"  variance relative to plain: CUPED {v['cuped']/v['plain']:.2f}, CUPAC {v['cupac']/v['plain']:.2f}")


if __name__ == "__main__":
    demo()
