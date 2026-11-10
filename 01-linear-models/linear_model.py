"""The linear model, least squares, ANOVA, t and F tests, prediction, Box-Cox (STAT 404 ch. 2).

Linear model
    Y = X beta + E,  E ~ N(0, sigma^2 I),  X is n x k with full column rank.
    Polynomial (x^2) and interaction (x1 x2) terms are just more columns of X:
    the model is linear in beta, not necessarily in x.

Least squares
    beta_hat = (X^T X)^{-1} X^T y,  fitted y_hat = X beta_hat = H y,  residuals r = y - y_hat.
    s^2 = ||r||^2 / (n - k) is unbiased for sigma^2.

ANOVA (corrected for the intercept)
    SS(Total) = sum (y_i - ybar)^2 = SS(Regression) + SS(Residual),
    F = [SS(Reg)/(k-1)] / [SS(Res)/(n-k)] ~ F(k-1, n-k) if all slopes are 0.

Inference on one coefficient
    se(beta_j) = s sqrt([(X^T X)^{-1}]_jj),  t = beta_j / se ~ t(n-k).

F test of some parameters (extra sum of squares)
    F = [(SS_res(reduced) - SS_res(full)) / q] / [SS_res(full)/(n-k)] ~ F(q, n-k).

Prediction at x0
    mean:      x0^T beta_hat +/- t s sqrt(x0^T (X^T X)^{-1} x0)
    new Y:     x0^T beta_hat +/- t s sqrt(1 + x0^T (X^T X)^{-1} x0)

Box-Cox
    y^(lambda) = (y^lambda - 1) / (lambda gdot^(lambda-1)), log transform at lambda = 0,
    gdot = geometric mean of y. Choose lambda to minimize SS_res of the transformed fit
    (equivalently maximize the profile log-likelihood -n/2 log SS_res).

Run:  python 01-linear-models/linear_model.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class OLSFit:
    beta: np.ndarray
    se: np.ndarray
    t: np.ndarray
    p: np.ndarray
    s2: float
    df_resid: int
    ss_resid: float
    XtX_inv: np.ndarray
    fitted: np.ndarray
    resid: np.ndarray


def fit_ols(X, y) -> OLSFit:
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    Q, R = np.linalg.qr(X)                     # never form (X^T X)^{-1} to solve
    beta = np.linalg.solve(R, Q.T @ y)
    fitted = X @ beta
    resid = y - fitted
    df = n - k
    ss = float(resid @ resid)
    s2 = ss / df
    Rinv = np.linalg.inv(R)
    XtX_inv = Rinv @ Rinv.T
    se = np.sqrt(s2 * np.diag(XtX_inv))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df)
    return OLSFit(beta, se, t, p, s2, df, ss, XtX_inv, fitted, resid)


def anova_table(X, y):
    """ANOVA corrected for the intercept. X must contain a column of ones."""
    fit = fit_ols(X, y)
    n, k = X.shape
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_reg = ss_tot - fit.ss_resid
    F = (ss_reg / (k - 1)) / fit.s2
    return {
        "Regression": dict(df=k - 1, ss=ss_reg, ms=ss_reg / (k - 1), F=F, p=stats.f.sf(F, k - 1, n - k)),
        "Residual": dict(df=n - k, ss=fit.ss_resid, ms=fit.s2),
        "Total": dict(df=n - 1, ss=ss_tot),
        "R2": ss_reg / ss_tot,
    }


def extra_ss_test(X_full, X_reduced, y):
    """F test that the columns in X_full but not X_reduced have zero coefficients."""
    full, red = fit_ols(X_full, y), fit_ols(X_reduced, y)
    q = X_full.shape[1] - X_reduced.shape[1]
    F = ((red.ss_resid - full.ss_resid) / q) / full.s2
    return F, stats.f.sf(F, q, full.df_resid), q


def predict(fit: OLSFit, x0, level=0.95):
    """Return (estimate, CI for the mean, PI for a new observation)."""
    x0 = np.asarray(x0, float)
    est = float(x0 @ fit.beta)
    h = float(x0 @ fit.XtX_inv @ x0)
    tq = stats.t.ppf(0.5 + level / 2, fit.df_resid)
    s = np.sqrt(fit.s2)
    ci = (est - tq * s * np.sqrt(h), est + tq * s * np.sqrt(h))
    pi = (est - tq * s * np.sqrt(1 + h), est + tq * s * np.sqrt(1 + h))
    return est, ci, pi


def boxcox_transform(y, lam, gdot=None):
    y = np.asarray(y, float)
    g = np.exp(np.log(y).mean()) if gdot is None else gdot
    if abs(lam) < 1e-12:
        return g * np.log(y)
    return (y ** lam - 1) / (lam * g ** (lam - 1))


def boxcox_profile(X, y, lambdas):
    """SS_res of the scaled Box-Cox transform for each lambda (smaller is better),
    plus an approximate 95% CI for lambda from the profile log-likelihood."""
    lambdas = np.asarray(lambdas, float)
    ss = np.array([fit_ols(X, boxcox_transform(y, lam)).ss_resid for lam in lambdas])
    n = len(y)
    loglik = -n / 2 * np.log(ss)
    best = lambdas[np.argmax(loglik)]
    inside = lambdas[loglik >= loglik.max() - stats.chi2.ppf(0.95, 1) / 2]
    return ss, best, (inside.min(), inside.max())


def demo():
    rng = np.random.default_rng(1)
    n = 40
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    y = 10 + 2 * x1 - 1 * x2 + 1.5 * x1 * x2 + 0.5 * rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x1, x2, x1 * x2, x1 ** 2])
    fit = fit_ols(X, y)
    names = ["1", "x1", "x2", "x1*x2", "x1^2"]
    print("coef      estimate     se        t        p")
    for nm, b, se, t, p in zip(names, fit.beta, fit.se, fit.t, fit.p):
        print(f"{nm:7s} {b:9.3f} {se:8.3f} {t:8.2f} {p:9.2g}")
    tab = anova_table(X, y)
    print(f"overall F = {tab['Regression']['F']:.1f} on {tab['Regression']['df']}, "
          f"{tab['Residual']['df']} DF,  R^2 = {tab['R2']:.3f}")
    F, p, q = extra_ss_test(X, X[:, :3], y)
    print(f"extra-SS F test of x1*x2 and x1^2 together: F = {F:.1f} on {q} DF, p = {p:.2g}")
    est, ci, pi = predict(fit, [1, 0.5, 0.5, 0.25, 0.25])
    print(f"at x1 = x2 = 0.5: mean {est:.2f}, 95% CI ({ci[0]:.2f}, {ci[1]:.2f}), "
          f"95% PI ({pi[0]:.2f}, {pi[1]:.2f})")

    # Box-Cox: the truth is linear on the log scale with multiplicative noise
    x = np.repeat(np.linspace(1, 5, 5), 8)
    ylog = np.exp(0.2 + 0.4 * x + 0.25 * rng.standard_normal(len(x)))
    Xg = np.column_stack([np.ones_like(x)] + [(x == v).astype(float) for v in np.unique(x)[1:]])
    lams = np.linspace(-1.5, 2, 141)
    _, best, (lo, hi) = boxcox_profile(Xg, ylog, lams)
    print(f"Box-Cox on log-normal data: lambda_hat = {best:.2f}, 95% CI ({lo:.2f}, {hi:.2f})  (truth: 0, the log)")


if __name__ == "__main__":
    demo()
