"""Heteroskedasticity-robust (HC) and cluster-robust (CR) "sandwich" standard errors for least squares.
Not in the STAT 404 notes, which assume independent, equal-variance errors throughout.

Why
    The OLS formula Var(beta_hat) = sigma^2 (X^T X)^{-1} assumes Var(E) = sigma^2 I. If observations are
    grouped into clusters (users with many sessions, pupils in schools, sub-plots in a whole plot, users in a
    city that was randomized as a whole) the errors within a cluster are correlated, and that formula can be
    badly too small, most of all for regressors that are constant within a cluster (a cluster-level treatment).

Sandwich estimator
    Var(beta_hat) = (X^T X)^{-1}  [ X^T Omega X ]  (X^T X)^{-1}        (bread, meat, bread)
    HC (independent, unequal variances):  meat = sum_i  e_i^2 x_i x_i^T
        HC1 multiplies by n/(n-k); HC2 uses e_i^2 / (1 - h_ii).
    CR (any correlation inside a cluster g, independence across clusters):
        meat = sum_g  X_g^T e_g e_g^T X_g
        CR1 (Stata / statsmodels default): times G/(G-1) * (n-1)/(n-k)
        CR2 (Bell & McCaffrey 2002): e_g replaced by (I - H_gg)^{-1/2} e_g, less biased with few clusters.
    Use t with G - 1 degrees of freedom: with G clusters there are only G independent pieces of information.

Rules of thumb
    Cluster at the level of randomization (Abadie, Athey, Imbens & Wooldridge 2023), or at any coarser level
    at which errors are correlated. Clustering is about the design, not a free robustness add-on.
    With fewer than about 30-50 clusters CR1 is anti-conservative; prefer CR2 with t_{G-1}, the wild cluster
    bootstrap, or the analysis of cluster means.
    Design effect for a cluster-level treatment with clusters of size m and intra-cluster correlation rho:
        Var_clustered / Var_iid = 1 + (m - 1) rho.

Run:  python 01-linear-models/robust_se.py
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _bread(X):
    return np.linalg.inv(X.T @ X)


def ols_resid(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta, y - X @ beta


def classic_cov(X, e):
    n, k = X.shape
    return _bread(X) * (e @ e) / (n - k)


def hc_cov(X, e, kind="HC1"):
    n, k = X.shape
    B = _bread(X)
    if kind == "HC0":
        w = e ** 2
    elif kind == "HC1":
        w = e ** 2 * n / (n - k)
    elif kind == "HC2":
        h = np.einsum("ij,jk,ik->i", X, B, X)
        w = e ** 2 / (1 - h)
    else:
        raise ValueError(kind)
    return B @ (X.T * w) @ X @ B


def _inv_sqrt_psd(A):
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, 1e-12, None)
    return vecs @ np.diag(vals ** -0.5) @ vecs.T


def cluster_cov(X, e, clusters, kind="CR1"):
    """Cluster-robust covariance of beta_hat. clusters: one label per row."""
    n, k = X.shape
    B = _bread(X)
    labels, idx = np.unique(clusters, return_inverse=True)
    G = len(labels)
    meat = np.zeros((k, k))
    for g in range(G):
        rows = idx == g
        Xg, eg = X[rows], e[rows]
        if kind == "CR2":
            Hgg = Xg @ B @ Xg.T
            eg = _inv_sqrt_psd(np.eye(len(eg)) - Hgg) @ eg
        s = Xg.T @ eg
        meat += np.outer(s, s)
    V = B @ meat @ B
    if kind == "CR1":
        V *= G / (G - 1) * (n - 1) / (n - k)
    elif kind not in ("CR0", "CR2"):
        raise ValueError(kind)
    return V, G


def ols_summary(X, y, clusters=None, kind="CR2", j=1, level=0.95):
    """Estimate, SE, CI and p for coefficient j, with cluster-robust inference on t_{G-1} if clusters given."""
    beta, e = ols_resid(X, y)
    if clusters is None:
        V, df = classic_cov(X, e), X.shape[0] - X.shape[1]
    else:
        V, G = cluster_cov(X, e, clusters, kind)
        df = G - 1
    se = float(np.sqrt(V[j, j]))
    q = stats.t.ppf(0.5 + level / 2, df)
    return dict(est=float(beta[j]), se=se, ci=(beta[j] - q * se, beta[j] + q * se),
                p=float(2 * stats.t.sf(abs(beta[j] / se), df)), df=df)


def simulate_clustered(rng, G, m, rho, tau=0.0, cluster_treatment=True):
    """G clusters of m units, intra-cluster correlation rho; treatment assigned to whole clusters or units."""
    cl = np.repeat(np.arange(G), m)
    u = rng.normal(0, np.sqrt(rho), G)[cl]
    e = rng.normal(0, np.sqrt(1 - rho), G * m)
    if cluster_treatment:
        t = (rng.permutation(G) < G // 2).astype(float)[cl]
    else:
        t = (rng.permutation(G * m) < G * m // 2).astype(float)
    y = tau * t + u + e
    return np.column_stack([np.ones(G * m), t]), y, cl


def demo():
    rng = np.random.default_rng(24)
    m, rho, reps = 20, 0.1, 3000
    print(f"cluster-randomized treatment, m = {m} units per cluster, ICC = {rho}, design effect "
          f"1 + (m-1) rho = {1 + (m - 1) * rho:.1f}")
    print("  G   true sd | mean SE: classic  HC1   CR1   CR2 | 95% coverage: classic  CR1(t_G-1)  CR2(t_G-1)")
    for G in (10, 20, 50):
        est, se = [], {k: [] for k in ("classic", "HC1", "CR1", "CR2")}
        cov = {k: 0 for k in ("classic", "CR1", "CR2")}
        for _ in range(reps):
            X, y, cl = simulate_clustered(rng, G, m, rho)
            beta, e = ols_resid(X, y)
            est.append(beta[1])
            V = {"classic": classic_cov(X, e), "HC1": hc_cov(X, e, "HC1"),
                 "CR1": cluster_cov(X, e, cl, "CR1")[0], "CR2": cluster_cov(X, e, cl, "CR2")[0]}
            for k in se:
                se[k].append(np.sqrt(V[k][1, 1]))
            cov["classic"] += abs(beta[1]) <= stats.t.ppf(0.975, len(y) - 2) * np.sqrt(V["classic"][1, 1])
            for k in ("CR1", "CR2"):
                cov[k] += abs(beta[1]) <= stats.t.ppf(0.975, G - 1) * np.sqrt(V[k][1, 1])
        print(f"  {G:2d}  {np.std(est):.3f}  |        {np.mean(se['classic']):.3f}  {np.mean(se['HC1']):.3f} "
              f"{np.mean(se['CR1']):.3f} {np.mean(se['CR2']):.3f} |       {cov['classic']/reps:.3f}"
              f"      {cov['CR1']/reps:.3f}       {cov['CR2']/reps:.3f}")

    # when the treatment varies within clusters, clustering matters much less
    est, se_c, se_cr = [], [], []
    for _ in range(reps):
        X, y, cl = simulate_clustered(rng, 20, m, rho, cluster_treatment=False)
        beta, e = ols_resid(X, y)
        est.append(beta[1])
        se_c.append(np.sqrt(classic_cov(X, e)[1, 1]))
        se_cr.append(np.sqrt(cluster_cov(X, e, cl, "CR2")[0][1, 1]))
    print(f"unit-randomized treatment, same clustering: true sd {np.std(est):.3f}, classic SE "
          f"{np.mean(se_c):.3f}, CR2 SE {np.mean(se_cr):.3f}")


if __name__ == "__main__":
    demo()
