"""Analysis of covariance (ANCOVA) in a randomized experiment. Not developed in the STAT 404 notes,
which introduce it in one paragraph of ch. 10.6.

Setting
    Completely randomized experiment, treatment indicator T in {0, 1}, response Y, and a covariate z
    measured before randomization (or one that the treatment cannot affect).

Three unbiased estimators of the average treatment effect tau
    difference in means:  ybar_1 - ybar_0
    change score:         (y - z)bar_1 - (y - z)bar_0         (fixes the slope on z at 1)
    ANCOVA:               the OLS coefficient on T in  Y = a + tau T + beta (z - zbar) + E
                          (estimates the slope beta from the data)

Variance (equal arm sizes, Var Y = Var z = sigma^2, corr(Y, z) = rho within arms)
    difference in means  2 sigma^2 / n_arm * 1
    change score         2 sigma^2 / n_arm * 2(1 - rho)     better than DiM only if rho > 1/2
    ANCOVA               2 sigma^2 / n_arm * (1 - rho^2)    never worse (asymptotically)

Lin (2013) interacted ANCOVA
    Y = a + tau T + beta (z - zbar) + gamma T (z - zbar) + E, with robust (HC) standard errors.
    Asymptotically at least as precise as DiM even when the model is wrong or effects are heterogeneous.

The rule that matters (Cox 1958; the notes, sec. 10.6)
    The covariate must not be affected by the treatment. Adjusting for a post-treatment variable that the
    treatment moves biases tau: it "explains away" part of the effect.

Run:  python 12-variance-reduction-ancova-cuped-cupac/ancova.py
"""

from __future__ import annotations

import numpy as np


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    se_classic = np.sqrt(np.diag(XtXi) * (r @ r) / (n - k))
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    meat = (X * (r ** 2 / (1 - h))[:, None]).T @ X              # HC2
    se_hc2 = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    return beta, se_classic, se_hc2


def diff_in_means(y, t):
    y1, y0 = y[t == 1], y[t == 0]
    return y1.mean() - y0.mean(), np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))


def change_score(y, z, t):
    return diff_in_means(y - z, t)


def ancova(y, z, t, interacted=False):
    zc = z - z.mean()
    cols = [np.ones_like(y), t, zc] + ([t * zc] if interacted else [])
    beta, se, se_hc2 = ols(np.column_stack(cols), y)
    return beta[1], (se_hc2[1] if interacted else se[1])


def simulate(rng, n, rho, tau=0.5):
    z = rng.standard_normal(n)
    t = (rng.permutation(n) < n // 2).astype(float)
    y = rho * z + np.sqrt(1 - rho ** 2) * rng.standard_normal(n) + tau * t
    return y, z, t


def demo():
    rng = np.random.default_rng(14)
    n, reps, tau = 200, 4000, 0.5
    print("sd of the estimate of tau over 4000 randomized experiments, n = 200 (truth tau = 0.5)")
    print(" rho    DiM   change  ANCOVA  Lin   | theory: change/DiM = sqrt(2(1-rho)), ANCOVA/DiM = sqrt(1-rho^2)")
    for rho in (0.0, 0.3, 0.5, 0.7, 0.9):
        est = {k: [] for k in ("dim", "chg", "anc", "lin")}
        for _ in range(reps):
            y, z, t = simulate(rng, n, rho, tau)
            est["dim"].append(diff_in_means(y, t)[0])
            est["chg"].append(change_score(y, z, t)[0])
            est["anc"].append(ancova(y, z, t)[0])
            est["lin"].append(ancova(y, z, t, interacted=True)[0])
        sd = {k: np.std(v) for k, v in est.items()}
        print(f" {rho:.1f}  {sd['dim']:.3f}  {sd['chg']:.3f}  {sd['anc']:.3f}  {sd['lin']:.3f} |"
              f"  {sd['chg']/sd['dim']:.2f} vs {np.sqrt(2*(1-rho)):.2f},  "
              f"{sd['anc']/sd['dim']:.2f} vs {np.sqrt(1-rho**2):.2f}")

    # coverage of the ANCOVA 95% CI
    cover = 0
    for _ in range(reps):
        y, z, t = simulate(rng, n, 0.7, tau)
        e, se = ancova(y, z, t)
        cover += abs(e - tau) <= 1.97 * se
    print(f"ANCOVA 95% CI coverage at rho = 0.7: {cover/reps:.3f}")

    # the warning: adjusting for a post-treatment covariate
    bias_pre, bias_post = [], []
    for _ in range(2000):
        n2 = 400
        u = rng.standard_normal(n2)
        t = (rng.permutation(n2) < n2 // 2).astype(float)
        m = u + 1.0 * t + 0.5 * rng.standard_normal(n2)           # mediator moved by T
        y = 0.8 * m + 0.2 * t + 0.5 * rng.standard_normal(n2)       # total effect 0.8 + 0.2 = 1.0
        bias_pre.append(ancova(y, u, t)[0] - 1.0)
        bias_post.append(ancova(y, m, t)[0] - 1.0)
    print(f"total effect 1.0: adjusting for a pre-treatment z gives bias {np.mean(bias_pre):+.3f}; "
          f"adjusting for a post-treatment mediator gives bias {np.mean(bias_post):+.3f}")


if __name__ == "__main__":
    demo()
