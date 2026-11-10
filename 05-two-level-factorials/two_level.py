"""2^p factorial experiments: the +/-1 contrast matrix, effect estimates, standard errors,
normal / half-normal plots, Lenth's pseudo standard error, and one-factor-at-a-time (OFAT)
(STAT 404 ch. 6).

Coding
    Each factor at a low (-1) and high (+1) level. With all 2^p combinations, the columns for the
    main effects and all interactions (elementwise products) are mutually orthogonal +/-1 vectors.

Regression on orthogonal contrasts
    Y = beta_0 + sum_S beta_S x_S + E, with x_S the product column for factor set S.
    beta_hat_S = x_S^T y / n (orthogonality makes every estimate a simple average),
    Var(beta_hat_S) = sigma^2 / n for every effect. The "effect" in the classical sense
    (mean at + minus mean at -) is 2 beta_S. SS(S) = n beta_hat_S^2, 1 DF each.

Unreplicated 2^p
    No DF left for error. Under "effect sparsity" most beta_S are 0, so the inert estimates are
    N(0, sigma^2/n): they fall on a straight line through 0 in a half-normal plot of |beta_hat|,
    and the active effects stand off the line.
    Lenth (1989): s0 = 1.5 median|b|, PSE = 1.5 median{|b| : |b| < 2.5 s0};
    margin of error ME = t_{0.975, m/3} PSE for m effects. (Not in the course notes.)

One factor at a time
    Changing one factor from a base point uses 2 runs per effect contrast; a factorial with the same
    number of runs uses every run in every estimate ("hidden replication") and also estimates
    interactions, which OFAT cannot see.

Run:  python 05-two-level-factorials/two_level.py
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import stats


def full_factorial(p):
    """2^p x p design in standard (Yates) order, first factor changing slowest."""
    return np.array(list(itertools.product([-1, 1], repeat=p)), dtype=float)


def contrast_matrix(D, max_order=None):
    """Columns for all main effects and interactions up to max_order. Labels use 1-based digits."""
    p = D.shape[1]
    max_order = p if max_order is None else max_order
    cols, labels = [], []
    for order in range(1, max_order + 1):
        for S in itertools.combinations(range(p), order):
            cols.append(np.prod(D[:, S], axis=1))
            labels.append("".join(str(i + 1) for i in S))
    return np.column_stack(cols), labels


def estimate_effects(X, y):
    """beta_hat for orthogonal +/-1 columns X (no intercept column), and the intercept."""
    n = len(y)
    return X.T @ y / n, y.mean()


def half_normal_quantiles(m):
    """Plotting positions for m ordered |effects|: Phi^{-1}(0.5 + 0.5 (i - 0.5) / m)."""
    return stats.norm.ppf(0.5 + 0.5 * (np.arange(1, m + 1) - 0.5) / m)


def lenth(b, alpha=0.05):
    b = np.abs(np.asarray(b, float))
    s0 = 1.5 * np.median(b)
    pse = 1.5 * np.median(b[b < 2.5 * s0])
    me = stats.t.ppf(1 - alpha / 2, len(b) / 3) * pse
    return pse, me


def demo():
    rng = np.random.default_rng(6)
    p = 4
    D = full_factorial(p)
    X, labels = contrast_matrix(D)
    print(f"2^{p}: {len(D)} runs, {X.shape[1]} effect columns, X^T X = n I: "
          f"{np.allclose(X.T @ X, len(D) * np.eye(X.shape[1]))}")

    beta_true = dict.fromkeys(labels, 0.0)
    beta_true.update({"1": 3.0, "3": -2.0, "13": 1.5})       # effect sparsity
    mu = 20 + sum(beta_true[l] * X[:, j] for j, l in enumerate(labels))
    y = mu + 1.0 * rng.standard_normal(len(D))

    b, b0 = estimate_effects(X, y)
    order = np.argsort(np.abs(b))
    q = half_normal_quantiles(len(b))
    pse, me = lenth(b)
    print("half-normal plot coordinates (largest 5): quantile, |beta|, effect")
    for qi, j in list(zip(q, order))[-5:]:
        print(f"  {qi:5.2f}  {abs(b[j]):5.2f}  {labels[j]}")
    active = [labels[j] for j in range(len(b)) if abs(b[j]) > me]
    print(f"Lenth PSE = {pse:.2f} (truth sigma/sqrt(n) = {1/np.sqrt(len(D)):.2f}), ME = {me:.2f}, "
          f"active: {active}")

    # replicated 2^3 x 2: standard error from pure error
    rng = np.random.default_rng(4)
    D3 = np.repeat(full_factorial(3), 2, axis=0)
    X3, lab3 = contrast_matrix(D3)
    y3 = 10 + 2 * X3[:, 0] - X3[:, 1] * X3[:, 2] + rng.standard_normal(len(D3))
    b3, _ = estimate_effects(X3, y3)
    resid = y3 - (y3.mean() + X3 @ b3)
    s2 = resid @ resid / (len(D3) - 8)
    se = np.sqrt(s2 / len(D3))
    sig = [f"{l} ({bb/se:.1f})" for l, bb in zip(lab3, b3) if abs(bb / se) > stats.t.ppf(0.975, 8)]
    print(f"replicated 2^3 x 2: se(beta) = {se:.2f} on 8 DF (truth 0.25); significant (t): {', '.join(sig)}")

    # OFAT vs factorial with 8 runs each, 3 factors, many simulated experiments
    reps, sd = 4000, 1.0
    ofat_var, fac_var = [], []
    base = np.array([-1.0, -1.0, -1.0])
    ofat_pts = [base] + [base * np.where(np.arange(3) == i, -1, 1) for i in range(3)]
    ofat_pts = np.repeat(np.array(ofat_pts), 2, axis=0)      # 4 distinct points, 2 runs each = 8
    for _ in range(reps):
        f = lambda Z: 1 + 1.0 * Z[:, 0] + sd * rng.standard_normal(len(Z))
        yo = f(ofat_pts)
        ofat_var.append((yo[2:4].mean() - yo[0:2].mean()) / 2)    # beta_1 uses only 4 of the 8 runs
        Df = full_factorial(3)
        fac_var.append((Df[:, 0] @ f(Df)) / 8)                    # beta_1 from all 8 runs
    print(f"OFAT vs 2^3, 8 runs each: var(beta_1) OFAT {np.var(ofat_var):.3f}, "
          f"factorial {np.var(fac_var):.3f}  (theory sigma^2/4 = 0.25 vs sigma^2/8 = 0.125)")


if __name__ == "__main__":
    demo()
