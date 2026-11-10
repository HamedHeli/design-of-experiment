"""Multi-factor (full) factorial experiments: interaction effects, ANOVA for any balanced
factorial, and 1-DF polynomial contrasts for main and interaction effects (STAT 404 ch. 5).

Interaction
    Factors A and B interact when the effect of changing A depends on the level of B.
    In a balanced factorial every effect is estimated from marginal means by inclusion-exclusion:
        main effect of A at level i:  ybar_i.. - ybar
        AB interaction at (i, j):     ybar_ij. - ybar_i.. - ybar_.j. + ybar
        ABC at (i, j, k):             ybar_ijk - ybar_ij. - ybar_i.k - ybar_.jk + ybar_i.. + ybar_.j. + ybar_..k - ybar
    and SS(effect) = (number of observations per margin cell) * sum(effect^2),
    with DF = product of (levels - 1) over the factors involved.

Unreplicated designs
    With one run per combination the highest-order interaction has to serve as the error term
    (assumed negligible). Replication (ch. 5.5) gives a pure-error estimate instead.

1-DF polynomial contrasts (3 equally spaced levels)
    linear L = (-1, 0, 1), quadratic Q = (1, -2, 1). An interaction contrast is the elementwise
    product of main-effect contrasts, e.g. (Q of A) x (L of B) on the A x B table of means.
    The 4 contrasts L x L, L x Q, Q x L, Q x Q split SS(AB) (4 DF) exactly.

Run:  python 04-factorial-experiments/factorial.py
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import stats


def _marginal_mean(M, keep):
    """Mean of cell-mean array M over all axes not in `keep`, broadcastable back to M."""
    axes = tuple(a for a in range(M.ndim) if a not in keep)
    return M.mean(axis=axes, keepdims=True) if axes else M


def effect_array(M, S):
    """Inclusion-exclusion estimate of the effect for factor subset S (tuple of axes)."""
    out = 0.0
    for r in range(len(S) + 1):
        for T in itertools.combinations(S, r):
            out = out + (-1) ** (len(S) - r) * _marginal_mean(M, T)
    return out


def factorial_anova(Y, names, max_order=None, error_from_order=None):
    """Balanced factorial ANOVA.

    Y: array of shape (l1, ..., lk) (unreplicated) or (l1, ..., lk, r) with r replicates.
    max_order: highest interaction order kept in the model; everything above is pooled into error.
    Returns a dict of rows name -> (df, ss, ms, F, p) plus 'Error'.
    """
    k = len(names)
    Y = np.asarray(Y, float)
    replicated = Y.ndim == k + 1
    M = Y.mean(axis=-1) if replicated else Y
    r = Y.shape[-1] if replicated else 1
    max_order = k if max_order is None else max_order
    rows, err_ss, err_df = {}, 0.0, 0
    for order in range(1, k + 1):
        for S in itertools.combinations(range(k), order):
            eff = effect_array(M, S)
            # summing the broadcast effect over all cells counts each margin cell once per
            # combination of the other factors; times r for replicates
            ss = float(r * (np.broadcast_to(eff, M.shape) ** 2).sum())
            df = int(np.prod([M.shape[a] - 1 for a in S]))
            label = ":".join(names[a] for a in S)
            if order <= max_order:
                rows[label] = [df, ss]
            else:
                err_ss += ss
                err_df += df
    if replicated:
        err_ss += float(((Y - M[..., None]) ** 2).sum())
        err_df += M.size * (r - 1)
    ms_e = err_ss / err_df
    out = {}
    for label, (df, ss) in rows.items():
        F = (ss / df) / ms_e
        out[label] = dict(df=df, ss=ss, ms=ss / df, F=F, p=stats.f.sf(F, df, err_df))
    out["Error"] = dict(df=err_df, ss=err_ss, ms=ms_e)
    return out


L3 = np.array([-1.0, 0.0, 1.0])
Q3 = np.array([1.0, -2.0, 1.0])


def contrast_ss(M, coefs, r_per_cell):
    """1-DF SS for a contrast on the full cell-means array M given per-axis coefficient vectors
    (None = average over that axis)."""
    C = np.ones(M.shape)
    for ax, c in enumerate(coefs):
        shape = [1] * M.ndim
        shape[ax] = M.shape[ax]
        vec = np.full(M.shape[ax], 1.0 / M.shape[ax]) if c is None else np.asarray(c, float)
        C = C * vec.reshape(shape)
    est = float((C * M).sum())
    return est, est ** 2 / float((C ** 2).sum() / r_per_cell)


def demo():
    rng = np.random.default_rng(5)
    names = ["A", "B", "C"]
    lv = np.array([-1.0, 0.0, 1.0])
    a, b, c = np.meshgrid(lv, lv, lv, indexing="ij")
    # curved in A, linear in B, an A-linear x B-linear interaction, C inert
    mu = 50 + 4 * a - 3 * a ** 2 + 2 * b + 1.5 * a * b
    Y = mu + 1.0 * rng.standard_normal(mu.shape)              # unreplicated 3^3 = 27 runs

    an = factorial_anova(Y, names, max_order=2)               # ABC (8 DF) is the error term
    print("3^3 factorial, ABC pooled as error")
    print("source   DF       SS        F        p")
    for k, row in an.items():
        if k == "Error":
            print(f"{k:7s} {row['df']:3d} {row['ss']:8.2f}")
        else:
            print(f"{k:7s} {row['df']:3d} {row['ss']:8.2f} {row['F']:8.2f} {row['p']:9.2g}")

    print("1-DF contrasts:")
    for lab, coefs in [("A_L", (L3, None, None)), ("A_Q", (Q3, None, None)),
                       ("B_L", (None, L3, None)), ("B_Q", (None, Q3, None)),
                       ("A_L x B_L", (L3, L3, None)), ("A_Q x B_L", (Q3, L3, None)),
                       ("A_L x B_Q", (L3, Q3, None)), ("A_Q x B_Q", (Q3, Q3, None))]:
        est, ss = contrast_ss(Y, coefs, 1)
        F = ss / an["Error"]["ms"]
        print(f"  {lab:10s} SS = {ss:7.2f}  p = {stats.f.sf(F, 1, an['Error']['df']):.2g}")
    s = sum(contrast_ss(Y, (ca, cb, None), 1)[1] for ca in (L3, Q3) for cb in (L3, Q3))
    print(f"  the four A x B contrasts sum to {s:.2f} = SS(A:B) {an['A:B']['ss']:.2f}")


if __name__ == "__main__":
    demo()
