"""Latin square and Graeco-Latin square designs: construction, randomization, ANOVA
(STAT 404 ch. 9).

Latin square
    l treatments, two blocking factors (rows, columns) each at l levels, l^2 runs. Every treatment
    appears once in every row and once in every column, so treatments are balanced against both
    blocking factors at once.

Construction and randomization
    Start from the cyclic square L[i, j] = (i + j) mod l, then randomly permute the rows, the
    columns, and the treatment labels.

Model (additive: no interactions between rows, columns and treatments)
    Y_ijk = mu + rho_i + gamma_j + tau_k + E,  only l^2 of the l^3 (i, j, k) cells observed.
    ANOVA: rows, columns, treatments each l-1 DF; residual (l-1)(l-2) DF.
    Balance means each treatment mean is an unbiased estimate of mu + tau_k under additivity.

Graeco-Latin square
    Superimpose two orthogonal Latin squares (every (Latin, Greek) pair occurs once): a third blocking
    factor or a second treatment factor in the same l^2 runs. For prime l: (i + j) mod l and
    (i + 2j) mod l. None exists for l = 6 (Euler's 36 officers).

Run:  python 08-latin-squares/latin_square.py
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def cyclic_square(l):
    i, j = np.indices((l, l))
    return (i + j) % l


def randomize_square(L, rng):
    l = len(L)
    L = L[rng.permutation(l)][:, rng.permutation(l)]
    return rng.permutation(l)[L]


def is_latin(L):
    l = len(L)
    full = set(range(l))
    return all(set(r) == full for r in L) and all(set(c) == full for c in L.T)


def graeco_latin(l):
    i, j = np.indices((l, l))
    return (i + j) % l, (i + 2 * j) % l


def are_orthogonal(L1, L2):
    l = len(L1)
    return len(set(zip(L1.ravel(), L2.ravel()))) == l * l


def latin_anova(Y, L):
    """Y[i, j] observed in row i, column j under treatment L[i, j]."""
    Y = np.asarray(Y, float)
    l = len(Y)
    g = Y.mean()
    ss_r = l * ((Y.mean(1) - g) ** 2).sum()
    ss_c = l * ((Y.mean(0) - g) ** 2).sum()
    tmeans = np.array([Y[L == k].mean() for k in range(l)])
    ss_t = l * ((tmeans - g) ** 2).sum()
    ss_tot = ((Y - g) ** 2).sum()
    ss_e = ss_tot - ss_r - ss_c - ss_t
    df, df_e = l - 1, (l - 1) * (l - 2)
    ms_e = ss_e / df_e
    F = (ss_t / df) / ms_e
    return dict(ss_r=ss_r, ss_c=ss_c, ss_t=ss_t, ss_e=ss_e, df_e=df_e, ms_e=ms_e,
                F_t=F, p_t=stats.f.sf(F, df, df_e), tmeans=tmeans)


def demo():
    rng = np.random.default_rng(10)
    l = 5
    L = randomize_square(cyclic_square(l), rng)
    print("randomized 5 x 5 Latin square (treatments A-E):")
    for row in L:
        print("   " + " ".join("ABCDE"[k] for k in row))
    print(f"still Latin after randomization: {is_latin(L)}")

    tau = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    rows = rng.normal(0, 3, l)                     # e.g. operators
    cols = rng.normal(0, 3, l)                     # e.g. days
    Y = 20 + rows[:, None] + cols[None, :] + tau[L] + 0.6 * rng.standard_normal((l, l))
    an = latin_anova(Y, L)
    print(f"Latin square ANOVA: F(treatments) = {an['F_t']:.2f} on {l-1},{an['df_e']} DF, "
          f"p = {an['p_t']:.2g}")
    print("treatment means minus A:", np.round(an["tmeans"] - an["tmeans"][0], 2), "(truth", tau, ")")
    # analysed as a CRD (ignoring both blocking factors)
    ss_e_crd = an["ss_e"] + an["ss_r"] + an["ss_c"]
    df_crd = l * l - l
    F_crd = (an["ss_t"] / (l - 1)) / (ss_e_crd / df_crd)
    print(f"same data as a CRD: F = {F_crd:.2f}, p = {stats.f.sf(F_crd, l - 1, df_crd):.2g}")

    G1, G2 = graeco_latin(5)
    print(f"5 x 5 Graeco-Latin square: both Latin {is_latin(G1) and is_latin(G2)}, "
          f"orthogonal {are_orthogonal(G1, G2)}")


if __name__ == "__main__":
    demo()
