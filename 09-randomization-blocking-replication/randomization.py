"""Counter-measures against other variables: randomization (and randomization tests), blocking,
replication, and auxiliary variables (STAT 404 ch. 10).

Why randomize
    An unknown unit property z can bias a comparison if it is correlated with the treatment.
    Random allocation makes the imbalance zbar_A - zbar_B a random variable with mean 0 whatever z is.
    With n_rep units per arm out of n = 2 n_rep and W_i = +/-1 the arm indicator,
        zbar_A - zbar_B = sum W_i z_i / n_rep,  E = 0,  Var = 4 s_z^2 / n   (s_z^2 with n-1 divisor),
    so the imbalance shrinks as replication grows, and blocking on z removes it outright.

Randomization test (Fisher)
    Strong (sharp) null: every unit's response would be the same under either treatment. Then the
    data are fixed and only the labels are random: recompute the statistic for every possible
    allocation (or a Monte Carlo sample) and report the fraction at least as extreme as observed.
    Matched pairs: 2^m equally likely sign flips of the m within-pair differences.
    The usual t test is a good approximation to this exact test, and needs no normality assumption
    to be justified, only the randomization that was actually done.

Replication
    Var(effect estimate) is proportional to 1/n_rep, and replication is what estimates sigma^2.

Run:  python 09-randomization-blocking-replication/randomization.py
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import stats


def paired_randomization_test(d):
    """Exact two-sided test of the sharp null for within-pair differences d."""
    d = np.asarray(d, float)
    obs = abs(d.mean())
    signs = np.array(list(itertools.product([-1, 1], repeat=len(d))))
    ref = np.abs((signs * d).mean(1))
    return float(np.mean(ref >= obs - 1e-12)), ref


def two_sample_permutation_test(y, treat, n_perm, rng):
    y = np.asarray(y, float)
    treat = np.asarray(treat, bool)
    obs = abs(y[treat].mean() - y[~treat].mean())
    count = 0
    for _ in range(n_perm):
        t = rng.permutation(treat)
        count += abs(y[t].mean() - y[~t].mean()) >= obs - 1e-12
    return (count + 1) / (n_perm + 1)


def imbalance_variance(z):
    """Theoretical variance of zbar_A - zbar_B under complete randomization."""
    z = np.asarray(z, float)
    return 4 * z.var(ddof=1) / len(z)


def demo():
    rng = np.random.default_rng(11)

    # 1. randomization balances an unknown covariate on average
    n = 20
    z = rng.gamma(2.0, 10.0, n)                         # e.g. patient age-related severity
    diffs = []
    for _ in range(20000):
        a = rng.permutation(n) < n // 2
        diffs.append(z[a].mean() - z[~a].mean())
    print(f"zbar_A - zbar_B over 20000 randomizations: mean {np.mean(diffs):.3f}, "
          f"var {np.var(diffs):.2f} (theory {imbalance_variance(z):.2f})")

    # 2. exact randomization test for 8 matched pairs vs the paired t test
    m = 8
    d = 2.0 + rng.normal(0, 2.5, m)                      # B - A within pair
    p_rand, ref = paired_randomization_test(d)
    p_t = stats.ttest_1samp(d, 0).pvalue
    print(f"8 pairs: randomization p = {p_rand:.4f} ({len(ref)} sign patterns), paired t p = {p_t:.4f}")

    # 3. two-sample permutation test vs the pooled t test
    y = np.r_[rng.normal(10, 2, 12), rng.normal(11.8, 2, 12)]
    treat = np.r_[np.zeros(12, bool), np.ones(12, bool)]
    p_perm = two_sample_permutation_test(y, treat, 20000, rng)
    print(f"CRD with 12 + 12 units: permutation p = {p_perm:.4f}, two-sample t p = "
          f"{stats.ttest_ind(y[treat], y[~treat]).pvalue:.4f}")

    # 4. blocking vs randomization when z is known: paired (blocked) vs unpaired analysis
    effect, reps = 2.0, 4000
    se_pair, se_crd = [], []
    for _ in range(reps):
        zz = rng.normal(0, 5, 2 * m)                     # strong unit effects
        zz.sort()
        yA = zz[0::2] + rng.normal(0, 1, m)             # pairs of similar units
        yB = zz[1::2] + effect + rng.normal(0, 1, m)
        se_pair.append((yB - yA).mean())
        perm = rng.permutation(2 * m)                    # same units, completely randomized
        yy = zz[perm] + np.r_[np.zeros(m), np.full(m, effect)] + rng.normal(0, 1, 2 * m)
        se_crd.append(yy[m:].mean() - yy[:m].mean())
    print(f"sd of the effect estimate, 16 units: paired on z {np.std(se_pair):.2f}, "
          f"completely randomized {np.std(se_crd):.2f}")

    # 5. replication
    for n_rep in (5, 10, 20, 40):
        est = [rng.normal(effect, 3, n_rep).mean() - rng.normal(0, 3, n_rep).mean() for _ in range(4000)]
        print(f"  n_rep = {n_rep:2d}: sd(estimate) = {np.std(est):.2f}  (theory {3*np.sqrt(2/n_rep):.2f})")


if __name__ == "__main__":
    demo()
