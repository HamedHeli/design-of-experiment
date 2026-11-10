"""Checks every number in worked-examples.md. Each example is small enough to do by hand.

Run:  python basics-variance-and-standard-errors/worked_examples.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "01-linear-models"))
import basics as b  # noqa: E402
from robust_se import classic_cov, cluster_cov, ols_resid  # noqa: E402

A = np.array([4.0, 6, 8, 10, 12])
B = np.array([7.0, 9, 11, 13, 15, 17, 19])
BEFORE = np.array([10.0, 12, 9, 14, 11])
AFTER = np.array([12.0, 13, 11, 15, 14])


def ex1_mean_sd_se():
    return dict(mean=A.mean(), ss=((A - A.mean()) ** 2).sum(), var=b.sample_var(A), sd=np.sqrt(b.sample_var(A)),
                se=b.se_mean(A), var_n=b.sample_var(A, 0))


def ex2_variance_of_difference():
    return dict(var_indep=9 + 16, sd_indep=5.0, var_corr=9 + 16 - 2 * 0.6 * 3 * 4)


def ex3_two_groups():
    p = b.pooled_two_sample(B, A)
    w = b.welch_two_sample(B, A)
    for r in (p, w):
        q = stats.t.ppf(0.975, r["df"])
        r["ci"] = (r["diff"] - q * r["se"], r["diff"] + q * r["se"])
    return dict(varB=b.sample_var(B), pooled=p, welch=w)


def ex4_paired():
    d = AFTER - BEFORE
    pr = b.paired(BEFORE, AFTER)
    up = b.welch_two_sample(AFTER, BEFORE)
    return dict(d=d, var_d=b.sample_var(d), paired=pr, unpaired=up,
                corr=np.corrcoef(BEFORE, AFTER)[0, 1])


def ex5_contrast():
    means, n, mse, df = np.array([10.0, 12, 17]), 4, 6.0, 9
    c = np.array([-0.5, -0.5, 1.0])
    est = c @ means
    se = np.sqrt(mse * (c ** 2 / n).sum())
    return dict(est=est, se=se, t=est / se, p=2 * stats.t.sf(abs(est / se), df))


def ex6_proportions():
    return b.two_proportions(120, 1000, 100, 1000)


def ex7_lift():
    mT, mC, seT, seC = 10.5, 10.0, 0.2, 0.2
    lift = mT / mC - 1
    var = seT ** 2 / mC ** 2 + mT ** 2 * seC ** 2 / mC ** 4
    return dict(lift=lift, se=np.sqrt(var), var=var)


def ex8_design_effect():
    deff = b.design_effect(20, 0.05)
    return dict(deff=deff, n_eff=1000 / deff, se_inflation=np.sqrt(deff))


def ex9_cluster_robust():
    y = np.array([5.0, 7, 9, 11, 4, 6, 2, 4])
    cl = np.repeat([1, 2, 3, 4], 2)
    t = np.array([1.0, 1, 1, 1, 0, 0, 0, 0])
    X = np.column_stack([np.ones(8), t])
    beta, e = ols_resid(X, y)
    out = dict(diff=beta[1], resid=e, sums=np.bincount(cl, e)[1:],
               se_classic=np.sqrt(classic_cov(X, e)[1, 1]))
    for k in ("CR0", "CR1", "CR2"):
        out["se_" + k] = np.sqrt(cluster_cov(X, e, cl, k)[0][1, 1])
    m = np.array([6.0, 10, 5, 3])
    out["se_means"] = np.sqrt(m[:2].var(ddof=1) / 2 + m[2:].var(ddof=1) / 2)
    return out


def demo():
    r = ex1_mean_sd_se()
    print(f"1. A: mean {r['mean']}, SS {r['ss']}, s^2 {r['var']}, s {r['sd']:.3f}, SE {r['se']:.3f}, "
          f"divide-by-n variance {r['var_n']}")
    r = ex2_variance_of_difference()
    print(f"2. Var(X - Y): independent {r['var_indep']} (SD {r['sd_indep']}), corr 0.6: {r['var_corr']:.1f}")
    r = ex3_two_groups()
    p, w = r["pooled"], r["welch"]
    print(f"3. B: s^2 {r['varB']:.3f}; diff {p['diff']}; pooled sp^2 {p['sp']**2:.1f}, SE {p['se']:.3f}, "
          f"t {p['t']:.3f} on {p['df']} DF, p {p['p']:.3f}, CI ({p['ci'][0]:.2f}, {p['ci'][1]:.2f})")
    print(f"   Welch SE {w['se']:.3f}, t {w['t']:.3f} on {w['df']:.2f} DF, p {w['p']:.3f}, "
          f"CI ({w['ci'][0]:.2f}, {w['ci'][1]:.2f})")
    r = ex4_paired()
    print(f"4. d {r['d']}, s_d^2 {r['var_d']:.2f}, paired SE {r['paired']['se']:.3f}, t {r['paired']['t']:.2f}, "
          f"p {r['paired']['p']:.4f}; unpaired SE {r['unpaired']['se']:.3f}, t {r['unpaired']['t']:.2f}, "
          f"p {r['unpaired']['p']:.3f}; corr(before, after) {r['corr']:.2f}")
    r = ex5_contrast()
    print(f"5. contrast {r['est']}, SE {r['se']}, t {r['t']}, p {r['p']:.4f}")
    r = ex6_proportions()
    print(f"6. diff {r['diff']:.3f}, SE(CI) {r['se_ci']:.5f}, CI ({r['ci'][0]:.4f}, {r['ci'][1]:.4f}), "
          f"SE(test) {r['se_test']:.5f}, z {r['z']:.3f}, p {r['p']:.3f}")
    r = ex7_lift()
    print(f"7. lift {r['lift']:.3f}, Var {r['var']:.6f}, SE {r['se']:.4f}")
    r = ex8_design_effect()
    print(f"8. design effect {r['deff']:.2f}, effective n {r['n_eff']:.0f}, SE x {r['se_inflation']:.3f}")
    r = ex9_cluster_robust()
    print(f"9. diff {r['diff']:.1f}, residuals {r['resid']}, cluster sums {r['sums']}")
    print(f"   SE classic {r['se_classic']:.3f}, CR0 {r['se_CR0']:.3f}, CR1 {r['se_CR1']:.3f}, "
          f"CR2 {r['se_CR2']:.3f}, cluster-means {r['se_means']:.3f}")


if __name__ == "__main__":
    demo()
