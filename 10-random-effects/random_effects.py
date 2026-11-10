"""Fixed versus random effects: expected mean squares, F tests, variance components, and
prediction with random blocks (STAT 404 ch. 11).

Random block effects
    In an RBD the blocks (batches, days, litters) are often a sample from a population of blocks.
    Model: Y_ij = mu + tau_i + B_j + E_ij,  B_j ~ N(0, sigma_B^2), E_ij ~ N(0, sigma^2), independent.

Expected mean squares (RBD, t treatments, b blocks)
    E[MS(Trt)]   = sigma^2 + b sum tau_i^2 / (t - 1)
    E[MS(Block)] = sigma^2 + t sigma_B^2
    E[MS(Res)]   = sigma^2
    The F test for treatments is the same as with fixed blocks; the block test now asks sigma_B^2 = 0.

Variance components (ANOVA / method of moments)
    sigma_hat^2 = MS(Res),  sigma_hat_B^2 = (MS(Block) - MS(Res)) / t   (can be negative: truncate at 0)
    intraclass correlation  rho = sigma_B^2 / (sigma_B^2 + sigma^2).

Prediction: which variance applies
    Differences between treatments, tau_i - tau_k: blocks cancel, Var = 2 sigma^2 / b.
    The mean of treatment i in *these* blocks: Var = sigma^2 / b.
    The mean of treatment i over the *population* of blocks: Var = (sigma_B^2 + sigma^2) / b.
    A new observation in a new block: sigma_B^2 + sigma^2 (+ the estimation variance).

One-way random effects (no treatments): MS(Between) has E = sigma^2 + r sigma_A^2 with r per group,
    the classic setting for repeatability / gauge studies.

Run:  python 10-random-effects/random_effects.py
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def rbd_mean_squares(Y):
    Y = np.asarray(Y, float)
    t, b = Y.shape
    g = Y.mean()
    a = Y.mean(1) - g
    c = Y.mean(0) - g
    R = Y - g - a[:, None] - c[None, :]
    return dict(ms_t=b * (a ** 2).sum() / (t - 1), ms_b=t * (c ** 2).sum() / (b - 1),
                ms_r=(R ** 2).sum() / ((t - 1) * (b - 1)))


def variance_components(ms, t):
    s2 = ms["ms_r"]
    sb2 = max((ms["ms_b"] - ms["ms_r"]) / t, 0.0)
    return s2, sb2, sb2 / (sb2 + s2)


def simulate_rbd(rng, tau, b, sigma_b, sigma):
    t = len(tau)
    return 10 + np.asarray(tau)[:, None] + rng.normal(0, sigma_b, b)[None, :] + rng.normal(0, sigma, (t, b))


def treatment_mean_ci(Y, i, s2, sb2, population=True, level=0.95):
    """CI for the mean of treatment i, either in these blocks or over the population of blocks.
    The population version uses a Satterthwaite DF (not covered in the notes)."""
    t, b = Y.shape
    est = Y[i].mean()
    if not population:
        df = (t - 1) * (b - 1)
        se = np.sqrt(s2 / b)
    else:
        se = np.sqrt((sb2 + s2) / b)
        ms = rbd_mean_squares(Y)
        # sb2 + s2 = MS(Block)/t + (t-1)/t MS(Res)
        c1, c2 = 1 / t, (t - 1) / t
        df = (c1 * ms["ms_b"] + c2 * ms["ms_r"]) ** 2 / (
            (c1 * ms["ms_b"]) ** 2 / (b - 1) + (c2 * ms["ms_r"]) ** 2 / ((t - 1) * (b - 1)))
    q = stats.t.ppf(0.5 + level / 2, df)
    return est, (est - q * se, est + q * se)


def demo():
    rng = np.random.default_rng(12)
    tau = np.array([0.0, 1.0, 2.0, 0.5])
    t, b, sigma_b, sigma = len(tau), 6, 2.0, 1.0
    sims = [rbd_mean_squares(simulate_rbd(rng, tau, b, sigma_b, sigma)) for _ in range(20000)]
    avg = {k: np.mean([s[k] for s in sims]) for k in sims[0]}
    print("expected mean squares, simulated vs theory (t = 4, b = 6, sigma_B = 2, sigma = 1):")
    print(f"  MS(Trt)   {avg['ms_t']:.2f}  vs {sigma**2 + b * ((tau - tau.mean())**2).sum() / (t - 1):.2f}")
    print(f"  MS(Block) {avg['ms_b']:.2f}  vs {sigma**2 + t * sigma_b**2:.2f}")
    print(f"  MS(Res)   {avg['ms_r']:.2f}  vs {sigma**2:.2f}")

    Y = simulate_rbd(rng, tau, b, sigma_b, sigma)
    ms = rbd_mean_squares(Y)
    s2, sb2, icc = variance_components(ms, t)
    print(f"one experiment: sigma^2 = {s2:.2f}, sigma_B^2 = {sb2:.2f}, ICC = {icc:.2f} "
          f"(truth 1, 4, {4/5:.2f})")
    for pop in (False, True):
        est, (lo, hi) = treatment_mean_ci(Y, 1, s2, sb2, population=pop)
        print(f"  95% CI for treatment 2 mean, {'population of blocks' if pop else 'these blocks      '}: "
              f"{est:.2f} ({lo:.2f}, {hi:.2f})")

    # coverage of the two intervals for the population mean 10 + tau_2
    cover = {False: 0, True: 0}
    for _ in range(4000):
        Y = simulate_rbd(rng, tau, b, sigma_b, sigma)
        s2, sb2, _ = variance_components(rbd_mean_squares(Y), t)
        for pop in (False, True):
            _, (lo, hi) = treatment_mean_ci(Y, 1, s2, sb2, population=pop)
            cover[pop] += lo <= 11.0 <= hi
    print(f"coverage of the population mean: fixed-block interval {cover[False]/4000:.2f}, "
          f"random-block interval {cover[True]/4000:.2f}")


if __name__ == "__main__":
    demo()
