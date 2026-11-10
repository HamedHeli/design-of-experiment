"""Why design experiments: a lurking variable biases an observational comparison, randomization
removes the bias, blocking removes the variance it adds, replication shrinks what is left
(STAT 404 ch. 1, with ch. 10 for the formal version).

Setup
    Units (patients, batches) differ in a property z that also affects the response:
        Y = 10 + tau * T + 3 z + E,   tau = 1.
    Observational study: units with high z are more likely to get the treatment (doctors give the new
    drug to sicker patients, operators use the new setting on good batches). Then
        E[ybar_T - ybar_C] = tau + 3 (E[z | T] - E[z | C])  != tau.
    Randomized experiment: T is assigned by a coin, independent of z, so the bias term is 0.
    Randomized block design: pair units with similar z and randomize within pairs; the z-term then
    nearly cancels within every pair, so the estimate is also much less variable.

Run:  python 00-introduction-and-principles/observational_vs_randomized.py
"""

from __future__ import annotations

import numpy as np


def simulate(rng, n, how, tau=1.0):
    z = rng.standard_normal(n)
    if how == "observational":
        t = (rng.random(n) < 1 / (1 + np.exp(-2 * z))).astype(float)   # high z -> treated
    elif how == "randomized":
        t = (rng.permutation(n) < n // 2).astype(float)
    elif how == "blocked":
        order = np.argsort(z)                                     # pairs of adjacent z values
        t = np.zeros(n)
        for a, b in order.reshape(-1, 2):
            t[a if rng.random() < 0.5 else b] = 1.0
    y = 10 + tau * t + 3 * z + rng.standard_normal(n)
    return y, t, z


def estimate(y, t, z, how):
    if how != "blocked":
        return y[t == 1].mean() - y[t == 0].mean()
    order = np.argsort(z).reshape(-1, 2)
    d = [y[a] - y[b] if t[a] == 1 else y[b] - y[a] for a, b in order]
    return float(np.mean(d))


def demo():
    rng = np.random.default_rng(0)
    print("estimate of tau = 1 over 5000 studies of n = 40 units")
    for how in ("observational", "randomized", "blocked"):
        est = [estimate(*simulate(rng, 40, how), how) for _ in range(5000)]
        print(f"  {how:13s}  mean {np.mean(est):5.2f}   sd {np.std(est):.2f}")
    for n in (20, 80, 320):
        est = [estimate(*simulate(rng, n, "randomized"), "randomized") for _ in range(3000)]
        print(f"  randomized, n = {n:3d}: sd {np.std(est):.2f}")


if __name__ == "__main__":
    demo()
