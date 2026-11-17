"""Multi-armed bandits: the exploration / exploitation trade-off as adaptive experimental design.
Not in the STAT 404 notes.

Setting
    K arms (treatments, e.g. versions of a web page). Arm k returns a reward Y ~ Bernoulli(p_k)
    (converted or not). Units arrive one at a time; for each one we choose an arm and see its reward
    immediately. The p_k are unknown.

Two goals that pull apart
    Exploit: give each unit the arm that looks best so far (earn reward now).
    Explore: give some units the arms that look worse, because they may only *look* worse
             (learn, so that future units are treated better).
    A fixed design (CRD with equal allocation, the classic A/B/n test) is pure exploration until the
    end; always choosing the current leader is pure exploitation and can lock onto a wrong arm forever.

Regret (the price of not knowing p)
    R_T = T * max_k p_k - sum_t p_{A_t}       (pseudo-regret: expected reward lost to sub-optimal arms)
        = sum_k (p* - p_k) E[N_k(T)],         N_k(T) = number of units given arm k.
    Equal allocation: R_T grows linearly, T (p* - mean p).
    Fixed epsilon-greedy: still linear, about eps * T * (p* - mean p), plus a greedy phase cost.
    Decaying epsilon, UCB, Thompson sampling: logarithmic, O(log T) (Lai & Robbins lower bound:
    sum_k (p* - p_k) / KL(p_k, p*) * log T for any consistent policy).

Epsilon-greedy
    with probability eps   : choose an arm uniformly at random      (explore)
    with probability 1-eps : choose argmax_k  s_k / n_k              (exploit)
    Decaying version: eps_t = min(1, c K / t), so exploration fades as evidence accumulates.

Thompson sampling (Thompson 1933), Beta-Bernoulli version
    Prior p_k ~ Beta(1, 1). After s_k successes in n_k units, posterior Beta(1 + s_k, 1 + n_k - s_k).
    For each new unit: draw theta_k from every posterior, choose argmax_k theta_k, update.
    So P(choose arm k) = P(arm k is the best | data): "probability matching". Uncertain arms still get
    drawn sometimes (exploration), and the allocation sharpens on its own as the posteriors narrow.

UCB1 (Auer, Cesa-Bianchi & Fischer 2002), for comparison
    choose argmax_k  s_k / n_k + sqrt(2 ln t / n_k):  optimism in the face of uncertainty.

The design cost: inference after adaptive allocation
    The arm means s_k / n_k are biased downward for arms that had bad luck early (they stop being
    sampled before the bad luck can be corrected), and losing arms get few units, so the SE of any
    comparison with them is large. A bandit maximizes reward during the experiment; a fixed design
    maximizes what you learn about every arm.

Run:  python 13-adaptive-experiments-bandits/bandits.py
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, stats


# --- policies: each maps the current counts (reps x K) to one arm per replicate --------------------
def _argmax_random_ties(v, rng):
    return np.argmax(v + 1e-12 * rng.random(v.shape), axis=1)


def choose_uniform(n, s, t, rng):
    """Fixed design: every unit is randomized to an arm with equal probability (CRD / A/B/n test)."""
    return rng.integers(n.shape[1], size=n.shape[0])


def choose_greedy(n, s, t, rng, eps=0.1, decay=None):
    """Epsilon-greedy. decay=c gives eps_t = min(1, c K / (t + 1)) instead of a fixed eps."""
    reps, K = n.shape
    e = min(1.0, decay * K / (t + 1)) if decay is not None else eps
    means = np.where(n > 0, s / np.maximum(n, 1), np.inf)       # untried arms first
    arm = _argmax_random_ties(means, rng)
    explore = rng.random(reps) < e
    arm[explore] = rng.integers(K, size=int(explore.sum()))
    return arm


def choose_thompson(n, s, t, rng, a0=1.0, b0=1.0):
    """Thompson sampling with independent Beta(a0, b0) priors."""
    theta = rng.beta(a0 + s, b0 + n - s)
    return np.argmax(theta, axis=1)


def choose_ucb1(n, s, t, rng):
    means = s / np.maximum(n, 1)
    bonus = np.sqrt(2 * np.log(t + 1) / np.maximum(n, 1))
    return _argmax_random_ties(np.where(n > 0, means + bonus, np.inf), rng)


POLICIES = {
    "equal allocation (A/B/n)": (choose_uniform, {}),
    "epsilon-greedy, eps = 0.1": (choose_greedy, {"eps": 0.1}),
    "epsilon-greedy, decaying": (choose_greedy, {"decay": 100.0}),
    "Thompson sampling": (choose_thompson, {}),
    "UCB1": (choose_ucb1, {}),
}


def simulate(policy, p, horizon, reps, rng, checkpoints=(), **kw):
    """Run `reps` independent bandit experiments in parallel (vectorized over replicates).

    Returns n, s (reps x K counts and successes), regret (reps,) total pseudo-regret,
    and the mean cumulative regret at each checkpoint.
    """
    p = np.asarray(p, float)
    K = len(p)
    n = np.zeros((reps, K))
    s = np.zeros((reps, K))
    regret = np.zeros(reps)
    idx = np.arange(reps)
    curve, cps = {}, set(checkpoints)
    for t in range(horizon):
        arm = policy(n, s, t, rng, **kw)
        r = rng.random(reps) < p[arm]
        n[idx, arm] += 1
        s[idx, arm] += r
        regret += p.max() - p[arm]
        if t + 1 in cps:
            curve[t + 1] = regret.mean()
    return n, s, regret, curve


# --- the Bayesian quantities behind Thompson sampling ---------------------------------------------
def prob_b_beats_a(a1, b1, a2, b2):
    """P(p_B > p_A) for independent p_A ~ Beta(a1, b1), p_B ~ Beta(a2, b2), by 1-D integration:
    integral of f_A(x) * P(p_B > x) dx."""
    val, _ = integrate.quad(lambda x: stats.beta.pdf(x, a1, b1) * stats.beta.sf(x, a2, b2), 0, 1)
    return val


def prob_best(succ, fail, rng, draws=200_000, a0=1.0, b0=1.0):
    """Monte Carlo P(arm k is best | data) for every arm: the Thompson sampling allocation."""
    th = rng.beta(a0 + np.asarray(succ, float), b0 + np.asarray(fail, float), size=(draws, len(succ)))
    return np.bincount(th.argmax(axis=1), minlength=len(succ)) / draws


def worked_example():
    """Two arms after 22 units: A 3/10, B 6/12. Next-unit allocation under each rule."""
    pb = prob_b_beats_a(1 + 3, 1 + 7, 1 + 6, 1 + 6)
    eps = 0.1
    return {
        "post_A": (4, 8), "post_B": (7, 7),
        "mean_A": 4 / 12, "mean_B": 7 / 14,
        "greedy_P_B": 1 - eps + eps / 2,            # B leads: exploit + half of the exploration
        "thompson_P_B": pb,
    }


# --- demos ---------------------------------------------------------------------------------------
def demo():
    rng = np.random.default_rng(13)

    w = worked_example()
    print("worked example: A converted 3/10, B 6/12, Beta(1, 1) priors")
    print(f"  posteriors A ~ Beta{w['post_A']} (mean {w['mean_A']:.3f}), B ~ Beta{w['post_B']} (mean {w['mean_B']:.3f})")
    print(f"  P(next unit gets B): eps-greedy (0.1) {w['greedy_P_B']:.3f}, Thompson = P(p_B > p_A) {w['thompson_P_B']:.3f}")

    p = np.array([0.040, 0.050, 0.055, 0.060])   # control + three variants, conversion rates
    T, reps = 20_000, 400
    cps = (1_000, 5_000, 20_000)
    print(f"\n{len(p)} arms, p = {p.tolist()}, horizon {T:,} units, {reps} simulated experiments per policy")
    head = (f"{'policy':<28}{'regret@1k':>10}{'@5k':>8}{'@20k':>8}{'share best':>11}"
            f"{'picks best':>11}{'n control':>10}{'bias ctrl':>10}{'sd lift':>9}")
    print(head)
    for name, (pol, kw) in POLICIES.items():
        n, s, reg, curve = simulate(pol, p, T, reps, rng, checkpoints=cps, **kw)
        means = s / np.maximum(n, 1)
        share = (n[:, -1] / T).mean()
        picks = (np.argmax(means, axis=1) == p.argmax()).mean()
        bias0 = (means[:, 0] - p[0]).mean()
        lift = means[:, -1] - means[:, 0]                          # estimated best - control
        print(f"{name:<28}{curve[1_000]:>10.1f}{curve[5_000]:>8.1f}{curve[20_000]:>8.1f}{share:>11.2f}"
              f"{picks:>11.2f}{n[:, 0].mean():>10.0f}{bias0:>+10.4f}{lift.std():>9.4f}")

    print("\nfixed epsilon: the explore/exploit dial (same arms, horizon 20k)")
    for eps in (0.0, 0.02, 0.05, 0.1, 0.2, 0.5):
        n, s, reg, _ = simulate(choose_greedy, p, T, reps, rng, eps=eps)
        picks = (np.argmax(s / np.maximum(n, 1), axis=1) == p.argmax()).mean()
        print(f"  eps = {eps:<4}: regret {reg.mean():6.1f}, picks best {picks:.2f}, "
              f"stuck on a non-best arm (<30% to best) {(n[:, -1] / T < 0.3).mean():.2f}")

    print("\nposterior P(best) at the end of one Thompson experiment:")
    n, s, _, _ = simulate(choose_thompson, p, T, 1, rng)
    pb = prob_best(s[0], n[0] - s[0], rng)
    for k in range(len(p)):
        print(f"  arm {k}: n = {n[0, k]:6.0f}, rate {s[0, k] / max(n[0, k], 1):.4f} (true {p[k]}), P(best) {pb[k]:.3f}")


if __name__ == "__main__":
    demo()
