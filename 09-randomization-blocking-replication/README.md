# 09 Counter-measures against other variables: randomization, blocking, replication, covariates

**STAT 404 ch. 10** · [back to the graph](../README.md)

**What is learned:** how unknown and known nuisance variables are neutralized by design, and how to do
inference that relies on nothing but the randomization that was actually carried out.

## Chapters

| Notes | Covers |
| --- | --- |
| 10.1–10.2 | experimental vs observational studies; lurking variables and confounding |
| 10.3 | blocking known variables |
| 10.4 | randomization balances unknown variables; randomization-based inference (matched pairs, a telephone-fault example; randomization test vs t test) |
| 10.5 | replication |
| 10.6 | auxiliary explanatory variables (covariates): blocking on them, change scores, or analysis of covariance |

## The ideas

### Randomization balances unknown variables

Let `z` be any unit property. Under complete randomization with `n_rep` units per arm, the imbalance
`z̄_A − z̄_B = Σ Wᵢzᵢ / n_rep` (`Wᵢ = ±1`) has mean 0 and variance `4s_z²/n`. No single randomization is
perfectly balanced, but the imbalance is a chance quantity whose size is known, and it shrinks with `n`.
Blocking on a known `z` removes it instead of averaging it out.

### Randomization inference (Fisher)

Under the **strong (sharp) null** that each unit's response would be identical under either treatment, the
observed responses are fixed numbers and only the labels are random. The reference distribution of any
statistic is obtained by recomputing it for every possible allocation:

- matched pairs: all `2^m` sign patterns of the within-pair differences;
- CRD: all `C(n, n_A)` splits (or a Monte Carlo sample).

The p-value is the fraction of allocations at least as extreme as the observed one. No normality assumption
is needed; the familiar t and F tests turn out to be close approximations.

### Replication

`Var(effect estimate) ∝ 1/n_rep`, and replication is also what estimates `σ²` at all.

### Auxiliary variables

A covariate `z` (age, baseline measurement, raw-material property) can be used in three ways: block on it,
analyse `y − z`, or include it in the model (**analysis of covariance**). It must not be affected by the
treatment. The notes stop here with a model statement; [12](../12-variance-reduction-ancova-cuped-cupac/)
develops ANCOVA and its online-experiment descendants CUPED and CUPAC.

## Code in this folder

[`randomization.py`](randomization.py): imbalance variance check, exact paired randomization test, Monte
Carlo two-sample permutation test, blocked vs completely randomized precision, effect of replication.

```bash
python 09-randomization-blocking-replication/randomization.py
```

| Result (saved run) | |
| --- | --- |
| z̄_A − z̄_B over 20 000 randomizations, n = 20 | mean −0.10, variance 26.7 (theory 27.0) |
| 8 matched pairs: exact randomization p (256 patterns) vs paired t | 0.070 vs 0.085 |
| 12 + 12 CRD: permutation p vs two-sample t | 0.604 vs 0.601 |
| sd of the effect estimate, 16 units with strong unit effects: paired on z vs CRD | **0.65** vs 2.58 |
| replication: sd of estimate at n_rep = 5, 10, 20, 40 | 1.89, 1.33, 0.94, 0.68 (theory 1.90, 1.34, 0.95, 0.67) |

## Knowledge gap

**What the notes leave out**

- **Weak vs strong null.** The randomization test is exact for the sharp null; for the weak null (average
  effect zero, individual effects vary) it can be anti-conservative unless a studentized statistic is used.
  Wu & Ding (2021), *JASA*.
- **Neyman's repeated-sampling view**: the difference in means is unbiased for the average effect with a
  conservative variance estimate, without any model. Imbens & Rubin ch. 6.
- **Rerandomization** (Morgan & Rubin 2012): reject allocations with poor covariate balance and adjust the
  inference accordingly. A design-time alternative to ANCOVA.
- **Interference** (one unit's treatment affecting another's response): marketplaces, social networks,
  vaccines. Breaks every argument in this chapter.

**Prerequisites to shore up**

- Combinatorics of allocations; expectation of a sum of dependent indicator variables.

**Open questions / further study**

- Confidence intervals by inverting randomization tests (Hodges–Lehmann). Rosenbaum, *Observational Studies*, ch. 2.

## Connections

- ← [00 Introduction](../00-introduction-and-principles/): the same principles, simulated.
- → [12 ANCOVA, CUPED, CUPAC](../12-variance-reduction-ancova-cuped-cupac/): covariate adjustment in depth.
- → [10 Random effects](../10-random-effects/): what changes when blocks are a sample.
- ← [02 CRD](../02-completely-randomized-designs/): the F test as an approximation to a randomization test.
