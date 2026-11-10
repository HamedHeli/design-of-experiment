# Basics: variance, standard deviation, standard error and the variance of a difference

**Foundation for every folder** (the notes use these results throughout, mostly in ch. 2–3 and 10) ·
[back to the graph](../README.md)

**What is learned:** how uncertain an estimate is. Every test in this repo is just
`t = estimate / standard error`, and every design choice (blocking, pairing, replication, CUPED) is a way to
make that standard error smaller.

## Cheat sheet

| Quantity | Formula | Describes | Shrinks with more data? |
| --- | --- | --- | --- |
| variance `σ²` | `E[(Y − μ)²]` | spread of individual observations, in squared units | no |
| standard deviation (SD) `σ` | `√σ²` | the same spread, in the units of `Y` | no |
| sample variance `s²` | `Σ(yᵢ − ȳ)² / (n − 1)` | estimate of `σ²` | no (gets more accurate) |
| standard error of a mean | `σ/√n`, estimated `s/√n` | spread of `ȳ` across repeated experiments | yes, like `1/√n` |
| SE of a difference of means | `√(σ₁²/n₁ + σ₂²/n₂)` | spread of `ȳ₁ − ȳ₂` | yes |
| pooled SD | `s_p = √{[(n₁−1)s₁² + (n₂−1)s₂²]/(n₁+n₂−2)}` | common `σ` when the groups share it | no |
| pooled SE of a difference | `s_p √(1/n₁ + 1/n₂)` | spread of `ȳ₁ − ȳ₂` assuming equal `σ` | yes |

## The ideas

### SD vs SE: the one distinction to never blur

The **SD** answers "how different are individual units?" It is a property of the population and does not go
down when you collect more data. The **SE** answers "if I repeated the whole experiment, how much would my
*estimate* move?" It is the SD of the sampling distribution of the estimate, and it falls like `1/√n`:
four times the data halves the SE. Error bars on a plot of means should say which one they are; confidence
intervals use the SE.

### Why divide by n − 1

The deviations `yᵢ − ȳ` always sum to 0, so only `n − 1` of them are free: the sample mean has used up one
**degree of freedom**. Dividing by `n` underestimates `σ²` by the factor `(n − 1)/n`. In general, residual
variance divides by `n − (number of estimated means/parameters)`, which is where every "residual DF" in an
ANOVA table comes from.

### Variance rules

- `Var(aX + b) = a²Var(X)`: shifting does nothing, scaling squares.
- `Var(X ± Y) = Var(X) + Var(Y) ± 2Cov(X, Y)`.
- **Independent:** `Var(X − Y) = Var(X) + Var(Y)`. Variances add even when you subtract. With SDs 3 and 4 the
  difference has SD 5, not 1 and not 7. SDs never add; variances do.
- **Positively correlated:** subtracting removes the shared part, `Var(X − Y)` can be far smaller than either.
  This single fact is why pairing, blocking, ANCOVA and CUPED work.
- A contrast of independent group means: `Var(Σcᵢȳᵢ) = Σcᵢ²σ²/nᵢ`.

### Difference of two independent means: pooled or Welch?

`Var(ȳ₁ − ȳ₂) = σ₁²/n₁ + σ₂²/n₂`. Two ways to estimate it:

- **Pooled** (Student's t): assume `σ₁ = σ₂`, average the two `s²` weighted by their DF, use
  `s_p√(1/n₁ + 1/n₂)` on `n₁ + n₂ − 2` DF. With `t` groups, `s_p²` is exactly the ANOVA's **MS(Residual)**,
  which is why every ANOVA-based SE in [02](../02-completely-randomized-designs/) onward is a pooled SE.
- **Welch**: keep the variances separate, `√(s₁²/n₁ + s₂²/n₂)`, with the Welch–Satterthwaite DF
  `(v₁ + v₂)² / [v₁²/(n₁−1) + v₂²/(n₂−1)]`, `vᵢ = sᵢ²/nᵢ`.

With equal sample sizes the two SEs are identical and the choice barely matters. With unequal sizes *and*
unequal SDs, pooling is dangerous when the small group has the large SD (see the results: 35% false positives
at a nominal 5%). Welch is the safer default for two groups; pooling is what lets ANOVA and contrasts borrow
strength across many groups when the equal-variance assumption is reasonable (check it, or transform:
[01 Box-Cox](../01-linear-models/)).

### Paired differences

Measuring the same unit twice and analysing `dᵢ = y_B − y_A` gives `Var(d̄) = (σ_A² + σ_B² − 2Cov)/m`.
The unit-to-unit variation sits in the covariance and cancels. Analysing paired data as two independent
samples throws that away (in the demo the SE is 4× too large and a real effect becomes p = 0.56). Pairing is
the simplest block design ([03](../03-randomized-blocks-and-two-way/)).

### Proportions

For a 0/1 outcome, `Var(Y) = p(1 − p)`, so `SE(p̂) = √(p(1 − p)/n)`; it is largest at `p = ½`. For the
difference of two rates, the confidence interval uses the unpooled SE; the z test of `H₀: p₁ = p₂` uses the
pooled rate `p̂ = (x₁ + x₂)/(n₁ + n₂)`, because under `H₀` both groups share it.

### Percentage lift (delta method)

A/B results are often reported as `ȳ_T/ȳ_C − 1`. A ratio's variance comes from a first-order Taylor expansion:
`Var ≈ Var(ȳ_T)/μ_C² + μ_T² Var(ȳ_C)/μ_C⁴`. It matches the bootstrap closely at realistic sample sizes.

### From SE to everything else

- 95% CI: `estimate ± t_{0.975, DF} × SE` (≈ ± 2 SE).
- Test: `t = estimate/SE`.
- Sample size per arm for a difference `δ` with 80% power at 5%: `n ≈ 2(1.96 + 0.84)²σ²/δ² ≈ 16σ²/δ²`.
- Halving the SE needs 4× the data, or a design that removes variance (blocking, pairing, CUPED).

## Code in this folder

[`basics.py`](basics.py): sample variance, SE of a mean, pooled and Welch two-sample t, paired t, contrast SE
(pooled or not), two-proportion CI and z test, delta-method SE of a percentage lift, each checked by
simulation.

```bash
python basics-variance-and-standard-errors/basics.py
```

| Result (saved run) | |
| --- | --- |
| σ = 2, n = 25: SD of ȳ over 20 000 samples | 0.398 = σ/√n = 0.400 |
| mean of s² dividing by n − 1 / by n | 4.001 / 3.840 (σ² = 4; (n−1)/n σ² = 3.84) |
| one sample with n = 25, 100, 400: SD / SE | 2.05 / 0.41, 2.14 / 0.21, 2.06 / 0.10: SD stays put, SE halves per 4× n |
| Var(X − Y), independent, SDs 3 and 4 | 25.0 (SD 5) |
| Var(X − Y), correlation 0.8 | 5.8 |
| equal SDs, 15 + 15: pooled vs Welch SE | 0.750 vs 0.750 (28 vs 27.7 DF) |
| no true difference, n = 8 (SD 4) vs n = 40 (SD 1): type I error | pooled **0.348**, Welch 0.052 |
| 12 units measured twice: paired vs "independent" SE | 0.37 (p = 0.036) vs 1.48 (p = 0.56) |
| conversion 5.6% vs 5.0%, 10 000 each | diff 0.0060, SE 0.0032, z = 1.89, p = 0.058 |
| revenue lift 6.0%: delta-method vs bootstrap SE | 2.12 vs 2.13 percentage points |

The tests check the t tests against `scipy.stats`, and that the pooled contrast SE equals the one from the
ANOVA in [02](../02-completely-randomized-designs/).

## Knowledge gap

**What this folder leaves out**

- **Unequal variance in ANOVA** beyond two groups: Welch's ANOVA, Games–Howell comparisons.
- **Clustered data.** If units are grouped (users within cities) and randomized by group, the SE must account
  for the intra-cluster correlation: design effect `1 + (m − 1)ρ`. Ignoring it is the split-plot mistake of
  [11](../11-split-plot-and-repeated-measures/).
- **Heavy tails.** The SE formulas are fine, but the normal approximation for the CI can be slow to kick in for
  revenue-like metrics; trimming, winsorizing or the bootstrap help.
- **Standard error of a standard deviation** (≈ `σ/√(2(n−1))` for normal data), useful in variance-component
  work ([10](../10-random-effects/)).

**Prerequisites to shore up**

- Expectation and variance of linear combinations; covariance and correlation.
- The central limit theorem (why `ȳ` is approximately normal).

**Open questions / further study**

- When does the bootstrap beat the delta method, and vice versa? Efron & Tibshirani, *An Introduction to the
  Bootstrap*, ch. 6 and 21.
- Reading: Wasserman, *All of Statistics*, ch. 3, 5 and 6; Kohavi, Tang & Xu, *Trustworthy Online Controlled
  Experiments*, ch. 18 (variance estimation pitfalls).

## Connections

- → [01 Linear models](../01-linear-models/): `se(β̂ⱼ) = s√[(XᵀX)⁻¹]ⱼⱼ` is the same idea in matrix form.
- → [02 CRD](../02-completely-randomized-designs/): pooled variance = MS(Residual); contrast SEs.
- → [03 Randomized blocks](../03-randomized-blocks-and-two-way/) and [09](../09-randomization-blocking-replication/):
  pairing and blocking are the covariance term at work.
- → [12 ANCOVA, CUPED, CUPAC](../12-variance-reduction-ancova-cuped-cupac/): `Var(Y − θX) = Var(Y)(1 − ρ²)` is
  the variance rule with the optimal `θ`.
