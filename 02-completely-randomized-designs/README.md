# 02 Completely randomized designs, one-way ANOVA and contrasts

**STAT 404 ch. 3** · [back to the graph](../README.md)

**What is learned:** one mean per treatment, `μ₁, …, μ_t`, and specific comparisons among them (contrasts),
from units allocated to treatments completely at random.

## Chapters

| Notes | Covers |
| --- | --- |
| 3.1 | the CRD; worked examples on solder pull strength vs thickness and on animal behaviour under five playback sounds |
| 3.2–3.3 | ANOVA and the F test of equal treatment means |
| 3.4 | three regression codings: t dummies, intercept + t−1 dummies, intercept + t−1 contrasts |
| 3.5 | model checking, Box-Cox (a square-root transform fixes the behaviour data) |
| 3.6 | treatment contrasts, orthogonal contrasts, orthogonal polynomial contrasts |
| 3.7 | doing it in R: randomization, plots, `aov`, contrasts |

## The models

### The CRD

`t` treatments, `nᵢ` units each, `n = Σnᵢ`. Randomly permute the `n` treatment labels over the units. Model
`Yᵢⱼ = μ + τᵢ + Eᵢⱼ`. The only thing distinguishing units of different treatments is the random allocation,
so differences in means estimate treatment effects.

### One-way ANOVA

`SS(Trt) = Σ nᵢ(ȳᵢ − ȳ)²` on `t − 1` DF and `SS(Res) = Σ (yᵢⱼ − ȳᵢ)²` on `n − t` DF; `F = MS(Trt)/MS(Res)`.
`MS(Res)` is the pooled within-treatment variance.

### Three codings, one fit

| Coding | Columns | β means |
| --- | --- | --- |
| cell means | t dummies, no intercept | `βᵢ = μᵢ` |
| reference | intercept + t−1 dummies | `β₀ = μ₁`, `βᵢ = μᵢ − μ₁` (R's default `contr.treatment`) |
| sum-to-zero contrasts | intercept + t−1 contrast columns | `β₀` = average of the `μᵢ`, the rest are deviations |

All three span the same column space, so fitted values, residuals and the ANOVA are identical. Only the
interpretation of individual β's changes, which is why a t test on "the" coefficient means nothing until
the coding is fixed.

### Contrasts

A **contrast** is `κ = Σcᵢμᵢ` with `Σcᵢ = 0`: "extinct vs extant predators", "low vs the average of the others".
Its estimate `Σcᵢȳᵢ` has variance `σ²Σcᵢ²/nᵢ`, and its 1-DF sum of squares is `(Σcᵢȳᵢ)²/Σ(cᵢ²/nᵢ)`.
Two contrasts are **orthogonal** when `Σcᵢdᵢ/nᵢ = 0`; a full set of `t − 1` orthogonal contrasts splits
`SS(Trt)` into `t − 1` independent 1-DF pieces.

### Orthogonal polynomial contrasts

For a quantitative factor at equally spaced levels with equal replication, the linear, quadratic, cubic …
contrasts (`−2, −1, 0, 1, 2`; `2, −1, −2, −1, 2`; …) answer "is the response curved?" one degree at a time.

## Code in this folder

[`crd.py`](crd.py): randomization, one-way ANOVA, the three design matrices, contrast estimates with CIs and
1-DF SS, orthogonal polynomial contrasts from a QR of the Vandermonde matrix (works for unequal spacing too).

```bash
python 02-completely-randomized-designs/crd.py
```

| Result (saved run: 5 equally spaced levels, 6 units each, curved true means) | |
| --- | --- |
| one-way ANOVA | F = 15.3 on (4, 25) DF, p = 1.8e-6 |
| three codings | identical fitted values (max diff 1.8e-14); intercepts 20.04, 20.04, 23.13 |
| SS split: linear / quadratic / cubic / quartic | 24.9 / **66.7** / 0.05 / 0.88, summing to SS(Trt) = 92.55 |

## Knowledge gap

**What the notes leave out**

- **Post-hoc comparisons.** Contrasts chosen *after* looking at the data need simultaneous protection:
  Tukey's HSD for all pairs, Scheffé for any contrast, Dunnett for many-vs-control.
- **Unequal variances.** Welch's ANOVA and the Games-Howell procedure when the Box-Cox fix is not enough.
- **Sample-size planning for a CRD** from the noncentral F distribution with noncentrality
  `Σnᵢτᵢ²/σ²`.
- **Non-parametric alternatives** (Kruskal-Wallis) and the randomization test of
  [09](../09-randomization-blocking-replication/), which needs no normality.

**Prerequisites to shore up**

- Dummy variables and column spaces ([01](../01-linear-models/)).
- Linear combinations of independent normals.

**Open questions / further study**

- What does "orthogonal" buy you when the design is unbalanced, and why do Type I / II / III sums of squares
  then disagree? Montgomery ch. 3; Fox, *Applied Regression Analysis*, ch. 8.

## Connections

- → [03 Randomized blocks](../03-randomized-blocks-and-two-way/): add a blocking factor when units are not
  homogeneous.
- → [04 Factorials](../04-factorial-experiments/): polynomial contrasts per factor and their products become
  interaction contrasts.
- → [09 Randomization](../09-randomization-blocking-replication/): the F test as an approximation to a
  randomization test.
