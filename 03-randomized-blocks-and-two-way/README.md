# 03 Randomized block designs and 2-way factorials

**STAT 404 ch. 4** · [back to the graph](../README.md)

**What is learned:** treatment effects `τᵢ` *and* block effects `βⱼ`, so that known unit-to-unit variation
(days, batches, litters, locations) is removed from the treatment comparison instead of inflating the error.

## Chapters

| Notes | Covers |
| --- | --- |
| 4.1–4.3 | the randomized block design (RBD); randomizing independently within each block |
| 4.4–4.6 | the additive regression model, its ANOVA, and why balance makes the SS unique |
| 4.7 | model checking, including Tukey's 1-DF test for non-additivity |
| 4.8 | contrasts in an RBD |
| 4.9 | the 2-way factorial with replication: interaction becomes estimable |
| 4.10 | R: plotting and `aov(y ~ trt + block)` |

## The models

### Randomized block design

`b` blocks of `t` similar units. In every block, each treatment appears once, in a freshly randomized order.
Model `Yᵢⱼ = μ + τᵢ + βⱼ + Eᵢⱼ` (additive: the treatment effect is the same in every block).

### ANOVA and balance

`SS(Trt) = bΣ(ȳᵢ. − ȳ..)²`, `SS(Block) = tΣ(ȳ.ⱼ − ȳ..)²`, `SS(Res)` on `(t − 1)(b − 1)` DF. Because every
treatment appears in every block exactly once, treatment and block columns are orthogonal: the treatment SS
is the same whether blocks are fitted first or not, and treatment means are unaffected by block effects.

### How much did blocking buy?

Relative efficiency estimated from the RBD's own ANOVA:
`RE = [(b − 1)MS(Block) + b(t − 1)MS(Res)] / [(bt − 1)MS(Res)]`: the factor by which a CRD on the same units
would need more replication for the same precision. Blocking costs `b − 1` error DF, so it pays only when
blocks really differ.

### Non-additivity

If the treatment effect changes across blocks (often because effects are multiplicative), the residual
contains it. Tukey's test fits one extra term `γ·τᵢ·βⱼ` (1 DF). Rejection usually means "analyse on the log
scale".

### 2-way factorial

Two *treatment* factors A (a levels) and B (b levels), all `ab` combinations, `r ≥ 2` replicates each:
`Y = μ + αᵢ + βⱼ + (αβ)ᵢⱼ + E`. With `r = 1` the interaction and the error are the same thing, which is
exactly the RBD's situation; replication is what separates them.

## Code in this folder

[`rbd.py`](rbd.py): within-block randomization, RBD ANOVA, relative efficiency, Tukey's non-additivity test,
replicated 2-way ANOVA.

```bash
python 03-randomized-blocks-and-two-way/rbd.py
```

| Result (saved run: 5 treatments, 7 blocks with sd 3, noise sd 0.5) | |
| --- | --- |
| RBD F for treatments | 11.6 on (4, 24) DF, p = 2.2e-5 |
| same data analysed as a CRD | F = 0.42, p = 0.79: the effect is invisible |
| relative efficiency of blocking | 24.6× |
| Tukey non-additivity: additive data / multiplicative data / its log | p = 0.73 / 7.7e-14 / 0.12 |
| replicated 3 × 4 factorial with one interacting cell, AB test | F = 3.6, p = 0.011 |

## Knowledge gap

**What the notes leave out**

- **Incomplete blocks.** When a block holds fewer units than there are treatments, balanced incomplete
  block designs (BIBDs) keep every pair of treatments together equally often. Analysis needs the intra-block
  estimates (adjusted treatment means). Montgomery ch. 4.4; Cox & Reid, *The Theory of the Design of
  Experiments*, ch. 4.
- **Missing observations** destroy balance; the treatment SS must then be computed as an extra SS
  ([01](../01-linear-models/)).
- **Random vs fixed blocks** matters for what a confidence interval for a treatment mean covers; that is
  [10](../10-random-effects/).

**Prerequisites to shore up**

- Two-factor additive models; orthogonality of effects in balanced designs.

**Open questions / further study**

- In online experiments the analogue of blocking is *stratification* (by country, platform) or post-
  stratification. How does post-stratification compare to blocking at design time? Miratrix, Sekhon & Yu
  (2013), JRSS-B.

## Connections

- → [04 Factorials](../04-factorial-experiments/): more than two factors and their interactions.
- → [06 Blocking 2-level factorials](../06-blocking-two-level-factorials/): when a block cannot hold all treatments.
- → [08 Latin squares](../08-latin-squares/): two blocking factors at once.
- → [10 Random effects](../10-random-effects/): blocks as a random sample.
- → [12 ANCOVA](../12-variance-reduction-ancova-cuped-cupac/): adjust for a continuous covariate instead of blocking on it.
