# 10 Fixed versus random effects

**STAT 404 ch. 11** · [back to the graph](../README.md)

**What is learned:** variance components (`σ_B²` between blocks, `σ²` within) in addition to treatment
effects, and which variance belongs in a standard error depending on what the inference is about.

## Chapters

| Notes | Covers |
| --- | --- |
| 11.1 | fixed vs random factors |
| 11.2 | expected mean squares (EMS) |
| 11.3 | F tests built from EMS |
| 11.4 | variance components |
| 11.5 | prediction: fixed effects, random block effects, inference |
| 11.7 | appendix: deriving the EMS for an RBD |

## The models

### Fixed or random?

A factor is **fixed** when its levels are the ones you care about (these 4 fertilizers) and **random** when
its levels are a sample from a population you want to generalize to (these 6 batches out of all batches).
Blocks are usually random.

### Expected mean squares (RBD with random blocks)

`Yᵢⱼ = μ + τᵢ + Bⱼ + Eᵢⱼ`, `Bⱼ ~ N(0, σ_B²)`:

| Source | DF | E[MS] |
| --- | --- | --- |
| Treatments | t − 1 | σ² + b Στᵢ²/(t − 1) |
| Blocks | b − 1 | σ² + t σ_B² |
| Residual | (t − 1)(b − 1) | σ² |

An F test divides two MS whose expectations differ only by the term being tested; here the treatment F test
is unchanged from the fixed-block analysis. In more complex designs (split plots, [11](../11-split-plot-and-repeated-measures/))
the EMS are what tell you which denominator to use.

### Variance components

Equate MS to their expectations: `σ̂² = MS(Res)`, `σ̂_B² = (MS(Block) − MS(Res))/t` (truncate negative values
at 0). The intraclass correlation `σ_B²/(σ_B² + σ²)` says how much of the variation is between blocks.

### Which variance for prediction?

| Target | Variance of the estimate |
| --- | --- |
| a treatment difference `τᵢ − τₖ` | 2σ²/b (blocks cancel) |
| treatment mean in *these* blocks | σ²/b |
| treatment mean over the *population* of blocks | (σ_B² + σ²)/b |
| a new observation in a new block | σ_B² + σ² (+ estimation variance) |

Using the fixed-block interval for a population-level claim is overconfident when blocks vary a lot.

## Code in this folder

[`random_effects.py`](random_effects.py): RBD mean squares, simulated EMS vs theory, ANOVA variance components,
fixed- vs random-block confidence intervals (with a Satterthwaite DF, not in the notes), and their coverage.

```bash
python 10-random-effects/random_effects.py
```

| Result (saved run: t = 4, b = 6, σ_B = 2, σ = 1) | |
| --- | --- |
| E[MS] simulated vs theory: Trt, Block, Res | 5.37 vs 5.38, 17.03 vs 17.00, 1.00 vs 1.00 |
| one experiment: σ̂², σ̂_B², ICC | 1.05, 3.75, 0.78 (truth 1, 4, 0.80) |
| 95% CI for a treatment mean: these blocks / population of blocks | (10.12, 11.91) / (8.90, 13.13) |
| coverage of the population mean | fixed-block interval **0.63**, random-block interval 0.95 |

## Knowledge gap

**What the notes leave out**

- **REML and mixed-model software.** ANOVA estimators work for balanced data; unbalanced data need restricted
  maximum likelihood (`lme4::lmer`, `statsmodels.MixedLM`). Pinheiro & Bates, *Mixed-Effects Models in S and
  S-PLUS*.
- **Satterthwaite / Kenward–Roger degrees of freedom** for tests whose denominator is a combination of mean
  squares.
- **Confidence intervals for variance components** (Graybill–Wang, profile likelihood, parametric bootstrap).
- **Shrinkage (BLUPs)** of individual block effects toward 0, the link to empirical Bayes.

**Prerequisites to shore up**

- Variance of sums with shared random terms; the χ² distribution of mean squares.

**Open questions / further study**

- Random treatment × block interactions: what do they do to the treatment F test?
- In A/B testing, users are a random sample, but so are *days*. When does day-to-day variation need its own
  variance component? (Long-running experiments, novelty effects.)

## Connections

- ← [03 Randomized blocks](../03-randomized-blocks-and-two-way/): same design, blocks now random.
- → [11 Split-plot](../11-split-plot-and-repeated-measures/): two random error terms, two strata.
