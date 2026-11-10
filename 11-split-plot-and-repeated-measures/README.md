# 11 Split-plot and repeated-measures experiments

**STAT 404 ch. 12** · [back to the graph](../README.md)

**What is learned:** effects of a hard-to-change factor applied to large units (whole plots) and an easy-to-
change factor applied to sub-units within them, each tested against its own error term.

## Chapters

| Notes | Covers |
| --- | --- |
| 12.1 | two sizes of experimental unit; a tree-regeneration example |
| 12.2 | whole-plot and sub-plot factors, the regression model, two variance components when estimating contrasts, reasons to split plots, and "a tale of two residual mean squares" (a grassland-stability field experiment) |
| 12.3 | repeated-measures designs as split plots |

## The model

### Design

In each block, randomize the levels of the whole-plot factor W to the whole plots (a large field strip, an
oven run, a subject). Split each whole plot into sub-plots and randomize the sub-plot factor S within it.
There are two randomizations, so there are two kinds of experimental unit and two error terms.

### Model

`Y = μ + block + W + η(whole plot) + S + W×S + E`, with whole-plot error `η ~ N(0, σ_W²)` and sub-plot
error `E ~ N(0, σ²)`.

### ANOVA in two strata

| Stratum | Source | DF | Tested against |
| --- | --- | --- | --- |
| whole plot | Blocks | b − 1 | |
| | W | w − 1 | whole-plot error, E[MS] = σ² + sσ_W² |
| | whole-plot error (Block × W) | (b − 1)(w − 1) | |
| sub plot | S | s − 1 | sub-plot error, E[MS] = σ² |
| | W × S | (w − 1)(s − 1) | sub-plot error |
| | sub-plot error | w(b − 1)(s − 1) | |

S and W × S are compared *within* whole plots and are estimated precisely; W is compared between whole plots
with few DF. Using the sub-plot error for W (analysing the data as an ordinary factorial) makes W look far more
significant than it is.

### Why do it anyway

Some factors are physically hard or expensive to reset (furnace temperature, irrigation level, a field
treatment); splitting is how factorial information is still obtained. The price is less precision on W.

### Repeated measures

A subject is a whole plot; the between-subject treatment is W; time is S. The split-plot analysis assumes
equal correlation between any two times (compound symmetry). Measurements closer in time are usually more
correlated, which needs a sphericity correction or a mixed model with a time-series covariance.

## Code in this folder

[`split_plot.py`](split_plot.py): simulation of a split plot in blocks, the two-stratum ANOVA, and a type I
error comparison against the naive single-error analysis.

```bash
python 11-split-plot-and-repeated-measures/split_plot.py
```

| Result (saved run: 4 blocks × 3 W levels × 4 S levels, σ_W = 1.5, σ = 0.5) | |
| --- | --- |
| F for W (vs whole-plot error, 6 DF) | 3.76, p = 0.087 |
| F for S and W×S (vs sub-plot error, 27 DF) | 39.4 (p = 5e-10) and 2.33 (p = 0.061) |
| type I error for W at 5%, no W effect | split-plot analysis **0.048**, naive analysis **0.542** |

## Knowledge gap

**What the notes leave out**

- **Mixed-model formulation.** `y ~ W * S + (1 | block) + (1 | block:W)` in `lme4` gives the same tests for
  balanced data and also handles imbalance and missing sub-plots.
- **Split-split plots and strip plots** (two factors applied in crossed strips).
- **Sphericity tests and corrections** (Mauchly; Greenhouse–Geisser, Huynh–Feldt) and modern alternatives
  (GEE, mixed models with AR(1) covariance) for repeated measures.
- **Split-plot structure in industrial experiments** run with hard-to-change factors, often unrecognized:
  Jones & Nachtsheim (2009), *J. Quality Technology*.

**Prerequisites to shore up**

- EMS and variance components ([10](../10-random-effects/)).

**Open questions / further study**

- Online experiments randomized by *cluster* (city, store) with user-level outcomes are split plots in
  disguise: the cluster is the whole plot, and analysing at the user level makes the same error as the naive
  analysis here.

## Connections

- ← [10 Random effects](../10-random-effects/): EMS pick the denominators.
- ← [04 Factorials](../04-factorial-experiments/): the same treatment structure with a restricted randomization.
- → [12 CUPED](../12-variance-reduction-ancova-cuped-cupac/): cluster-randomized A/B tests need the whole-plot error,
  and covariates help there too.
