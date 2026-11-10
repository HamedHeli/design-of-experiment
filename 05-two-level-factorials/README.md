# 05 2-level factorial experiments

**STAT 404 ch. 6** · [back to the graph](../README.md)

**What is learned:** with every factor at two levels coded −1 / +1, one coefficient per main effect and per
interaction, each estimated by a simple signed average, all independent and all with the same precision.

## Chapters

| Notes | Covers |
| --- | --- |
| 6.1–6.2 | why 2 levels: many factors, few runs |
| 6.3 | the 2³: regression on orthogonal contrasts, ANOVA, presenting conclusions, 2³ vs 3³ for the same process |
| 6.4 | graphical identification of significant effects: normal and half-normal plots |
| 6.5 | 2^p × n_rep in general: models for unreplicated and replicated designs, estimates, standard errors, SS |
| 6.6 | prediction at a chosen setting |
| 6.7 | interpreting effects as treatment contrasts |
| 6.8 | one factor at a time, and why it is inferior |

## The models

### The contrast matrix

All `2^p` combinations of ±1. Products of columns give interaction columns; all `2^p − 1` columns plus the
intercept are mutually orthogonal (`XᵀX = nI`). So

`β̂_S = x_Sᵀy / n`, `Var(β̂_S) = σ²/n`, `SS(S) = n β̂_S²`.

The classical "effect" (mean at + minus mean at −) is `2β_S`. Every estimate uses every run: a 2⁴ in 16 runs
estimates each main effect as precisely as a two-group comparison with 8 per group, *and* gets all
interactions for free ("hidden replication").

### Unreplicated designs and half-normal plots

With `n = 2^p` runs and `2^p` parameters there are no DF for error. **Effect sparsity**: usually only a few
effects are real. The inert estimates are `N(0, σ²/n)` noise, so a half-normal plot of the ordered `|β̂|`
against `Φ⁻¹(0.5 + 0.5(i − 0.5)/m)` is a straight line through the origin for them; the real effects stand
off to the right. An alternative used in the notes: assume all 3-factor and higher interactions are zero and
pool them as error.

### Replicated designs

With `n_rep ≥ 2`, pure error has `2^p(n_rep − 1)` DF and `se(β̂) = s/√n` for every effect.

### Prediction and interpretation

The predicted mean at a setting is `β̂₀ + Σ β̂_S x_S` using the significant terms (respecting hierarchy). An
interaction `12` means: the effect of factor 1 is `2(β₁ + β₁₂x₂)`, different at the two levels of factor 2.

### One factor at a time (OFAT)

Start at a base point and change each factor in turn. With the same budget, each OFAT effect uses only a
fraction of the runs, so it has larger variance, and OFAT cannot estimate any interaction.

## Code in this folder

[`two_level.py`](two_level.py): full factorial and contrast matrix (any interaction order), effect estimates,
half-normal plotting positions, Lenth's pseudo standard error, replicated-design standard errors, and an OFAT
vs factorial simulation.

```bash
python 05-two-level-factorials/two_level.py
```

| Result (saved run) | |
| --- | --- |
| unreplicated 2⁴, truth β₁ = 3, β₃ = −2, β₁₃ = 1.5, σ = 1 | largest \|β̂\|: 1 (2.72), 3 (2.39), 13 (1.82); next 23 (0.51) |
| Lenth PSE, margin of error, active effects | 0.30 (truth 0.25), 0.77, {1, 3, 13} |
| replicated 2³ × 2 | se = 0.30 on 8 DF; significant: 1, 23 (the true effects) |
| OFAT vs 2³, 8 runs, Var(β̂₁) | 0.251 vs **0.123** (theory 0.25 vs 0.125) |

## Knowledge gap

**What the notes leave out**

- **Formal tests for unreplicated designs.** Lenth's method (1989, *Technometrics*), implemented here, turns
  the half-normal plot into a rule; Box-Meyer's Bayesian posterior probability of activity is another.
- **Centre points.** Adding a few runs at the centre of a quantitative 2^p checks for curvature and gives pure
  error without replicating the corners.
- **Effect hierarchy / heredity** as formal model-selection rules.
- **Definitive screening designs** (Jones & Nachtsheim 2011): 3-level designs in about `2p + 1` runs that
  keep main effects clear of 2-factor interactions and detect curvature.

**Prerequisites to shore up**

- Orthogonal designs and why `XᵀX = nI` makes estimates independent.
- Order statistics and Q-Q plots.

**Open questions / further study**

- How should one choose factor ranges? Too narrow and nothing is significant; too wide and the response is
  not linear over the range.
- Reading: Box, Hunter & Hunter ch. 5; Wu & Hamada ch. 4.

## Connections

- → [06 Blocking](../06-blocking-two-level-factorials/): spread a 2^p over blocks by sacrificing high-order effects.
- → [07 Fractions](../07-fractional-factorials/): run only some of the 2^p combinations.
- ← [04 Factorials](../04-factorial-experiments/): the general case with more levels.
- → [12 CUPED](../12-variance-reduction-ancova-cuped-cupac/): a 2-arm A/B test is a 2¹ factorial; covariates add
  precision the same way extra factors cannot.
