# 01 Linear models, least squares and the ANOVA table

**STAT 404 ch. 2** · [back to the graph](../README.md)

**What is learned:** a coefficient vector `β` in `Y = Xβ + E`, plus an estimate of the noise variance `σ²`,
so that every later design (CRD, blocks, factorials, Latin squares) can be analysed as a choice of `X`.

## Chapters

| Notes | Covers |
| --- | --- |
| 2.1–2.3 | the linear model; polynomial and interaction terms as extra columns |
| 2.4 | least-squares estimation |
| 2.5 | ANOVA tables, uncorrected and corrected for the intercept |
| 2.6 | assumptions, estimating σ², t tests and CIs for one βⱼ, F test of all slopes, F test of some parameters |
| 2.7 | prediction: CI for the mean response and PI for a new observation |
| 2.8 | model checking: Box-Cox transformation, residual analysis |

## The models

### The linear model

`Yᵢ = β₀ + β₁xᵢ₁ + … + β_{k−1}xᵢ,ₖ₋₁ + Eᵢ`, with `Eᵢ` independent `N(0, σ²)`. "Linear" means linear in `β`:
`x²` and `x₁x₂` columns are allowed, and they are how curvature and **interaction** (the effect of `x₁`
depends on `x₂`) enter.

### Least squares

Minimize `‖y − Xβ‖²`: `β̂ = (XᵀX)⁻¹Xᵀy`, fitted values `ŷ = Hy` with the hat matrix `H = X(XᵀX)⁻¹Xᵀ`,
residuals `r = y − ŷ`. Geometrically `ŷ` is the orthogonal projection of `y` onto the column space of `X`,
which is why sums of squares add up.

### ANOVA

`SS(Total) = SS(Regression) + SS(Residual)` with `n − 1 = (k − 1) + (n − k)` degrees of freedom (corrected for
the intercept). `s² = SS(Residual)/(n − k)` estimates `σ²`. The overall F statistic
`MS(Reg)/MS(Res) ~ F(k − 1, n − k)` tests that all slopes are zero.

### Tests and intervals

- One coefficient: `t = β̂ⱼ / se(β̂ⱼ)`, `se = s √[(XᵀX)⁻¹]ⱼⱼ`, on `n − k` DF.
- Some coefficients (the **extra sum of squares** principle): fit the full and the reduced model, then
  `F = [(SS_res,red − SS_res,full)/q] / s²_full`. Every ANOVA line in later folders is one of these tests.
- Prediction at `x₀`: the mean `x₀ᵀβ` has variance `σ² x₀ᵀ(XᵀX)⁻¹x₀`; a new `Y` adds `σ²`, so the
  prediction interval never shrinks to zero width.

### Box-Cox transformation and residual checks

When the spread grows with the mean, or effects are multiplicative, analyse `y^λ` instead of `y`. Box-Cox
picks `λ` by maximizing the profile likelihood, using the scaled transform
`(y^λ − 1)/(λ ẏ^{λ−1})` (`ẏ` the geometric mean) so residual sums of squares are comparable across `λ`.
`λ = 1` is no transform, `½` the square root, `0` the log, `−1` the reciprocal. Residual plots (vs fitted, vs
each `x`, normal Q-Q, vs run order) check what the transform could not fix.

## Code in this folder

[`linear_model.py`](linear_model.py): QR-based least squares with standard errors, ANOVA table, extra-SS F
test, CI and PI for prediction, Box-Cox profile with an approximate 95% interval for `λ`.

```bash
python 01-linear-models/linear_model.py
```

| Result (saved run) | |
| --- | --- |
| `y = 10 + 2x₁ − x₂ + 1.5x₁x₂ + noise`, n = 40: β̂ for `x₁`, `x₂`, `x₁x₂`, `x₁²` | 2.21, −1.02, 1.87, −0.13 (p = 0.53) |
| overall F, R² | 213 on (4, 35) DF, 0.961 |
| extra-SS F test of `x₁x₂` and `x₁²` together | F = 44.0 on 2 DF |
| at `x₁ = x₂ = 0.5`: mean, 95% CI, 95% PI | 11.07, (10.87, 11.27), (10.26, 11.88) |
| Box-Cox on log-normal data | λ̂ = 0.20, 95% CI (−0.10, 0.50), contains the log |

The tests check every number against `statsmodels`.

## Knowledge gap

**What the notes leave out**

- **Why the F and t distributions hold.** Cochran's theorem: under normality, `SS(Res)/σ²` is χ² on `n − k`
  DF and independent of `β̂`. The notes state the results; Seber & Lee, *Linear Regression Analysis*, ch. 3–4
  prove them.
- **Robustness of the tests.** The F test for means is fairly robust to non-normality with balanced designs,
  much less to unequal variances with unequal group sizes. Welch-type tests and HC (sandwich) standard
  errors are the modern fix; HC2 is used in [12](../12-variance-reduction-ancova-cuped-cupac/ancova.py).
- **Multiple testing.** Testing many coefficients at 5% each inflates false positives; Bonferroni, Holm,
  Tukey HSD and FDR control are not covered.
- **Leverage and influence** (hat values, Cook's distance) are only touched through residual plots.

**Prerequisites to shore up**

- Matrix algebra: rank, inverse, projection matrices, `tr(H) = k`.
- The normal, χ², t and F distributions and how they are built from each other.

**Open questions / further study**

- How do generalized linear models (counts, proportions) change the ANOVA picture? McCullagh & Nelder, ch. 2.
- Reading: Montgomery, *Design and Analysis of Experiments*, ch. 10 (regression for designed experiments).

## Connections

- → [02 CRD](../02-completely-randomized-designs/): one categorical factor coded by dummy columns.
- → [04 Factorials](../04-factorial-experiments/) and [05 2-level factorials](../05-two-level-factorials/): designs
  that make `XᵀX` diagonal, so every effect is estimated independently.
- → [12 ANCOVA](../12-variance-reduction-ancova-cuped-cupac/): one more column, a covariate, in the same model.
