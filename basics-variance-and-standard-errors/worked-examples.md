# Worked examples: the basic calculations by hand

[back to Basics](README.md) · every number here is checked by [`worked_examples.py`](worked_examples.py)

Each example uses numbers small enough to do with a pencil. Work them once by hand; after that the formulas
in the rest of the repo are just these steps with bigger `n`.

---

## 1. Mean, variance, SD and SE of one sample

Data **A = 4, 6, 8, 10, 12** (n = 5).

| step | calculation | result |
| --- | --- | --- |
| mean | (4 + 6 + 8 + 10 + 12) / 5 | **ȳ = 8** |
| deviations | 4−8, 6−8, 8−8, 10−8, 12−8 | −4, −2, 0, 2, 4 (they sum to 0) |
| squared deviations | 16, 4, 0, 4, 16 | sum of squares **SS = 40** |
| sample variance | SS / (n − 1) = 40 / 4 | **s² = 10** |
| standard deviation | √10 | **s = 3.162** |
| standard error of the mean | s / √n = 3.162 / √5 | **SE = 1.414** |

- Dividing by `n` instead would give 40/5 = 8, too small: the deviations were measured from `ȳ`, which was
  fitted to these same 5 numbers, so only 4 of them are free.
- **SD = 3.16** says a typical value is about 3 away from 8. **SE = 1.41** says that if you drew another 5
  values, the new mean would typically be about 1.4 away from this one. With 20 values instead of 5 the SD
  would stay near 3.2, the SE would halve to about 0.7.

## 2. The variance of a difference

`X` has SD 3 (variance 9), `Y` has SD 4 (variance 16).

| case | calculation | result |
| --- | --- | --- |
| independent | Var(X − Y) = 9 + 16 | **25**, SD **5** (not 4 − 3 = 1) |
| correlation 0.6 | 9 + 16 − 2 × 0.6 × 3 × 4 = 25 − 14.4 | **10.6**, SD 3.26 |

The minus sign in `X − Y` disappears in the variance (`(−1)² = 1`), so independent noise always *adds*.
Only a shared component (positive correlation) cancels; that is the entire reason for pairing and blocking.

## 3. Two independent groups: pooled vs Welch

Group **A = 4, 6, 8, 10, 12** (from example 1: n = 5, ȳ = 8, s² = 10).
Group **B = 7, 9, 11, 13, 15, 17, 19** (n = 7).

B: mean 13, deviations −6 … 6, SS = 36 + 16 + 4 + 0 + 4 + 16 + 36 = 112, **s² = 112/6 = 18.667**.
Difference **ȳ_B − ȳ_A = 13 − 8 = 5**.

**Pooled (assumes equal σ):**

| step | calculation | result |
| --- | --- | --- |
| pooled variance | [(5−1)·10 + (7−1)·18.667] / (5 + 7 − 2) = (40 + 112) / 10 | **s_p² = 15.2**, s_p = 3.899 |
| SE of the difference | 3.899 × √(1/5 + 1/7) = 3.899 × 0.5855 | **2.283** |
| t, DF | 5 / 2.283 | **t = 2.19** on 10 DF, p = 0.053 |
| 95% CI | 5 ± 2.228 × 2.283 | (−0.09, 10.09) |

The pooled variance is a DF-weighted average of the two s²: it adds the two sums of squares (40 + 112) and
divides by the total residual DF (4 + 6). That is exactly what MS(Residual) does in an ANOVA.

**Welch (no equal-σ assumption):**

| step | calculation | result |
| --- | --- | --- |
| variance of each mean | v_A = 10/5 = 2, v_B = 18.667/7 = 2.667 | |
| SE of the difference | √(2 + 2.667) | **2.160** |
| Welch–Satterthwaite DF | (2 + 2.667)² / [2²/4 + 2.667²/6] = 21.78 / 2.185 | **9.97** |
| t | 5 / 2.160 | **t = 2.32**, p = 0.043, CI (0.18, 9.82) |

Here pooling gives the larger SE because the *smaller* group has the *smaller* variance, so pooling drags the
small group's variance up. In the opposite situation (small group, large variance) pooling understates the SE,
which is the dangerous case shown in the Basics README.

## 4. Paired data: the same people before and after

| person | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| before | 10 | 12 | 9 | 14 | 11 |
| after | 12 | 13 | 11 | 15 | 14 |
| d = after − before | 2 | 1 | 2 | 1 | 3 |

| step | calculation | result |
| --- | --- | --- |
| mean difference | (2 + 1 + 2 + 1 + 3)/5 | **d̄ = 1.8** |
| variance of d | deviations 0.2, −0.8, 0.2, −0.8, 1.2 → SS = 2.8, / 4 | **s_d² = 0.70** |
| SE | √(0.70/5) | **0.374** |
| t | 1.8 / 0.374 | **t = 4.81** on 4 DF, p = 0.009 |

**The wrong way**, treating before and after as two independent groups: s² = 3.7 and 2.5, SE =
√(3.7/5 + 2.5/5) = √1.24 = **1.114**, t = 1.62, p = 0.15. Same data, three times the SE, no effect found.

Why: before and after are correlated at 0.90 (people who start high stay high).
`Var(after − before) = Var(after) + Var(before) − 2Cov` removes that person-to-person variation; the unpaired
SE keeps it.

## 5. A contrast of several means (using the ANOVA's pooled variance)

Three groups of n = 4, means **10, 12, 17**, MS(Residual) = **6** on 9 DF. Question: does group 3 differ from
the average of groups 1 and 2? Contrast `c = (−½, −½, 1)`, coefficients sum to 0.

| step | calculation | result |
| --- | --- | --- |
| estimate | −½·10 − ½·12 + 1·17 = 17 − 11 | **6** |
| variance | MS(Res) × Σcᵢ²/nᵢ = 6 × (0.25/4 + 0.25/4 + 1/4) = 6 × 0.375 | **2.25** |
| SE, t | √2.25 = 1.5; 6 / 1.5 | **t = 4.0** on 9 DF, p = 0.003 |

Same rule as example 2: each mean contributes `cᵢ² × σ²/nᵢ`, and the variances add.

## 6. Two proportions (conversion rates)

Control: 100 conversions out of 1000 (10%). Treatment: 120 out of 1000 (12%). Difference **0.02**.

| step | calculation | result |
| --- | --- | --- |
| SE for the CI (unpooled) | √(0.12·0.88/1000 + 0.10·0.90/1000) = √(0.0001056 + 0.00009) | **0.01399** |
| 95% CI | 0.02 ± 1.96 × 0.01399 | (−0.0074, 0.0474) |
| pooled rate under H₀ | (120 + 100) / 2000 | p̂ = 0.11 |
| SE for the test | √(0.11·0.89 × (1/1000 + 1/1000)) | **0.01399** |
| z | 0.02 / 0.01399 | **z = 1.43**, p = 0.15 |

A two-point lift on a 10% base with 1000 users per arm is not detectable. The sample size formula
`n ≈ 16 p(1−p)/δ² = 16 × 0.11 × 0.89 / 0.02² ≈ 3900` per arm for 80% power.

## 7. Percentage lift (delta method)

Treatment mean **10.5** (SE 0.2), control mean **10.0** (SE 0.2). Lift = 10.5/10.0 − 1 = **5%**.

| step | calculation | result |
| --- | --- | --- |
| variance of the ratio | SE_T²/μ_C² + μ_T² SE_C²/μ_C⁴ = 0.04/100 + 110.25 × 0.04/10 000 = 0.000400 + 0.000441 | **0.000841** |
| SE | √0.000841 | **0.029 = 2.9 percentage points** |

95% CI for the lift: 5% ± 5.7%, so roughly (−0.7%, 10.7%). Note the SE of the lift is *not* just
(SE of the difference)/10 = 0.283/10 = 2.83 points: the noise in the control mean (the denominator) adds the
extra 0.000041.

## 8. Design effect for clustered data

1000 users in 50 cities (m = 20 per city), intra-city correlation ρ = 0.05, treatment assigned by city.

| step | calculation | result |
| --- | --- | --- |
| design effect | 1 + (m − 1)ρ = 1 + 19 × 0.05 | **1.95** |
| effective sample size | 1000 / 1.95 | **≈ 513** independent users |
| SE inflation | √1.95 | the naive SE is **1.40× too small** |

A correlation of only 0.05 almost halves the information because each city's 20 users share it.

## 9. Cluster-robust SE by hand

Four clusters (e.g. cities) of two users each; clusters 1–2 treated, 3–4 control.

| cluster | arm | y | arm mean | residuals y − mean | cluster sum of residuals S_g |
| --- | --- | --- | --- | --- | --- |
| 1 | T | 5, 7 | 8 | −3, −1 | **−4** |
| 2 | T | 9, 11 | 8 | 1, 3 | **4** |
| 3 | C | 4, 6 | 4 | 0, 2 | **2** |
| 4 | C | 2, 4 | 4 | −2, 0 | **−2** |

Estimate: **8 − 4 = 4**.

### Why divide by n², and where the cluster sums come from

The arm mean is `ȳ_T = (sum of all n_T values) / n_T`, so

`Var(ȳ_T) = Var(Σ yᵢ) / n_T²`.

The `n²` is not a "variance divides by n" step; it is `Var(aX) = a²Var(X)` with `a = 1/n_T`. Now the sum of all
values in the arm is the sum of the cluster sums. Clusters are independent of each other, so their variances
add; *inside* a cluster we make no assumption at all:

`Var(Σ yᵢ) = Σ_g Var(S_g)`.

Each `Var(S_g)` is estimated by the square of that cluster's residual sum, `S_g²` (its expected value is
about `Var(S_g)` because the residuals have mean 0). This is the step that captures within-cluster
correlation: if the two users in a cluster move together, their residuals have the same sign and `S_g²` is
large; if they were independent, `S_g²` would on average equal the sum of the individual `rᵢ²`.

| quantity | treated arm | control arm |
| --- | --- | --- |
| Σ S_g² | 16 + 16 = 32 | 4 + 4 = 8 |
| Var(arm mean) = Σ S_g² / n_arm² | 32 / 16 = 2.0 | 8 / 16 = 0.5 |

**CR0:** Var(difference) = 2.0 + 0.5 = 2.5, **SE = 1.581**.

**Compare with the naive (independent) version** on the same footing, which uses each user's `rᵢ²` instead of
each cluster's `S_g²`: treated Σrᵢ² = 9 + 1 + 1 + 9 = 20, control 0 + 4 + 4 + 0 = 8, so
Var = 20/16 + 8/16 = 1.75, SE = **1.323**. In the treated clusters the two users move together (−3, −1 and
1, 3), so `S_g² = 16` exceeds the users' own `Σrᵢ² = 10` in each cluster: that excess is the within-cluster
covariance (`S² = Σr² + 2 Σ rᵢrⱼ`), and it is what the naive version throws away. In the control clusters one
residual is 0, so there is no covariance to add (4 = 4). The textbook pooled SE, √[(28/6)(1/4 + 1/4)] =
**1.528**, uses the same `Σrᵢ²` with a DF correction.

### The small-sample corrections (the "divide by G − 1")

`S_g²` treats each cluster sum as if its mean were known to be 0. It is not: the residuals were measured from
an estimated arm mean, which uses up one degree of freedom per arm, exactly as `n − 1` did in example 1. With
`G` clusters there are only `G − 1` free cluster sums, so the honest estimate of `Var(S_g)` from the cluster
sums is `Σ S_g² / (G − 1)`, not `/ G`.

- **CR1** (Stata, statsmodels) applies one global factor `G/(G−1) × (n−1)/(n−k)` = 4/3 × 7/6 = 1.556:
  Var = 2.5 × 1.556 = 3.889, **SE = 1.972**.
- **CR2** makes the correction per cluster from the leverage. In this balanced example it works out to a
  `G_arm/(G_arm − 1)` = 2/1 factor in each arm: (2.0 + 0.5) × 2 = 5.0, **SE = 2.236**.
- **Analysing cluster means** gives the same 2.236: treated city means 6 and 10 (variance 8), control 5 and 3
  (variance 2), SE = √(8/2 + 2/2) = √5 = 2.236.

| method | SE of the difference (4) | DF for the t test |
| --- | --- | --- |
| naive, Σrᵢ²/n² | 1.323 | |
| classic pooled, users independent | 1.528 | 6 |
| CR0 | 1.581 | |
| CR1 | 1.972 | G − 1 = 3 |
| CR2 | **2.236** | G − 1 = 3 |
| t test on the 4 cluster means | **2.236** | 2 |

With 4 clusters the honest answer is t = 4/2.236 = 1.79 on 2–3 DF (p ≈ 0.17–0.22), not t = 2.62 on 6 DF
(p = 0.04) from the naive SE. Eight users were really four independent pieces of information.
