# 04 Factorial experiments: interactions and 1-DF contrasts

**STAT 404 ch. 5** · [back to the graph](../README.md)

**What is learned:** main effects and interaction effects of several treatment factors studied together, all
`l₁ × l₂ × … × l_k` combinations, and their decomposition into interpretable 1-DF pieces (linear, quadratic,
linear × linear, …).

## Chapters

| Notes | Covers |
| --- | --- |
| 5.1 | why factorials: a 3 × 3 × 3 experiment on a bone-implant manufacturing process |
| 5.2 | the regression model and ANOVA for three factors; definition of an interaction effect |
| 5.3 | main-effect contrasts, interaction-effect contrasts, ANOVA with 1-DF contrasts, the equivalent polynomial regression |
| 5.4 | presenting conclusions with interaction plots |
| 5.5 | replicated factorial designs |

## The models

### Why not one factor at a time?

A factorial uses every run to estimate every main effect (each effect is an average over the levels of the
other factors), and it is the only way to see **interactions**: the effect of A depending on the level of B.
Changing one factor at a time can miss the best combination entirely when factors interact
([05](../05-two-level-factorials/) quantifies this).

### Effects from marginal means

For a balanced factorial with cell means `ȳᵢⱼₖ`:

- main effect of A at level i: `ȳᵢ.. − ȳ...`
- AB interaction at (i, j): `ȳᵢⱼ. − ȳᵢ.. − ȳ.ⱼ. + ȳ...` (what is left of the AB table after main effects)
- ABC: the same inclusion–exclusion over all subsets.

`SS(effect)` = (observations per margin cell) × Σ effect², with DF = Π(levels − 1) over the factors in the
effect. Everything is orthogonal, so the SS add up to the total.

### Unreplicated designs

With one run per combination there is no pure error. The usual fix is to assume the highest-order interaction
is negligible and use it as the error term (8 DF for ABC in a 3³). Replication (5.5) gives an honest error
term instead, at a price of `l₁l₂…l_k` extra runs per replicate.

### 1-DF contrasts and the polynomial model

For three equally spaced levels the linear `(−1, 0, 1)` and quadratic `(1, −2, 1)` contrasts split each main
effect's 2 DF. Interaction contrasts are elementwise products: `A_L × B_L` asks whether the slope in A changes
linearly with B. The four products split `SS(AB)` exactly. Fitting `x_A, x_A², x_B, x_A x_B, …` as a polynomial
regression gives the same fit, so the conclusions can be stated as a response-surface equation.

## Code in this folder

[`factorial.py`](factorial.py): ANOVA for any balanced factorial (any number of factors and levels,
replicated or not, with high-order interactions optionally pooled into error), and 1-DF contrast SS on the
cell-means array.

```bash
python 04-factorial-experiments/factorial.py
```

| Result (saved run: unreplicated 3³, truth curved in A, linear in B, A×B linear-by-linear, C inert) | |
| --- | --- |
| ANOVA with ABC (8 DF) as error | A p = 2.8e-7, B p = 9.2e-5, C p = 0.45, A:B p = 0.008, A:C and B:C n.s. |
| 1-DF contrasts | A_L, A_Q, B_L, A_L×B_L significant; B_Q and the other A×B pieces not |
| check | the four A×B contrast SS sum to SS(A:B) = 29.51 |

The tests compare the ANOVA with `statsmodels` on a replicated 3 × 2 × 4 design.

## Knowledge gap

**What the notes leave out**

- **Response surface methodology.** The polynomial model of 5.3.4 is the start of RSM: central composite and
  Box-Behnken designs, steepest ascent, and finding a stationary point. Box & Draper, *Response Surfaces,
  Mixtures, and Ridge Analyses*; Myers, Montgomery & Anderson-Cook.
- **Effect hierarchy and heredity** as principles for choosing models (interactions rarely matter when their
  main effects do not), stated informally here, formalized by Wu & Hamada.
- **Unequal spacing** of quantitative levels: the integer contrasts no longer apply; use the QR construction
  in [`crd.py`](../02-completely-randomized-designs/crd.py).

**Prerequisites to shore up**

- The interaction plot: parallel lines = no interaction.
- Orthogonal contrasts ([02](../02-completely-randomized-designs/)).

**Open questions / further study**

- For a 3^k with many factors, how do you fraction it? (3^(k−p) designs, orthogonal arrays.) Wu & Hamada,
  *Experiments: Planning, Analysis, and Optimization*, ch. 6.

## Connections

- → [05 2-level factorials](../05-two-level-factorials/): with two levels each effect is a single ±1 column.
- → [11 Split-plot](../11-split-plot-and-repeated-measures/): what happens when some factors are hard to change
  and are not reset every run.
- ← [02 CRD](../02-completely-randomized-designs/): a factorial is a CRD whose treatments have structure.
