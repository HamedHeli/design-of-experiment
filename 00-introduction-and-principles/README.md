# 00 Introduction: what makes an experiment "designed"

**STAT 404 ch. 1** · [back to the graph](../README.md)

**What is learned:** nothing is fitted yet. This folder is about *where the data come from*: why a planned,
randomized comparison can answer a causal question that a large observational dataset often cannot.

## Chapters

| Notes | Covers |
| --- | --- |
| 1.1 Experimental Design | the five features of a statistically designed experiment; a historical tour of studies (a 19th-century speed-of-light experiment, a large school milk trial, a comparison of two prenatal tests, two climate-change field experiments) |
| 1.2 Exercises | |

## The ideas

### Five features of a designed experiment

1. **Comparative.** Two or more *treatments* are compared (drug A vs drug B; low / medium / high acid).
2. **Experimental units.** Each unit (patient, batch, plot, user) gets exactly one treatment. Units differ in
   ways that also affect the response.
3. **Carefully planned.** Grouping similar units into *blocks* removes their differences from the comparison;
   a planned layout also makes the analysis simple even with many factors.
4. **Under control.** The experimenter, not the unit or an operator, decides who gets what.
5. **Randomized.** The actual allocation is random (within blocks, if there are blocks). This breaks any link
   between treatment and unknown unit properties, which is why the randomized controlled trial is the gold
   standard.

### Vocabulary used in every later folder

- **Treatment / factor / level.** A treatment is one combination of the levels of one or more *factors*
  (temperature at 400 or 600 °C, time at 8 or 12 s → 4 treatments).
- **Response** `y`: what is measured on each unit.
- **Replication:** independent units getting the same treatment. Repeated *measurements* on the same unit are
  not replicates.
- **Blocking factor:** a known source of unit-to-unit variation (day, batch, litter, operator) that is not of
  interest itself.

### The historical examples, in one line each

The examples in the chapter are chosen to show how design evolved: a careful physics measurement whose
"trials" drifted over time (the danger of no randomization and no blocking), a huge school trial where teachers
were allowed to swap children between groups (lost control → bias), a medical comparison where randomization
made the result credible, and modern ecological field experiments run as split plots (see
[11](../11-split-plot-and-repeated-measures/)).

## Code in this folder

[`observational_vs_randomized.py`](observational_vs_randomized.py): a unit property `z` affects the response.
In the "observational" study units with high `z` are more likely to be treated; in the experiment a coin decides;
in the blocked experiment units are paired on `z` and the coin decides within each pair.

```bash
python 00-introduction-and-principles/observational_vs_randomized.py
```

| Result (saved run, true effect τ = 1, n = 40) | mean of estimate | sd |
| --- | --- | --- |
| observational (treatment chosen by z) | **4.64** (biased) | 0.82 |
| completely randomized | 1.01 | 0.99 |
| randomized within pairs matched on z | 1.00 | **0.35** |
| randomized, n = 20 / 80 / 320 | | 1.42 / 0.70 / 0.35 |

Randomization removes bias; blocking removes variance; replication shrinks the variance that remains like
`1/√n`. Those are the three counter-measures that [09](../09-randomization-blocking-replication/) formalizes.

## Knowledge gap

**What the notes leave out**

- **The potential-outcomes language.** Modern causal inference writes each unit's two responses `Y_i(1)`,
  `Y_i(0)` and defines the effect as their difference. The notes use regression models instead; both views
  give the same estimators for the designs here, and the potential-outcome view is what A/B-testing and
  [12](../12-variance-reduction-ancova-cuped-cupac/) use. Imbens & Rubin, *Causal Inference*, ch. 1–6.
- **Power and sample size.** How many units to plan for is never computed in the notes. For a two-arm
  comparison, `n per arm ≈ 2 (z_{1−α/2} + z_{power})² σ² / δ²` (implemented in
  [`cuped.py`](../12-variance-reduction-ancova-cuped-cupac/cuped.py)).
- **External validity.** A perfectly randomized experiment on one population says nothing automatic about
  another. Transportability is its own topic.

**Prerequisites to shore up**

- Expectation and variance of a sample mean; independence.
- Basic hypothesis testing (t test, p-values, confidence intervals).

**Open questions / further study**

- When randomization is impossible or unethical, what is the next best design (natural experiments,
  regression discontinuity, instrumental variables)? Angrist & Pischke, *Mostly Harmless Econometrics*.
- Reading: Box, Hunter & Hunter, *Statistics for Experimenters*, ch. 1–3; Fisher, *The Design of
  Experiments* (1935), ch. 2.

## Connections

- → [01 Linear models](../01-linear-models/): every analysis in the course is a linear model in disguise.
- → [09 Counter-measures](../09-randomization-blocking-replication/): randomization, blocking and replication done
  formally, including inference that rests on the randomization alone.
- → [12 ANCOVA, CUPED, CUPAC](../12-variance-reduction-ancova-cuped-cupac/): the same three principles applied to
  online A/B tests with millions of users.
