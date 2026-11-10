# 08 Latin square designs

**STAT 404 ch. 9** · [back to the graph](../README.md)

**What is learned:** treatment effects after removing *two* crossed sources of nuisance variation (rows and
columns: operators and days, field rows and field columns), in only `l²` runs for `l` treatments.

## Chapters

| Notes | Covers |
| --- | --- |
| 9.1–9.2 | the design and where it is used |
| 9.3 | construction from a standard square and randomization of rows, columns and labels |
| 9.4–9.5 | the additive regression model; balance under additivity |
| 9.6 | ANOVA |
| 9.7 | Graeco-Latin squares |

## The model

### Design

An `l × l` grid; each treatment appears once in each row and once in each column. A cyclic square
`L[i, j] = (i + j) mod l` is always Latin; randomize by permuting rows, columns and treatment labels.

### Model and ANOVA

`Y = μ + ρᵢ + γⱼ + τₖ + E`, additive (no interactions among rows, columns and treatments). Rows, columns
and treatments each take `l − 1` DF; the residual has `(l − 1)(l − 2)`. Because each treatment meets each row
and each column once, treatment means are free of row and column effects: the design is a balanced
`l²`-run fraction of the `l³` row × column × treatment factorial.

### Costs and caveats

Small squares leave little error DF (2 DF for a 3 × 3), so several squares are often run and combined.
Any row × treatment or column × treatment interaction goes into the residual and is not detectable.

### Graeco-Latin squares

Superimpose a second Latin square orthogonal to the first (every pair of symbols appears once) to handle a
third blocking factor or a second treatment factor. For prime `l`, `(i + j) mod l` and `(i + 2j) mod l` work.
None exists for `l = 6`.

## Code in this folder

[`latin_square.py`](latin_square.py): cyclic construction, randomization, a Latin-property check, ANOVA,
Graeco-Latin construction and an orthogonality check.

```bash
python 08-latin-squares/latin_square.py
```

| Result (saved run: 5 × 5, row and column effects with sd 3, noise sd 0.6) | |
| --- | --- |
| Latin square F for treatments | 17.0 on (4, 12) DF, p = 7e-5 |
| same data analysed as a CRD | F = 0.52, p = 0.72 |
| treatment means minus A (truth 0, 0.5, 1, 1.5, 2) | 0, 1.12, 1.44, 1.63, 2.44 |
| 5 × 5 Graeco-Latin | both squares Latin, mutually orthogonal |

## Knowledge gap

**What the notes leave out**

- **Crossover designs.** Latin squares with subjects as rows and periods as columns are the basis of
  crossover clinical trials, where carryover effects need Williams squares (balanced for first-order carryover).
  Jones & Kenward, *Design and Analysis of Cross-Over Trials*.
- **Youden squares** (incomplete Latin squares) when the number of columns is less than `l`.
- **Mutually orthogonal Latin squares** for prime-power `l` via finite fields, and the connection to
  orthogonal arrays.

**Prerequisites to shore up**

- Additive models with several blocking factors ([03](../03-randomized-blocks-and-two-way/)).

**Open questions / further study**

- How should one analyse several Latin squares that share or do not share rows/columns? Montgomery ch. 4.2.

## Connections

- ← [03 Randomized blocks](../03-randomized-blocks-and-two-way/): one blocking factor.
- ← [07 Fractions](../07-fractional-factorials/): a Latin square is a fraction of a 3-factor design.
- → [11 Split-plot](../11-split-plot-and-repeated-measures/): the notes' split-plot exercises use Latin squares at the
  whole-plot level.
