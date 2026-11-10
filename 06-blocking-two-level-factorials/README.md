# 06 Plans for blocking 2-level factorial designs

**STAT 404 ch. 7** · [back to the graph](../README.md)

**What is learned:** a *plan*, not a fit: which treatment combinations go in which block so that block
differences fall on effects you are willing to lose (high-order interactions) and nowhere else.

## Chapters

| Notes | Covers |
| --- | --- |
| 7.1–7.2 | repeating a whole 2^p in each of several blocks |
| 7.3 | one 2³ spread across two blocks (B = 123) |
| 7.4 | one 2³ across four blocks: two blocking factors, or one 4-level factor coded by pseudo factors |
| 7.5 | spreading a 2^p across 2^q blocks in general; recommended generators |

## The ideas

### Repeating the design

If a block can hold all `2^p` runs, repeat the design in each block, randomize separately within each, and treat
the block as one more factor. Treatment × block interactions are usually pooled as error; that is an
assumption, sometimes a wrong one (e.g. when the goal is a process that is robust to day-to-day variation).

### Confounding with blocks

If a block holds only half the runs, choose a column you are willing to lose, say `123`, and put the runs with
`x₁x₂x₃ = −1` in block 1 and `+1` in block 2. Then the `123` column estimates `β_B + β₁₂₃`: the 3-factor
interaction is **confounded** with blocks. All other columns are balanced within each block, so their
estimates are untouched by block shifts.

### Generators, defining relation, aliases

Notation: treatment factors are numerals, blocking factors letters. Columns are ±1, so `x²= 1` and multiplying
words is a symmetric difference: `12 × 23 = 13`.

- A generator `B₁ = 12` is the word `I = 12B₁`.
- With `q` generators, all their products make up the **defining relation** (`2^q` words including I).
- The **alias group** of an effect E is E times every word in the defining relation.

Four blocks need two generators, and three block DF are used: the two generators *and their product*. With
`B₁ = 12, B₂ = 13`, the 23 interaction is also lost (`23 = B₁B₂`), which surprises people. Good generators make
all `2^q − 1` lost effects high order: for a 2⁵ in four blocks, `B₁ = 123, B₂ = 345` loses 123, 345 and 1245
only.

### Pseudo factors

A single 4-level blocking factor (four days) is coded by two 2-level pseudo factors; their product is a
real block contrast, so it cannot be assumed negligible the way an interaction between two physical blocking
factors might be.

## Code in this folder

[`blocking.py`](blocking.py): word algebra, block assignment from generators, the confounded effects,
defining relation and alias groups, and a check that block shifts leave unconfounded estimates unchanged.

```bash
python 06-blocking-two-level-factorials/blocking.py
```

| Result (saved run) | |
| --- | --- |
| 2³ in 2 blocks (B = 123) | confounded: 123 |
| 2³ in 4 blocks (B₁ = 12, B₂ = 13) | confounded: 12, 13, **23**; I = 12B₁ = 13B₂ = 23B₁B₂ |
| 2⁵ in 4 blocks (B₁ = 123, B₂ = 345) | confounded: 123, 345, 1245 |
| 2³ with day shifts of sd 5 added | β̂₁, β̂₂, β̂₃, β̂₁₂₃ identical with and without the shifts; β̂₁₂ = 1.70 (truth 0) |

## Knowledge gap

**What the notes leave out**

- **Partial confounding.** Replicate the design and confound a *different* interaction in each replicate, so
  every effect is estimable from some replicates. Montgomery ch. 7.
- **Blocking with more than two levels** (3^k in blocks) uses modular arithmetic over GF(3).
- **Optimal blocking** for non-regular designs, and the minimum-aberration criterion for choosing among
  blocking schemes (Sitter, Chen & Feder 1997).

**Prerequisites to shore up**

- Group theory lite: the words form a group under multiplication (Z₂ arithmetic).

**Open questions / further study**

- In online experiments, "blocks" are often time periods; carryover between periods breaks the additive
  model. How should switchback experiments be designed? Bojinov, Simchi-Levi & Zhao (2023), *Management
  Science*.

## Connections

- → [07 Fractions](../07-fractional-factorials/): identical algebra; a fraction is "one block" of the full design.
- ← [03 Randomized blocks](../03-randomized-blocks-and-two-way/): blocking when the block holds every treatment.
- → [08 Latin squares](../08-latin-squares/): two blocking factors for a single treatment factor.
