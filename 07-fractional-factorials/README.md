# 07 Fractional factorial designs

**STAT 404 ch. 8** · [back to the graph](../README.md)

**What is learned:** main effects (and some 2-factor interactions) of many factors from a carefully chosen
fraction `1/2^q` of the full `2^p` design, with a known map of which effects are aliased with which.

## Chapters

| Notes | Covers |
| --- | --- |
| 8.1–8.2 | cast a wide net: screen all candidate factors instead of guessing a few |
| 8.3 | the half fraction 2^(4−1), the defining relation, resolution |
| 8.4 | a quarter fraction 2^(5−2) and its uneven aliasing |
| 8.5 | fractions for 5–11 factors in 8 or 16 runs; recommended generators |
| 8.6 | factors with 2^k levels via pseudo factors |
| 8.7 | blocking a fractional factorial |
| 8.8 | foldover designs to break aliases |
| 8.9 | the limits of balanced (regular) fractions |

## The models

### Construction

Write a full factorial in the `p − q` basic factors; define each extra factor as an interaction column of
basic factors (`4 = 123`). The runs are the half (or quarter …) of the `2^p` design where the generator words
equal +1.

### Defining relation, aliases, resolution

The generator words and all their products are aliased with I. Each column of the design estimates the sum
of the effects in an alias group. **Resolution** = length of the shortest word:

| Resolution | Aliasing (assuming 3fi and higher are 0) | Example |
| --- | --- | --- |
| III | main effects aliased with 2fi | 2^(5−2), I = 124 = 135 = 2345; 2^(7−4) saturated |
| IV | main effects clear; 2fi aliased with each other | 2^(4−1), I = 1234 |
| V | main effects and 2fi clear | 2^(5−1), I = 12345 |

Among designs with the same resolution, **minimum aberration** prefers the one with the fewest shortest
words, so fewer 2fi pairs are tangled.

### Wide net

With a fixed budget it is safer to include all 10 candidate factors in a resolution III fraction than to pick
5 and run them fully: you learn which factors matter, then a follow-up experiment studies them in detail.

### Foldover

Repeat the fraction with all signs reversed. The combined design removes every odd-length word from the
defining relation: a resolution III design becomes resolution IV, freeing every main effect from its 2fi
aliases. Reversing a single factor frees that factor and its 2fi.

### Limits of regular fractions

Every column is ±1 and balanced, so the run size is a power of 2 and some effect sets (e.g. all 7 main
effects and all 21 2fi in 32 runs) cannot be estimated by any regular fraction even though the run count
allows it. Non-regular (Plackett–Burman) and optimal designs fill that gap.

## Code in this folder

[`fractional.py`](fractional.py): design construction from generators (with sign choice), defining relation,
resolution, word-length pattern, alias lists, foldover, and a 10-factor screening example.

```bash
python 07-fractional-factorials/fractional.py
```

| Result (saved run) | |
| --- | --- |
| 2^(4−1), 4 = 123 | I = 1234, resolution IV; 1 = 234, 12 = 34 |
| 2^(5−2), 4 = 12, 5 = 13 | I = 124 = 135 = 2345, resolution III; 1 = 24 = 35 |
| two 2^(7−2) resolution IV designs | word-length patterns (0, 2, 0, 1, 0) vs **(0, 1, 2, 0, 0)**: the second is minimum aberration |
| 2^(7−4) with real 23 interaction 1.5 | β̂₆ = 1.57 (6 = 23); after foldover β̂₆ = −0.03, 23 estimated as 1.55 |
| 10 factors in 16 runs, active 2, 7, 10 | top three \|β̂\| are exactly 2, 7, 10 (labelled A) |

The tests verify the defining relation by brute force over every product of columns.

## Knowledge gap

**What the notes leave out**

- **Plackett–Burman and other non-regular designs** (12, 20, 24 runs) with partial, spread-out aliasing.
- **Design catalogues.** Minimum-aberration generators for up to 128 runs: Wu & Hamada Appendix 5A; the R
  package `FrF2`.
- **Analysis strategies for complex aliasing**: effect heredity, Bayesian variable selection (Chipman, Hamada
  & Wu 1997).
- **Supersaturated designs** (more factors than runs) for very early screening.

**Prerequisites to shore up**

- The word algebra of [06](../06-blocking-two-level-factorials/).

**Open questions / further study**

- Sequential experimentation: when is a foldover better than a semifold, or than augmenting with D-optimal
  runs? Montgomery ch. 8.6–8.7.

## Connections

- ← [06 Blocking](../06-blocking-two-level-factorials/): same algebra; blocking factors become treatment factors.
- ← [05 2-level factorials](../05-two-level-factorials/): half-normal plots analyse unreplicated fractions.
- → [08 Latin squares](../08-latin-squares/): a Latin square is a fraction of a 3-factor design too (l² of l³ runs).
