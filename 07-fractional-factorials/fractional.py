"""Fractional factorial 2^(p-q) designs: generators, defining relation, resolution, aliasing,
minimum aberration, foldover (STAT 404 ch. 8).

Construction
    Write the full 2^(p-q) design in p-q "basic" factors, then define each extra factor as a product
    of basic columns, e.g. 4 = 123 gives the half fraction 2^(4-1) with I = 1234.

Defining relation and aliases
    The q generator words (e.g. 1234) and all their products are aliased with I; there are 2^q words
    including I. The alias set of an effect E is {E x W}. In 2^(4-1), I = 1234 so 1 = 234 and 12 = 34.

Resolution
    Length of the shortest word in the defining relation (excluding I).
    III: main effects aliased with 2-factor interactions (2fi).
    IV:  main effects clear of 2fi, but 2fi aliased with each other.
    V:   main effects and 2fi clear of each other.
    Minimum aberration: among designs of maximum resolution, minimize the number of shortest words.

Foldover
    Rerun the fraction with every sign reversed. Combined, the 2 fractions drop every odd-length word
    from the defining relation, so a resolution III design becomes resolution IV and every main effect
    is freed of its 2fi aliases.

Screening ("cast a wide net")
    With a 16-run budget, studying all 10 candidate factors in a resolution III fraction beats a full 2^4
    in the 4 factors guessed to matter: the important factors you did not guess are not lost.

Run:  python 07-fractional-factorials/fractional.py
"""

from __future__ import annotations

import itertools

import numpy as np

ALPH = "123456789ABCDEFG"                 # factor names; 10th factor is 'A', etc.


def mult(*ws):
    out = frozenset()
    for w in ws:
        out = out.symmetric_difference(frozenset(w))
    return out


def fmt(w):
    return "".join(sorted(w, key=ALPH.index)) if w else "I"


def fractional_design(k, generators):
    """k basic factors (full 2^k), generators like {'5': '123', '6': '234'}."""
    D = np.array(list(itertools.product([-1, 1], repeat=k)), dtype=float)
    cols = [D[:, i] for i in range(k)]
    names = list(ALPH[:k])
    for f, g in generators.items():
        sign = -1.0 if g.startswith("-") else 1.0
        g = g.lstrip("-")
        cols.append(sign * np.prod(D[:, [ALPH.index(c) for c in g]], axis=1))
        names.append(f)
    return np.column_stack(cols), names


def defining_relation(generators):
    words = [mult(g.lstrip("-"), f) for f, g in generators.items()]
    rel = set()
    for r in range(1, len(words) + 1):
        for S in itertools.combinations(words, r):
            rel.add(mult(*S))
    return sorted(rel, key=lambda w: (len(w), fmt(w)))


def resolution(generators):
    return min(len(w) for w in defining_relation(generators))


def wordlength_pattern(generators, p):
    rel = defining_relation(generators)
    return [sum(len(w) == L for w in rel) for L in range(3, p + 1)]


def aliases(effect, generators, max_order=None):
    out = [mult(effect, w) for w in [frozenset()] + defining_relation(generators)]
    if max_order:
        out = [w for w in out if len(w) <= max_order]
    return [fmt(w) for w in sorted(out, key=lambda w: (len(w), fmt(w)))]


def foldover(D):
    return np.vstack([D, -D])


def column(D, names, effect):
    return np.prod(D[:, [names.index(c) for c in effect]], axis=1)


def demo():
    rng = np.random.default_rng(9)
    g = {"4": "123"}
    print(f"2^(4-1), 4 = 123: I = {' = '.join(fmt(w) for w in defining_relation(g))}, "
          f"resolution {resolution(g)};  1 = {aliases('1', g)[1]}, 12 = {aliases('12', g)[1]}")
    g = {"4": "12", "5": "13"}
    print(f"2^(5-2), 4 = 12, 5 = 13: I = {' = '.join(fmt(w) for w in defining_relation(g))}, "
          f"resolution {resolution(g)};  1 aliased (up to 2fi) with {aliases('1', g, 2)[1:]}")

    # minimum aberration: two resolution IV 2^(7-2) designs
    a = {"6": "123", "7": "145"}
    b = {"6": "1234", "7": "1235"}
    for nm, gg in (("6 = 123, 7 = 145 ", a), ("6 = 1234, 7 = 1235", b)):
        print(f"2^(7-2) {nm}: resolution {resolution(gg)}, words of length 3..7: "
              f"{wordlength_pattern(gg, 7)}")

    # saturated 2^(7-4) and its foldover
    g7 = {"4": "12", "5": "13", "6": "23", "7": "123"}
    D, names = fractional_design(3, g7)
    beta = {"1": 2.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0}
    y = 10 + 2 * D[:, 0] + 1.5 * D[:, 1] * D[:, 2] + 0.2 * rng.standard_normal(8)   # 1 and 23 active
    est = {f: column(D, names, f) @ y / 8 for f in names}
    print(f"2^(7-4) resolution {resolution(g7)}: beta_6 estimate {est['6']:.2f} "
          f"(truth 0; 6 = 23 so it picks up the 23 interaction)")
    Df = foldover(D)
    yf = 10 + 2 * Df[:, 0] + 1.5 * Df[:, 1] * Df[:, 2] + 0.2 * rng.standard_normal(16)
    est_f = {f: column(Df, names, f) @ yf / 16 for f in names}
    print(f"after foldover (16 runs): beta_6 = {est_f['6']:.2f}, beta_1 = {est_f['1']:.2f}; "
          f"and the 23 interaction is estimable on its own: "
          f"{(Df[:, 1] * Df[:, 2]) @ yf / 16:.2f}")

    # wide net: 10 factors in 16 runs (resolution III) vs a 2^4 in 4 guessed factors
    g10 = {"5": "123", "6": "124", "7": "134", "8": "234", "9": "1234", "A": "12"}
    D10, n10 = fractional_design(4, g10)
    print(f"2^(10-6) in 16 runs: resolution {resolution(g10)}; all 10 columns orthogonal: "
          f"{np.allclose(D10.T @ D10, 16 * np.eye(10))}")
    active = {"2": 3.0, "7": -2.5, "A": 2.0}                 # factors 7 and 10 were not 'guessed'
    y10 = 50 + sum(b * column(D10, n10, f) for f, b in active.items()) + rng.standard_normal(16)
    b10 = {f: column(D10, n10, f) @ y10 / 16 for f in n10}
    top = sorted(b10, key=lambda f: -abs(b10[f]))[:3]
    print(f"  largest |beta|: {', '.join(f'{f} ({b10[f]:.2f})' for f in top)}; a 2^4 in factors 1-4 "
          "would have found only factor 2")


if __name__ == "__main__":
    demo()
