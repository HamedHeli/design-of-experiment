"""Blocking 2^p factorial designs: confounding treatment interactions with blocks, design
generators, the defining relation and alias groups (STAT 404 ch. 7).

Repeating the whole 2^p in each block
    Treat the block as another factor; randomize separately within each block. Usually the
    treatment x block interactions are pooled as error.

Spreading one 2^p over 2^q blocks
    Pick q block generators, e.g. B = 123 (2 blocks) or B1 = 12, B2 = 13 (4 blocks). Runs where the
    product column(s) have the same signs go in the same block. The column B1 then estimates
    beta_B1 + beta_12: the 12 interaction is *confounded* with blocks.

Word algebra (columns of +/-1, so x_i^2 = 1)
    The product of two effects is the symmetric difference of their factor sets: 12 x 23 = 13.
    The generators define words equal to I: I = 12B1 = 13B2, and all products of them
    (here also 23B1B2) form the defining relation. The alias group of an effect E is
    {E x W : W in the defining relation}.
    With 4 blocks there are 3 block DF, so 3 treatment effects are lost: the 2 generators and
    their product (12, 13 and 23 above). Choose generators so the lost effects are high order.

Pseudo factors
    A single 4-level blocking factor (4 days) is coded by 2 two-level pseudo factors B1, B2;
    the "interaction" B1B2 is a genuine block contrast, so its alias is also lost.

Run:  python 06-blocking-two-level-factorials/blocking.py
"""

from __future__ import annotations

import itertools

import numpy as np


def word(s):
    """'123' -> frozenset({'1','2','3'}); lists of names are kept as is (for 'B1')."""
    return frozenset(s) if isinstance(s, str) else frozenset(s)


def mult(*words):
    out = frozenset()
    for w in words:
        out = out.symmetric_difference(w)
    return out


def fmt(w):
    """Treatment digits first, then block names, I for the empty word."""
    if not w:
        return "I"
    t = sorted(x for x in w if len(x) == 1)
    b = sorted(x for x in w if len(x) > 1)
    return "".join(t) + "".join(b)


def group(gen_words):
    """All products of subsets of the generator words (the defining relation incl. I)."""
    out = set()
    for r in range(len(gen_words) + 1):
        for S in itertools.combinations(gen_words, r):
            out.add(mult(*S))
    return out


def blocked_design(p, generators):
    """generators: {'B1': '12', 'B2': '13'}. Returns the 2^p design, block index per run,
    and the treatment effects confounded with blocks."""
    D = np.array(list(itertools.product([-1, 1], repeat=p)), dtype=float)
    cols = []
    for g in generators.values():
        cols.append(np.prod(D[:, [int(c) - 1 for c in g]], axis=1))
    B = np.column_stack(cols) if cols else np.zeros((len(D), 0))
    block = ((B > 0).astype(int) * (2 ** np.arange(B.shape[1])[::-1])).sum(1)
    lost = {fmt(w) for w in group([word(g) for g in generators.values()]) if w}
    return D, block, sorted(lost, key=lambda s: (len(s), s))


def defining_relation(generators):
    words = [mult(word(g), word([b])) for b, g in generators.items()]
    return sorted((fmt(w) for w in group(words)), key=lambda s: (len(s), s))


def alias_group(effect, generators):
    words = [mult(word(g), word([b])) for b, g in generators.items()]
    e = word(effect)
    return sorted((fmt(mult(e, w)) for w in group(words)), key=lambda s: (len(s), s))


def demo():
    rng = np.random.default_rng(8)
    for p, gens in [(3, {"B": "123"}), (3, {"B1": "12", "B2": "13"}),
                    (5, {"B1": "123", "B2": "345"})]:
        D, block, lost = blocked_design(p, gens)
        g = ", ".join(f"{k} = {v}" for k, v in gens.items())
        print(f"2^{p} in {len(set(block))} blocks ({g}): confounded with blocks: {lost}")
    print("defining relation for B1 = 12, B2 = 13:  I = " +
          " = ".join(defining_relation({"B1": "12", "B2": "13"})[1:]))
    print("aliases of 23:", " = ".join(alias_group("23", {"B1": "12", "B2": "13"})))

    # blocks shift the data but leave every unconfounded effect estimate untouched
    D, block, _ = blocked_design(3, {"B1": "12", "B2": "13"})
    beta = {"1": 2.0, "2": -1.0, "3": 0.5}
    y_clean = 10 + sum(b * D[:, int(k) - 1] for k, b in beta.items())
    day = rng.normal(0, 5, 4)                                  # large day-to-day shifts
    y = y_clean + day[block]
    for k in ("1", "2", "3", "123"):
        x = np.prod(D[:, [int(c) - 1 for c in k]], axis=1)
        print(f"  beta_{k:3s} with blocks {x @ y / 8:6.3f}   without {x @ y_clean / 8:6.3f}")
    x12 = D[:, 0] * D[:, 1]
    print(f"  beta_12  with blocks {x12 @ y / 8:6.3f}   (truth 0: it now measures a block contrast)")


if __name__ == "__main__":
    demo()
