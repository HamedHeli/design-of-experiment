"""Randomized block design (RBD), efficiency of blocking, Tukey's test for non-additivity,
and the 2-way factorial with replication (STAT 404 ch. 4).

RBD model (additive)
    Y_ij = mu + tau_i + beta_j + E_ij,  treatments i = 1..t, blocks j = 1..b, one run per cell.
    Randomize the t treatments independently within every block.

ANOVA (balanced, so the sums of squares do not depend on the order of fitting)
    SS(Trt)   = b sum_i (ybar_i. - ybar..)^2                  t-1 DF
    SS(Block) = t sum_j (ybar_.j - ybar..)^2                  b-1 DF
    SS(Res)   = sum (y_ij - ybar_i. - ybar_.j + ybar..)^2     (t-1)(b-1) DF

Relative efficiency of the RBD over a CRD on the same units (estimated from the RBD)
    RE = [(b-1) MS(Block) + b(t-1) MS(Res)] / [(bt-1) MS(Res)]
    = the factor by which a CRD would need more replicates for the same precision.

Tukey's 1-DF test for non-additivity
    Adds gamma * alpha_i * beta_j to the model; SS(N) = [sum y_ij a_i b_j]^2 / (sum a_i^2 sum b_j^2)
    with a_i, b_j the estimated row and column effects. F = SS(N) / [(SS(Res) - SS(N)) / ((t-1)(b-1)-1)].

2-way factorial with n_rep replicates per cell
    Y_ijk = mu + alpha_i + beta_j + (alpha beta)_ij + E_ijk; replication gives a pure error estimate,
    so the interaction can be tested. Without replication (an RBD) it cannot: it is the residual.

Run:  python 03-randomized-blocks-and-two-way/rbd.py
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def randomize_rbd(t, b, rng):
    """Run order of treatments 0..t-1 within each of b blocks (row j = block j)."""
    return np.array([rng.permutation(t) for _ in range(b)])


def rbd_anova(Y):
    """Y is t x b (treatments x blocks), one observation per cell."""
    Y = np.asarray(Y, float)
    t, b = Y.shape
    g = Y.mean()
    a = Y.mean(1) - g
    c = Y.mean(0) - g
    R = Y - g - a[:, None] - c[None, :]
    ss_t, ss_b, ss_r = b * (a ** 2).sum(), t * (c ** 2).sum(), (R ** 2).sum()
    df_t, df_b, df_r = t - 1, b - 1, (t - 1) * (b - 1)
    ms_t, ms_b, ms_r = ss_t / df_t, ss_b / df_b, ss_r / df_r
    F_t = ms_t / ms_r
    return dict(ss_t=ss_t, ss_b=ss_b, ss_r=ss_r, df_t=df_t, df_b=df_b, df_r=df_r,
                ms_t=ms_t, ms_b=ms_b, ms_r=ms_r, F_t=F_t, p_t=stats.f.sf(F_t, df_t, df_r),
                F_b=ms_b / ms_r, trt_effects=a, block_effects=c)


def relative_efficiency(an, t, b):
    return ((b - 1) * an["ms_b"] + b * (t - 1) * an["ms_r"]) / ((b * t - 1) * an["ms_r"])


def tukey_nonadditivity(Y):
    Y = np.asarray(Y, float)
    t, b = Y.shape
    an = rbd_anova(Y)
    a, c = an["trt_effects"], an["block_effects"]
    ss_n = (a @ Y @ c) ** 2 / ((a ** 2).sum() * (c ** 2).sum())
    df = (t - 1) * (b - 1) - 1
    F = ss_n / ((an["ss_r"] - ss_n) / df)
    return ss_n, F, stats.f.sf(F, 1, df)


def two_way_anova(Y):
    """Y has shape (a, b, n_rep): replicated 2-way factorial, balanced."""
    Y = np.asarray(Y, float)
    A, B, r = Y.shape
    g = Y.mean()
    m = Y.mean(2)
    ai = m.mean(1) - g
    bj = m.mean(0) - g
    ab = m - g - ai[:, None] - bj[None, :]
    ss = dict(A=B * r * (ai ** 2).sum(), B=A * r * (bj ** 2).sum(), AB=r * (ab ** 2).sum(),
              Res=((Y - m[..., None]) ** 2).sum())
    df = dict(A=A - 1, B=B - 1, AB=(A - 1) * (B - 1), Res=A * B * (r - 1))
    ms = {k: ss[k] / df[k] for k in ss}
    out = {}
    for k in ("A", "B", "AB"):
        F = ms[k] / ms["Res"]
        out[k] = dict(ss=ss[k], df=df[k], F=F, p=stats.f.sf(F, df[k], df["Res"]))
    out["Res"] = dict(ss=ss["Res"], df=df["Res"], ms=ms["Res"])
    return out


def demo():
    rng = np.random.default_rng(4)
    t, b = 5, 7
    order = randomize_rbd(t, b, rng)
    print("run order in block 1:", order[0] + 1, "  block 2:", order[1] + 1)
    tau = np.array([0.0, 0.3, 0.6, 0.9, 1.2])
    block = 3.0 * rng.standard_normal(b)                      # big unit-to-unit (block) variation
    Y = 10 + tau[:, None] + block[None, :] + 0.5 * rng.standard_normal((t, b))

    an = rbd_anova(Y)
    print(f"RBD:  F(treatments) = {an['F_t']:.2f} on {an['df_t']},{an['df_r']} DF, p = {an['p_t']:.2g};"
          f"  F(blocks) = {an['F_b']:.1f}")
    # the same numbers analysed (wrongly) as if they came from a CRD
    from_crd_ss_res = an["ss_r"] + an["ss_b"]
    df_res = t * b - t
    F_crd = an["ms_t"] / (from_crd_ss_res / df_res)
    print(f"ignoring blocks: F = {F_crd:.2f}, p = {stats.f.sf(F_crd, t - 1, df_res):.2g}"
          "  (block variation swamps the treatment effect)")
    print(f"relative efficiency of blocking: {relative_efficiency(an, t, b):.1f}x "
          "(a CRD would need this many times more units)")
    _, F, p = tukey_nonadditivity(Y)
    print(f"Tukey non-additivity on the additive data: F = {F:.2f}, p = {p:.2f}")
    Ym = np.exp(0.3 * (tau[:, None] + block[None, :]))        # multiplicative: additive on log scale
    Ym = Ym * np.exp(0.05 * rng.standard_normal((t, b)))
    print(f"Tukey non-additivity on multiplicative data: p = {tukey_nonadditivity(Ym)[2]:.2g}, "
          f"after log: p = {tukey_nonadditivity(np.log(Ym))[2]:.2f}")

    # replicated 2-way factorial with a real interaction
    A, B, r = 3, 4, 3
    inter = np.zeros((A, B))
    inter[2, 3] = 3.0
    mu = 5 + np.arange(A)[:, None] * 0.8 + np.arange(B)[None, :] * 0.4 + inter
    Y2 = mu[..., None] + rng.standard_normal((A, B, r))
    tw = two_way_anova(Y2)
    print("2-way factorial (3 x 4, 3 replicates):  " + ",  ".join(
        f"{k}: F = {tw[k]['F']:.1f} p = {tw[k]['p']:.2g}" for k in ("A", "B", "AB")))


if __name__ == "__main__":
    demo()
