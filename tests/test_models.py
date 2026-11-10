"""Fast correctness checks for every implementation in the repo.

Each check compares against statsmodels / scipy, a closed form, brute-force enumeration or a
defining property. Run from the repo root:  python -m pytest -q
"""

from __future__ import annotations

import importlib.util
import itertools
import os
import sys

import numpy as np
import pytest
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(folder, name):
    path = os.path.join(ROOT, folder, name + ".py")
    sys.path.insert(0, os.path.join(ROOT, folder))
    spec = importlib.util.spec_from_file_location(f"{folder.replace('-', '_')}_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# 00 ---------------------------------------------------------------------------------
def test_randomization_removes_observational_bias():
    m = load("00-introduction-and-principles", "observational_vs_randomized")
    rng = np.random.default_rng(0)
    obs = np.mean([m.estimate(*m.simulate(rng, 40, "observational"), "observational") for _ in range(300)])
    rnd = np.mean([m.estimate(*m.simulate(rng, 40, "randomized"), "randomized") for _ in range(300)])
    assert obs > 3 and abs(rnd - 1) < 0.2


# 01 ---------------------------------------------------------------------------------
def test_ols_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    m = load("01-linear-models", "linear_model")
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(30), rng.normal(size=(30, 3))])
    y = X @ [1, 2, 0, -1] + rng.normal(size=30)
    fit, ref = m.fit_ols(X, y), sm.OLS(y, X).fit()
    assert np.allclose(fit.beta, ref.params) and np.allclose(fit.se, ref.bse) and np.allclose(fit.p, ref.pvalues)
    assert np.isclose(m.anova_table(X, y)["Regression"]["F"], ref.fvalue)
    F, p, _ = m.extra_ss_test(X, X[:, :2], y)
    Fr = ref.f_test(np.eye(4)[2:])
    assert np.isclose(F, float(np.squeeze(Fr.fvalue))) and np.isclose(p, float(Fr.pvalue))


def test_prediction_intervals_match_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    m = load("01-linear-models", "linear_model")
    rng = np.random.default_rng(2)
    X = np.column_stack([np.ones(25), rng.normal(size=25)])
    y = 1 + X[:, 1] + rng.normal(size=25)
    x0 = np.array([1.0, 0.7])
    _, ci, pi = m.predict(m.fit_ols(X, y), x0)
    f = sm.OLS(y, X).fit().get_prediction(x0[None]).summary_frame(alpha=0.05)
    assert np.allclose(ci, f[["mean_ci_lower", "mean_ci_upper"]].values[0])
    assert np.allclose(pi, f[["obs_ci_lower", "obs_ci_upper"]].values[0])


def test_boxcox_finds_log_for_lognormal():
    m = load("01-linear-models", "linear_model")
    rng = np.random.default_rng(3)
    g = np.repeat(np.arange(4), 30)
    X = np.eye(4)[g]
    y = np.exp(1 + 0.5 * g + 0.3 * rng.normal(size=len(g)))
    _, best, (lo, hi) = m.boxcox_profile(X, y, np.linspace(-1, 1.5, 101))
    assert lo <= 0 <= hi and abs(best) < 0.4


# 02 ---------------------------------------------------------------------------------
def test_one_way_anova_matches_scipy_and_codings_agree():
    m = load("02-completely-randomized-designs", "crd")
    rng = np.random.default_rng(4)
    g = np.repeat(np.arange(4), [5, 6, 7, 5])
    y = g * 0.5 + rng.normal(size=len(g))
    an = m.one_way_anova(y, g)
    ref = stats.f_oneway(*[y[g == k] for k in range(4)])
    assert np.isclose(an["F"], ref.statistic) and np.isclose(an["p"], ref.pvalue)
    fits = [X @ np.linalg.lstsq(X, y, rcond=None)[0] for X in m.design_matrices(g)]
    assert np.allclose(fits[0], fits[1]) and np.allclose(fits[0], fits[2])


def test_orthogonal_polynomial_contrasts_split_ss_treatments():
    m = load("02-completely-randomized-designs", "crd")
    C = m.orthogonal_poly_contrasts([1, 2, 3, 4, 5])
    assert np.allclose(C.sum(0), 0) and np.allclose(C.T @ C, np.eye(4))
    assert np.allclose(C[:, 0] / C[-1, 0], np.array([-2, -1, 0, 1, 2]) / 2)   # linear is linear
    rng = np.random.default_rng(5)
    g = np.repeat(np.arange(5), 4)
    y = rng.normal(size=20) + (g - 2) ** 2
    an = m.one_way_anova(y, g)
    assert np.isclose(sum(m.contrast(an, c)["ss"] for c in C.T), an["ss_trt"])


# 03 ---------------------------------------------------------------------------------
def test_rbd_anova_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.formula.api")
    from statsmodels.stats.anova import anova_lm
    import pandas as pd
    m = load("03-randomized-blocks-and-two-way", "rbd")
    rng = np.random.default_rng(6)
    Y = rng.normal(size=(4, 5)) + np.arange(5)[None]
    an = m.rbd_anova(Y)
    df = pd.DataFrame([(i, j, Y[i, j]) for i in range(4) for j in range(5)], columns=["t", "b", "y"])
    ref = anova_lm(sm.ols("y ~ C(t) + C(b)", df).fit())
    assert np.isclose(an["F_t"], ref.loc["C(t)", "F"]) and np.isclose(an["ss_r"], ref.loc["Residual", "sum_sq"])


def test_two_way_and_tukey():
    sm = pytest.importorskip("statsmodels.formula.api")
    from statsmodels.stats.anova import anova_lm
    import pandas as pd
    m = load("03-randomized-blocks-and-two-way", "rbd")
    rng = np.random.default_rng(7)
    Y = rng.normal(size=(3, 4, 2))
    out = m.two_way_anova(Y)
    df = pd.DataFrame([(i, j, Y[i, j, k]) for i in range(3) for j in range(4) for k in range(2)],
                      columns=["a", "b", "y"])
    ref = anova_lm(sm.ols("y ~ C(a) * C(b)", df).fit())
    assert np.isclose(out["AB"]["F"], ref.loc["C(a):C(b)", "F"])
    # Tukey's test flags a multiplicative table and not its log
    a, c = np.linspace(1, 3, 5), np.linspace(1, 4, 6)
    Ym = np.outer(a, c) * np.exp(0.01 * rng.normal(size=(5, 6)))
    assert m.tukey_nonadditivity(Ym)[2] < 1e-4 and m.tukey_nonadditivity(np.log(Ym))[2] > 0.01


# 04 ---------------------------------------------------------------------------------
def test_factorial_anova_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.formula.api")
    from statsmodels.stats.anova import anova_lm
    import pandas as pd
    m = load("04-factorial-experiments", "factorial")
    rng = np.random.default_rng(8)
    Y = rng.normal(size=(3, 2, 4, 2))
    an = m.factorial_anova(Y, ["A", "B", "C"])
    rows = [(i, j, k, Y[i, j, k, r]) for i in range(3) for j in range(2) for k in range(4) for r in range(2)]
    ref = anova_lm(sm.ols("y ~ C(a) * C(b) * C(c)", pd.DataFrame(rows, columns=list("abcy"))).fit())
    assert np.isclose(an["A:B:C"]["ss"], ref.loc["C(a):C(b):C(c)", "sum_sq"])
    assert np.isclose(an["A:C"]["F"], ref.loc["C(a):C(c)", "F"])
    assert np.isclose(an["Error"]["ss"], ref.loc["Residual", "sum_sq"])


def test_ss_decomposes_total_unreplicated():
    m = load("04-factorial-experiments", "factorial")
    Y = np.random.default_rng(9).normal(size=(3, 3, 3))
    an = m.factorial_anova(Y, list("ABC"), max_order=2)
    total = sum(r["ss"] for r in an.values())
    assert np.isclose(total, ((Y - Y.mean()) ** 2).sum())


# 05 ---------------------------------------------------------------------------------
def test_two_level_effects_are_regression_coefficients():
    m = load("05-two-level-factorials", "two_level")
    X, labels = m.contrast_matrix(m.full_factorial(4))
    y = np.random.default_rng(10).normal(size=16)
    b, b0 = m.estimate_effects(X, y)
    ref = np.linalg.lstsq(np.column_stack([np.ones(16), X]), y, rcond=None)[0]
    assert np.allclose(ref, np.r_[b0, b]) and len(labels) == 15


def test_lenth_pse_near_sigma_over_sqrt_n():
    m = load("05-two-level-factorials", "two_level")
    rng = np.random.default_rng(11)
    X, _ = m.contrast_matrix(m.full_factorial(5))
    pses = [m.lenth(m.estimate_effects(X, rng.normal(size=32))[0])[0] for _ in range(300)]
    assert abs(np.mean(pses) / (1 / np.sqrt(32)) - 1) < 0.15


# 06 ---------------------------------------------------------------------------------
def test_blocking_confounds_generators_and_products():
    m = load("06-blocking-two-level-factorials", "blocking")
    D, block, lost = m.blocked_design(3, {"B1": "12", "B2": "13"})
    assert lost == ["12", "13", "23"] and np.bincount(block).tolist() == [2, 2, 2, 2]
    for k in ("12", "13", "23"):                        # each lost column is constant within blocks
        x = np.prod(D[:, [int(c) - 1 for c in k]], axis=1)
        assert all(len(set(x[block == b])) == 1 for b in range(4))
    for k in ("1", "2", "3", "123"):                    # the rest are orthogonal to every block
        x = np.prod(D[:, [int(c) - 1 for c in k]], axis=1)
        assert all(x[block == b].sum() == 0 for b in range(4))


# 07 ---------------------------------------------------------------------------------
def test_fraction_resolution_and_aliases_by_brute_force():
    m = load("07-fractional-factorials", "fractional")
    g = {"5": "123", "6": "124"}
    D, names = m.fractional_design(4, g)
    assert m.resolution(g) == 4
    # brute force: every word in the defining relation is a column of all +1
    for w in m.defining_relation(g):
        assert np.all(m.column(D, names, m.fmt(w)) == 1)
    # and no other product of columns is constant
    words = {frozenset(c) for r in range(1, 7) for c in itertools.combinations(names, r)}
    const = {w for w in words if abs(m.column(D, names, "".join(w)).sum()) == len(D)}
    assert const == set(m.defining_relation(g))


def test_foldover_raises_resolution_iii_to_iv():
    m = load("07-fractional-factorials", "fractional")
    D, names = m.fractional_design(3, {"4": "12", "5": "13", "6": "23", "7": "123"})
    F = m.foldover(D)
    for a, b, c in itertools.combinations(names, 3):      # main effects now orthogonal to every 2fi
        assert m.column(F, names, a) @ m.column(F, names, b + c) == 0


# 08 ---------------------------------------------------------------------------------
def test_latin_square_properties_and_anova():
    m = load("08-latin-squares", "latin_square")
    rng = np.random.default_rng(12)
    for l in (3, 4, 5, 6):
        assert m.is_latin(m.randomize_square(m.cyclic_square(l), rng))
    G1, G2 = m.graeco_latin(7)
    assert m.is_latin(G1) and m.is_latin(G2) and m.are_orthogonal(G1, G2)
    sm = pytest.importorskip("statsmodels.formula.api")
    from statsmodels.stats.anova import anova_lm
    import pandas as pd
    L = m.randomize_square(m.cyclic_square(4), rng)
    Y = rng.normal(size=(4, 4))
    df = pd.DataFrame([(i, j, L[i, j], Y[i, j]) for i in range(4) for j in range(4)], columns=list("rcty"))
    ref = anova_lm(sm.ols("y ~ C(r) + C(c) + C(t)", df).fit())
    assert np.isclose(m.latin_anova(Y, L)["F_t"], ref.loc["C(t)", "F"])


# 09 ---------------------------------------------------------------------------------
def test_imbalance_variance_exact_by_enumeration():
    m = load("09-randomization-blocking-replication", "randomization")
    z = np.random.default_rng(13).normal(size=8)
    diffs = [z[list(a)].mean() - np.delete(z, list(a)).mean() for a in itertools.combinations(range(8), 4)]
    assert abs(np.mean(diffs)) < 1e-12 and np.isclose(np.var(diffs), m.imbalance_variance(z))


def test_paired_randomization_test():
    m = load("09-randomization-blocking-replication", "randomization")
    p, ref = m.paired_randomization_test([1.0, 2.0, 3.0, 4.0])
    assert len(ref) == 16 and np.isclose(p, 2 / 16)      # only all + and all - are as extreme


# 10 ---------------------------------------------------------------------------------
def test_expected_mean_squares():
    m = load("10-random-effects", "random_effects")
    rng = np.random.default_rng(14)
    tau = np.array([0.0, 1.0, 2.0])
    sims = [m.rbd_mean_squares(m.simulate_rbd(rng, tau, 5, 1.5, 1.0)) for _ in range(4000)]
    assert abs(np.mean([s["ms_b"] for s in sims]) - (1 + 3 * 1.5 ** 2)) < 0.3
    assert abs(np.mean([s["ms_r"] for s in sims]) - 1) < 0.05
    assert abs(np.mean([s["ms_t"] for s in sims]) - (1 + 5 * 2 / 2)) < 0.3


# 11 ---------------------------------------------------------------------------------
def test_split_plot_ss_add_up_and_type_one_error():
    m = load("11-split-plot-and-repeated-measures", "split_plot")
    rng = np.random.default_rng(15)
    Y = m.simulate(rng, 3, 2, 4, [0, 0], [0, 0, 0, 0], 1.0, 1.0)
    an = m.split_plot_anova(Y)
    assert np.isclose(sum(r["ss"] for r in an.values()), ((Y - Y.mean()) ** 2).sum())
    assert sum(r["df"] for r in an.values()) == Y.size - 1
    rej = np.mean([m.split_plot_anova(m.simulate(rng, 4, 3, 3, [0, 0, 0], [0, 0, 0], 2.0, 0.5))["W"]["p"] < 0.05
                   for _ in range(1500)])
    assert 0.03 < rej < 0.07


# 12 ---------------------------------------------------------------------------------
def test_ancova_variance_ratio():
    m = load("12-variance-reduction-ancova-cuped-cupac", "ancova")
    rng = np.random.default_rng(16)
    dim, anc = [], []
    for _ in range(1500):
        y, z, t = m.simulate(rng, 200, 0.8)
        dim.append(m.diff_in_means(y, t)[0])
        anc.append(m.ancova(y, z, t)[0])
    assert abs(np.var(anc) / np.var(dim) - (1 - 0.8 ** 2)) < 0.06
    assert abs(np.mean(anc) - 0.5) < 0.01


def test_cuped_equals_ancova_and_reduces_variance_by_rho_squared():
    m = load("12-variance-reduction-ancova-cuped-cupac", "cuped")
    rng = np.random.default_rng(17)
    y, x, t = m.simulate_users(rng, 20000)
    d, se, _, theta = m.cuped_test(y, x, t)
    b = np.linalg.lstsq(np.column_stack([np.ones_like(y), t, x - x.mean()]), y, rcond=None)[0]
    assert abs(d - b[1]) < 2e-3 and abs(theta - b[2]) < 1e-2
    rho = np.corrcoef(x, y)[0, 1]
    se0 = m.ttest_delta(y, t)[1]
    assert abs((se / se0) ** 2 - (1 - rho ** 2)) < 0.01


def test_cuped_unbiased_with_any_fixed_theta():
    m = load("12-variance-reduction-ancova-cuped-cupac", "cuped")
    rng = np.random.default_rng(18)
    est = []
    for _ in range(400):
        y, x, t = m.simulate_users(rng, 4000)
        est.append(m.ttest_delta(m.cuped_adjust(y, x, theta=0.3)[0], t)[0])
    truth = 2.0 * np.exp(1.0 + 0.32) * 0.02
    assert abs(np.mean(est) - truth) < 4 * np.std(est) / np.sqrt(400)


def test_cupac_crossfit_beats_cuped_and_stays_unbiased():
    pytest.importorskip("sklearn")
    m = load("12-variance-reduction-ancova-cuped-cupac", "cupac")
    rng = np.random.default_rng(19)
    W, mu = m.make_users(rng, 4000)
    t = (rng.random(4000) < 0.5).astype(float)
    y = m.outcome(rng, mu, t, 0.0)
    f = m.crossfit_predictions(W, y, 3, rng)
    assert np.corrcoef(f, y)[0, 1] ** 2 > np.corrcoef(W[:, -1], y)[0, 1] ** 2 + 0.1
    # null effect: |delta| within 3 se
    d, se, _, _ = m.cuped_test(y, f, t)
    assert abs(d) < 3 * se


# basics -------------------------------------------------------------------------------
def test_basics_match_scipy():
    m = load("basics-variance-and-standard-errors", "basics")
    rng = np.random.default_rng(21)
    a, b = rng.normal(0, 1, 9), rng.normal(0.5, 2, 14)
    p, w = m.pooled_two_sample(a, b), m.welch_two_sample(a, b)
    assert np.isclose(p["p"], stats.ttest_ind(a, b).pvalue)
    assert np.isclose(w["p"], stats.ttest_ind(a, b, equal_var=False).pvalue)
    c = rng.normal(0, 1, 9)
    assert np.isclose(m.paired(a, c)["p"], stats.ttest_rel(c, a).pvalue)
    assert np.isclose(m.se_mean(a), stats.sem(a))


def test_pooled_variance_is_anova_mse_and_contrast_se():
    m = load("basics-variance-and-standard-errors", "basics")
    crd = load("02-completely-randomized-designs", "crd")
    rng = np.random.default_rng(22)
    g = np.repeat(np.arange(3), [4, 6, 5])
    y = rng.normal(size=len(g))
    an = crd.one_way_anova(y, g)
    groups = [(np.sum(g == k), m.sample_var(y[g == k])) for k in range(3)]
    c = np.array([1.0, -0.5, -0.5])
    assert np.isclose(m.contrast_se(groups, c), crd.contrast(an, c)["se"])


def test_variance_of_difference_and_lift_by_simulation():
    m = load("basics-variance-and-standard-errors", "basics")
    rng = np.random.default_rng(23)
    lifts, ses = [], []
    for _ in range(2000):
        yT, yC = rng.exponential(11, 400), rng.exponential(10, 400)
        l, s = m.relative_lift(yT, yC)
        lifts.append(l)
        ses.append(s)
    assert abs(np.mean(ses) / np.std(lifts) - 1) < 0.08


# cluster-robust standard errors -------------------------------------------------------
def test_hc_and_cluster_cov_match_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    m = load("01-linear-models", "robust_se")
    rng = np.random.default_rng(26)
    X, y, cl = m.simulate_clustered(rng, 15, 8, 0.3, tau=0.2)
    X = np.column_stack([X, rng.normal(size=len(y))])
    beta, e = m.ols_resid(X, y)
    assert np.allclose(m.hc_cov(X, e, "HC1"), sm.OLS(y, X).fit(cov_type="HC1").cov_params())
    assert np.allclose(m.hc_cov(X, e, "HC2"), sm.OLS(y, X).fit(cov_type="HC2").cov_params())
    ref = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": cl}).cov_params()
    assert np.allclose(m.cluster_cov(X, e, cl, "CR1")[0], ref)


def test_cr2_with_singleton_clusters_is_hc2():
    m = load("01-linear-models", "robust_se")
    rng = np.random.default_rng(27)
    X = np.column_stack([np.ones(30), rng.normal(size=(30, 2))])
    y = rng.normal(size=30)
    _, e = m.ols_resid(X, y)
    assert np.allclose(m.cluster_cov(X, e, np.arange(30), "CR2")[0], m.hc_cov(X, e, "HC2"))


def test_cluster_by_user_matches_delta_method():
    m = load("12-variance-reduction-ancova-cuped-cupac", "cluster_randomized")
    r = load("01-linear-models", "robust_se")
    clicks, t, uid = m.simulate_sessions(np.random.default_rng(28))
    X = np.column_stack([np.ones_like(clicks), t])
    _, e = r.ols_resid(X, clicks)
    se_cr = np.sqrt(r.cluster_cov(X, e, uid, "CR0")[0][1, 1])
    _, se_d = m.delta_method_ratio_se(clicks, uid, t)
    assert abs(se_cr / se_d - 1) < 0.01
