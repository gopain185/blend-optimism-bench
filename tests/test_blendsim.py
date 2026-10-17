import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from blendsim import experiments as ex
from blendsim.blend import RULES, hill_climb, logistic_stack, nested_estimate
from blendsim.sim import auc, auc_to_d, make_pool


def test_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    s = rng.standard_normal(5000) + y
    assert auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_single_model_aucs_match_targets():
    pool = make_pool(np.random.default_rng(1), 8, best_auc=0.96, spread=0.002, ceiling=0.961)
    a = pool.single_aucs()
    assert a.max() == pytest.approx(0.96, abs=1e-12)
    assert a.min() >= 0.958 - 1e-12


def test_true_auc_matches_large_sample():
    rng = np.random.default_rng(2)
    pool = make_pool(rng, 5)
    y, S = pool.sample(rng, 400_000)
    w = rng.random(5)
    assert auc(y, S @ w) == pytest.approx(pool.true_auc(w), abs=0.0008)


def test_ceiling_bounds_every_blend():
    rng = np.random.default_rng(3)
    pool = make_pool(rng, 12, ceiling=0.961)
    assert pool.optimal_auc() < 0.961
    for _ in range(50):
        assert pool.true_auc(rng.standard_normal(12)) <= pool.optimal_auc() + 1e-12


def test_optimal_weights_are_optimal():
    rng = np.random.default_rng(4)
    pool = make_pool(rng, 6)
    best = pool.optimal_auc()
    w = pool.optimal_weights()
    for _ in range(50):
        assert pool.true_auc(w + 0.05 * rng.standard_normal(6)) <= best + 1e-12


def test_auc_to_d_roundtrip():
    from scipy.stats import norm
    assert norm.cdf(auc_to_d(0.96) / np.sqrt(2)) == pytest.approx(0.96)


def test_rules_return_valid_weights():
    rng = np.random.default_rng(5)
    pool = make_pool(rng, 6)
    y, S = pool.sample(rng, 20_000)
    for name, rule in RULES.items():
        w = rule(y, S)
        assert w.shape == (6,) and np.isfinite(w).all(), name
    w = hill_climb(y, S)
    assert (w >= 0).all() and w.sum() == pytest.approx(1.0)


def test_stack_beats_best_single_in_truth():
    rng = np.random.default_rng(6)
    pool = make_pool(rng, 10)
    y, S = pool.sample(rng, 100_000)
    assert pool.true_auc(logistic_stack(y, S)) > pool.single_aucs().max()


def test_nested_estimate_close_to_truth():
    rng = np.random.default_rng(7)
    pool = make_pool(rng, 6)
    y, S = pool.sample(rng, 100_000)
    assert nested_estimate(hill_climb, y, S) == pytest.approx(pool.true_auc(hill_climb(y, S)), abs=0.003)


def test_selection_optimism_is_positive_and_shrinks_with_rows():
    small = ex.n_rows(ns=(5_000,), k=10, reps=(30,))
    large = ex.n_rows(ns=(100_000,), k=10, reps=(10,))
    s = small[small.rule == "hill-climb"].selection_optimism.mean()
    l = large[large.rule == "hill-climb"].selection_optimism.mean()
    assert s > 0
    assert l < s
