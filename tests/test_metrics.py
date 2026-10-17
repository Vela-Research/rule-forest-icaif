import numpy as np
import pytest

from rule_forest import metrics


def test_prf():
    assert metrics.prf(0, 0, 5) == (0.0, 0.0, 0.0)
    p, r, f = metrics.prf(2, 2, 2)
    assert (p, r) == (0.5, 0.5) and f == pytest.approx(0.5)


def test_pooled_sums_folds_within_a_repeat():
    recs = [{"repeat": 1, "TP": 1, "FP": 0, "FN": 1}, {"repeat": 1, "TP": 0, "FP": 1, "FN": 0},
            {"repeat": 2, "TP": 1, "FP": 1, "FN": 0}]
    out = metrics.pooled(recs)
    f1 = metrics.prf(1, 1, 1)[2]
    f2 = metrics.prf(1, 1, 0)[2]
    assert out["f05"] == pytest.approx((f1 + f2) / 2)
    assert out["repeats"] == 2


def test_bootstrap_of_identical_arms_is_zero():
    rng = np.random.RandomState(0)
    y = (rng.rand(500) < 0.1).astype(int)
    M = (rng.rand(3, 500) < 0.1).astype(int)
    r = metrics.paired_bootstrap(M, M, y, "f05", n_boot=200)
    assert r["delta"] == 0 and r["ci95"] == [0.0, 0.0] and r["p"] == 1.0


def test_bootstrap_detects_a_clear_difference():
    rng = np.random.RandomState(1)
    y = (rng.rand(2000) < 0.1).astype(int)
    good = y.copy()
    bad = (rng.rand(2000) < 0.1).astype(int)
    r = metrics.paired_bootstrap(good, bad, y, "precision", n_boot=300)
    assert r["delta"] > 0.5 and r["ci95"][0] > 0 and r["p"] < 0.01


def test_prediction_matrix_requires_every_founder():
    recs = [{"repeat": 1, "test": np.array([0, 1]), "pred": np.array([1, 0])}]
    with pytest.raises(ValueError):
        metrics.prediction_matrix(recs, 3)
