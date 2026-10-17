import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from rule_forest import vote


def reference_tune(A, y, inner_splits, seed, max_t=70):
    """The original (N, T) search over T = 1..70, a direct loop kept here as the reference."""
    Q = len(A)
    S = np.cumsum(A, axis=0).astype(float)
    best, best_score = None, -1.0
    folds = list(StratifiedKFold(inner_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    for n in range(1, Q + 1):
        for t in range(1, max_t + 1):
            scores = []
            for _, val in folds:
                pred = S[n - 1, val] >= t
                yv = y[val]
                tp, fp, fn = (pred & (yv == 1)).sum(), (pred & (yv == 0)).sum(), (~pred & (yv == 1)).sum()
                p = tp / (tp + fp) if tp + fp > 0 else 0.0
                r = tp / (tp + fn) if tp + fn > 0 else 0.0
                scores.append(0.0 if p + r == 0 else (1 + 0.25) * p * r / (0.25 * p + r))
            m = np.mean(scores)
            if m > best_score:
                best_score, best = m, (n, t)
    return (*best, best_score)


def test_tune_matches_reference_loop():
    rng = np.random.RandomState(0)
    for trial in range(6):
        n, Q = 600, 8
        y = (rng.rand(n) < 0.08).astype(np.int8)
        A = (rng.rand(Q, n) < (0.1 + 0.25 * y)).astype(np.int8)
        order = vote.rank(A, y)
        got = vote.tune(A[order], y, 5, trial)
        ref = reference_tune(A[order], y, 5, trial)
        assert got[:2] == ref[:2]
        assert got[2] == ref[2]


def test_test_labels_are_never_used():
    rng = np.random.RandomState(1)
    y = (rng.rand(800) < 0.1).astype(np.int8)
    X = (rng.rand(12, 800) < (0.15 + 0.3 * y)).astype(np.int8)
    train, test = np.arange(600), np.arange(600, 800)
    a = vote.fit(X, y, train, inner_splits=5, seed=3)
    y2 = y.copy()
    y2[test] = 1 - y2[test]
    b = vote.fit(X, y2, train, inner_splits=5, seed=3)
    assert (a.selected, a.N, a.T, a.pool) == (b.selected, b.N, b.T, b.pool)
    assert np.array_equal(a.predict(X, test), b.predict(X, test))


def test_predict_and_scorecard():
    X = np.array([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=np.int8)
    f = vote.Fit(pool=[0, 1], selected=[0, 1], N=2, T=2, train_f05=0.0)
    assert f.predict(X).tolist() == [1, 0, 0, 0]
    assert f.votes(X, [0, 3]).tolist() == [2, 0]
    card = f.scorecard(["Q one?", "Q two?"])
    assert "at least 2 of these 2" in card and "1. Q one?" in card


def test_outer_folds_cover_every_example_once_per_repeat():
    y = np.array([0] * 90 + [1] * 10)
    seen = {}
    for rep, _k, tr, te in vote.outer_folds(y, n_repeats=2):
        seen.setdefault(rep, []).extend(te.tolist())
        assert not set(tr) & set(te)
    assert all(sorted(v) == list(range(100)) for v in seen.values())


def test_empty_pool_is_reported():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    X = np.zeros((3, 10), dtype=np.int8)               # no question is ever YES
    with pytest.raises(ValueError, match="precision floor"):
        vote.fit(X, y, np.arange(10), inner_splits=2)
