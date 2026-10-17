import numpy as np

from experiments.founders import fold_quotas, inclusion
from rule_forest import elastic_net, vote


def _data(seed=0, n=400, q=6):
    rng = np.random.RandomState(seed)
    y = (rng.rand(n) < 0.15).astype(np.int8)
    X = (rng.rand(q, n) < (0.2 + 0.3 * y)).astype(np.int8)
    return X, y


def test_elastic_net_never_sees_test_labels():
    X, y = _data()
    fold = [next(vote.outer_folds(y))]
    rep, k, tr, te = fold[0]
    a = elastic_net.nested_cv(X, y, "sweep", folds=fold)[0]
    y2 = y.copy()
    y2[te] = 1 - y2[te]
    b = elastic_net.nested_cv(X, y2, "sweep", folds=[(rep, k, tr, te)])[0]
    assert np.array_equal(a["pred"], b["pred"]) and a["threshold"] == b["threshold"]


def test_identical_answers_get_identical_probabilities():
    X, y = _data(1)
    m, _, _ = elastic_net.sweep_fit(X.T.astype(float), y)
    p = elastic_net.proba(m, np.asfortranarray(X.T.astype(float)))
    patterns = [tuple(r) for r in X.T]
    by_pattern = {}
    for pat, v in zip(patterns, p):
        by_pattern.setdefault(pat, set()).add(v)
    assert all(len(v) == 1 for v in by_pattern.values())


def test_shortlists_hold_exactly_k_founders():
    rng = np.random.RandomState(2)
    y = (rng.rand(8173) < 0.02).astype(int)
    folds = [te for _, _, _, te in list(vote.outer_folds(y, n_repeats=1))]
    scores = rng.randint(0, 6, 8173).astype(float)          # many ties at every cut
    for K in (40, 60, 402, 800):
        assert fold_quotas([len(f) for f in folds], K, 8173).sum() == K
        inc = inclusion(scores, folds, K, 8173)
        assert np.isclose(inc.sum(), K) and inc.min() >= 0 and inc.max() <= 1
