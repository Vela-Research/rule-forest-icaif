import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from rule_forest.pool import candidate_pool, precision


def test_floor_uses_training_founders_only():
    y = np.array([1, 0, 0, 0, 1, 1, 1, 1])
    X = np.array([[0, 1, 1, 1, 1, 1, 1, 1],      # no hit among the first four founders
                  [1, 0, 0, 0, 0, 0, 0, 0]])
    assert candidate_pool(X, y, np.arange(4), similarity="none") == [1]
    assert candidate_pool(X, y, np.arange(8), similarity="none") == [0, 1]


def test_hamming_drops_the_lower_precision_twin():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    a = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    b = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])     # differs from a on 1 of 10 founders
    c = np.array([0, 1, 0, 1, 1, 1, 1, 0, 0, 0])
    X = np.vstack([b, a, c])
    assert precision(X, y)[1] > precision(X, y)[0]
    assert candidate_pool(X, y, np.arange(10), similarity="hamming", threshold=0.15, floor=0) == [1, 2]


def test_hamming_tie_drops_the_later_question():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    X = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]])   # equal precision, distance 0.2
    assert candidate_pool(X, y, np.arange(10), similarity="hamming", threshold=0.25, floor=0) == [0]


def test_jaccard_compares_yes_sets_not_agreement():
    # The answers agree on 96 of 100 founders, but the two YES sets overlap 2 / 6.
    y = np.zeros(100, int); y[:3] = 1
    a = np.zeros(100, int); a[[0, 1, 2, 3]] = 1
    b = np.zeros(100, int); b[[0, 1, 4, 5]] = 1
    X = np.vstack([a, b])
    assert candidate_pool(X, y, np.arange(100), similarity="jaccard", threshold=0.9, floor=0) == [0, 1]
    assert candidate_pool(X, y, np.arange(100), similarity="jaccard", threshold=0.3, floor=0) == [0]


def test_cosine_clusters_on_condensed_distances():
    rng = np.random.RandomState(0)
    y = (rng.rand(300) < 0.1).astype(int)
    base = (rng.rand(4, 300) < 0.2).astype(int)
    noisy = base.copy()
    flip = rng.rand(*noisy.shape) < 0.02
    noisy[flip] = 1 - noisy[flip]
    X = np.vstack([base, noisy])
    kept = candidate_pool(X, y, np.arange(300), similarity="cosine", threshold=0.85, floor=0)
    labels = fcluster(linkage(pdist(X, "cosine"), "average"), t=0.15, criterion="distance")
    assert len(kept) == len(np.unique(labels))
    p = precision(X, y)
    for lab in np.unique(labels):
        members = np.where(labels == lab)[0]
        assert members[np.argmax(p[members])] in kept


def test_cosine_merges_near_copies_only():
    y = np.zeros(40, int); y[:4] = 1
    a = np.zeros(40, int); a[[0, 1, 2, 10, 11, 12, 13, 14]] = 1
    b = a.copy(); b[15] = 1                                  # cosine similarity 0.94 with a
    c = np.zeros(40, int); c[[3, 20, 21, 22, 23, 24]] = 1    # unrelated to a and b
    X = np.vstack([b, a, c])
    assert candidate_pool(X, y, np.arange(40), similarity="cosine", threshold=0.85, floor=0) == [1, 2]


def test_never_yes_questions_are_dropped():
    y = np.array([1, 0, 0, 0, 1, 0])
    X = np.array([[0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0]])
    for sim in ("hamming", "jaccard", "cosine"):
        assert 0 not in candidate_pool(X, y, np.arange(6), similarity=sim, threshold=0.5, floor=0)


def test_expert_cap_keeps_pool_size():
    rng = np.random.RandomState(2)
    y = (rng.rand(400) < 0.2).astype(int)
    X = (rng.rand(7, 400) < 0.3).astype(int)
    source = np.array(["llm"] * 5 + ["expert"] * 2)
    kept = candidate_pool(X, y, np.arange(400), similarity="none", floor=0, source=source, cap_expert=True)
    assert sorted(kept) == [0, 1, 2, 5, 6]
