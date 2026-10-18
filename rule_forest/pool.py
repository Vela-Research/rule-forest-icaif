"""Candidate pool. Rule Forest's label-based question filter, applied to one training set.

Three steps, all computed on the training founders only, so no test label is ever used:
1. Precision floor. Drop questions whose precision is below the floor (0.02001, just above the
   1.93% base rate), so questions that do no better than chance, and questions never answered YES.
2. Expert cap (expert arm only). Keep the first (n_llm - n_expert) LLM questions in scoring order,
   so that before step 3 the pool has as many questions as it would without the expert ones.
3. Redundancy removal. For every pair of near-duplicate questions, drop the one with the lower
   precision (on a tie, the one later in the current order, which for the expert arm is the
   training-F0.5 order of step 2). Near-duplicate means:
     hamming  the share of founders on whom the two answers differ is below `threshold`
     jaccard  |both YES| / |either YES| is above `threshold`
     cosine   the two fall in the same average-linkage cluster of cosine distances, cut at
              1 - threshold; each cluster keeps its highest-precision question
   A pair is judged even when one of its questions was already dropped by another pair.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from .metrics import prf

PRECISION_FLOOR = 0.02001
SIMILARITIES = ("hamming", "jaccard", "cosine", "none")


def counts(A, y):
    """TP, FP and FN of every row of A (Q, n) used as a YES-means-success rule."""
    pos = y == 1
    tp = (A[:, pos] == 1).sum(1)
    fp = (A[:, ~pos] == 1).sum(1)
    return tp, fp, int(pos.sum()) - tp


def precision(A, y):
    tp, fp, _ = counts(A, y)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)


def f05(A, y):
    """F0.5 of every row of A used as a YES-means-success rule."""
    tp, fp, fn = counts(A, y)
    return np.array([prf(int(a), int(b), int(c))[2] for a, b, c in zip(tp, fp, fn)])


def candidate_pool(X, y, founders, similarity="hamming", threshold=0.15, floor=PRECISION_FLOOR,
                   source=None, cap_expert=False):
    """Row indices of X kept for the training founders `founders` (column indices)."""
    if similarity not in SIMILARITIES:
        raise ValueError(f"similarity must be one of {SIMILARITIES}")
    A, yt = X[:, founders], y[founders]
    prec = precision(A, yt)
    ever_yes = A.sum(1) > 0
    keep = [q for q in range(len(X)) if prec[q] >= floor and ever_yes[q]]
    if cap_expert:
        # The expert questions replace the last LLM questions in file order, as in the original
        # run, so the count stays the same; dropping the lowest-ranked ones would also be reasonable.
        llm = [q for q in keep if source[q] == "llm"]
        exp = [q for q in keep if source[q] == "expert"]
        keep = llm[: max(len(llm) - len(exp), 0)] + exp
        rank = f05(A, yt)
        keep = sorted(keep, key=lambda q: -rank[q])
    if similarity == "none" or len(keep) < 2:
        return keep
    K = A[keep].astype(np.int32)
    p = prec[keep]
    removed = set()
    if similarity in ("hamming", "jaccard"):
        if similarity == "hamming":
            near = (K[:, None, :] != K[None, :, :]).sum(-1) / K.shape[1] < threshold
        else:
            both = K @ K.T
            either = K.sum(1)[:, None] + K.sum(1)[None, :] - both
            near = both / either > threshold
        for i, j in zip(*np.triu_indices(len(keep), 1)):
            if near[i, j]:
                removed.add(keep[i] if p[i] < p[j] else keep[j])
    else:
        labels = fcluster(linkage(pdist(K, "cosine"), "average"), t=1 - threshold, criterion="distance")
        for lab in np.unique(labels):
            members = np.where(labels == lab)[0]
            best = members[int(np.argmax(p[members]))]
            removed |= {keep[m] for m in members if m != best}
    return [q for q in keep if q not in removed]
