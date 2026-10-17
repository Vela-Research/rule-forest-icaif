"""Rule Forest's equal-weight vote and its nested cross-validation.

Within one training set the vote
1. builds the candidate pool (pool.candidate_pool) from the training founders,
2. ranks the pool's questions by their training F0.5,
3. picks N (how many top questions to keep) and T (how many must be YES) by an inner
   stratified K-fold over the training founders, maximising the mean F0.5 of the inner
   validation folds,
4. predicts success for a founder when at least T of the top N questions are YES.

Ties keep the smallest N, then the smallest T. Step 3 scores the ranking from step 2 on each
inner validation fold, as the original implementation does.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .pool import candidate_pool, f05

def rank(A, y):
    """Question order by training F0.5, highest first; ties keep their row order."""
    return np.argsort(-f05(A, y), kind="stable")


def f05_from_counts(tp, fp, n_pos):
    """F0.5 = 1.25 P R / (0.25 P + R), elementwise, 0 where P + R = 0."""
    tp = np.asarray(tp)
    zero = np.zeros(tp.shape)
    p = np.divide(tp, tp + fp, out=zero.copy(), where=(tp + fp) > 0)
    r = tp / n_pos if n_pos > 0 else zero.copy()
    return np.divide(1.25 * p * r, 0.25 * p + r, out=zero.copy(), where=(p + r) != 0)


def tune(A, y, inner_splits, seed):
    """(N, T, score) maximising the mean inner-fold F0.5 for ranked training answers A (Q, n)."""
    Q = len(A)
    S = np.cumsum(A, axis=0, dtype=np.int64)                  # S[N-1] = YES votes among the top N
    t = np.arange(1, Q + 1)
    folds = StratifiedKFold(inner_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
    scores = []
    for _, val in folds:
        Sv, yv = S[:, val], y[val]
        tp = np.zeros((Q, Q), np.int64)
        fp = np.zeros((Q, Q), np.int64)
        for n in range(Q):                                     # founders with >= T votes, per T
            tp[n] = np.bincount(Sv[n, yv == 1], minlength=Q + 1)[::-1].cumsum()[::-1][t]
            fp[n] = np.bincount(Sv[n, yv == 0], minlength=Q + 1)[::-1].cumsum()[::-1][t]
        scores.append(f05_from_counts(tp, fp, int((yv == 1).sum())))
    mean = np.mean(np.stack(scores, axis=-1), axis=-1)
    best = int(np.argmax(mean))
    return best // Q + 1, best % Q + 1, float(mean.flat[best])


@dataclass
class Fit:
    pool: list         # candidate pool, row indices of X
    selected: list     # the top N questions, row indices of X, in rank order
    N: int
    T: int
    train_f05: float

    def votes(self, X, cols=None):
        """YES answers among the selected questions, for columns `cols` of X (default all)."""
        A = X[self.selected]
        return (A if cols is None else A[:, cols]).sum(0)

    def predict(self, X, cols=None):
        return (self.votes(X, cols) >= self.T).astype(np.int8)

    def scorecard(self, questions):
        """The fitted rule as text."""
        lines = [f"Predict success when at least {self.T} of these {self.N} questions are answered YES:"]
        return "\n".join(lines + [f"  {i}. {questions[q]}" for i, q in enumerate(self.selected, 1)])


def fit(X, y, train, inner_splits=10, seed=0, pool=None, **pool_kw):
    """Fit the vote on the training founders `train` (column indices of X). `pool` fixes the
    candidate pool; otherwise it is built from the training founders with `pool_kw`."""
    train = np.asarray(train)
    rows = candidate_pool(X, y, train, **pool_kw) if pool is None else list(pool)
    if not rows:
        raise ValueError("no question passes the precision floor on these training examples")
    A, yt = X[rows][:, train], y[train]
    order = rank(A, yt)
    N, T, _ = tune(A[order], yt, inner_splits, seed)
    sel = [rows[k] for k in order[:N]]
    hit = X[sel][:, train].sum(0) >= T
    train_f05 = float(f05_from_counts(np.array((hit & (yt == 1)).sum()), (hit & (yt == 0)).sum(),
                                      int((yt == 1).sum())))
    return Fit(pool=rows, selected=sel, N=N, T=T, train_f05=train_f05)


def outer_folds(y, n_repeats=10, n_splits=10):
    """(repeat, fold, train, test) for repeated stratified K-fold, seed 42 + repeat - 1."""
    for rep in range(1, n_repeats + 1):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=42 + rep - 1)
        for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
            yield rep, k, tr, te


def nested_cv(X, y, n_repeats=10, n_splits=10, inner_splits=10, pools=None, **pool_kw):
    """Fit the vote in each outer training fold and predict its test fold. The inner seed is
    fold + repeat, as in the original runs (so some folds share an inner split seed). `pools`
    maps (repeat, fold) to a fixed candidate pool."""
    out = []
    for rep, k, tr, te in outer_folds(y, n_repeats, n_splits):
        f = fit(X, y, tr, inner_splits, k + rep, None if pools is None else pools[(rep, k)], **pool_kw)
        v = f.votes(X, te)
        out.append(record(rep, k, te, (v >= f.T).astype(np.int8), y[te], votes=v, N=f.N, T=f.T,
                          pool=f.pool, selected=f.selected, train_f05=f.train_f05))
    return out


def record(rep, k, test, pred, y_test, **extra):
    """One outer fold's test predictions and confusion counts."""
    return {"repeat": rep, "fold": k, "test": test, "pred": pred,
            "TP": int(((pred == 1) & (y_test == 1)).sum()), "FP": int(((pred == 1) & (y_test == 0)).sum()),
            "FN": int(((pred == 0) & (y_test == 1)).sum()), **extra}
