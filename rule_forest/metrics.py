"""Scoring and the paired founder bootstrap.

Pooled scoring (every cross-validated row in the paper). Within each repeat, the TP, FP and FN
of its 10 test folds are summed, precision, recall and F0.5 are computed from those sums, and
the reported value is the mean (and SD) over the repeats. A test fold with no flagged success
therefore does not score 0 on its own.
"""
from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd


def prf(tp, fp, fn):
    """Precision, recall and F0.5 from counts (0 where undefined)."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (0.0 if p + r == 0 else 1.25 * p * r / (0.25 * p + r))


def best_cut(y, scores):
    """Score value that maximises F0.5 on (y, scores) when used as a cut (score >= cut); the
    first such value on ties."""
    best_t, best = float(scores.min()) - 1.0, -1.0
    for t in np.unique(scores):
        yp = scores >= t
        s = prf(int((yp & (y == 1)).sum()), int((yp & (y == 0)).sum()), int((~yp & (y == 1)).sum()))[2]
        if s > best:
            best, best_t = s, float(t)
    return best_t


def pooled(records, group="repeat"):
    """Mean and SD over repeats of the pooled precision, recall and F0.5. `records` are outer
    folds with TP, FP and FN; `group` names the column(s) that identify one repeat."""
    df = pd.DataFrame(records)
    per_repeat = [prf(int(g.TP.sum()), int(g.FP.sum()), int(g.FN.sum())) for _, g in df.groupby(group)]
    out = {}
    for i, k in enumerate(("precision", "recall", "f05")):
        v = [r[i] for r in per_repeat]
        out[k] = float(np.mean(v))
        out[k + "_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    out["repeats"] = len(per_repeat)
    return out


def prediction_matrix(records, n):
    """(repeats, n) test predictions from outer-fold records that hold `test` and `pred`."""
    reps = sorted({r["repeat"] for r in records})
    M = np.full((len(reps), n), -1, dtype=np.int8)
    for r in records:
        M[reps.index(r["repeat"]), r["test"]] = r["pred"]
    if (M < 0).any():
        raise ValueError("every founder must be tested once per repeat")
    return M


def pooled_stat(M, y, w, metric="f05"):
    """Pooled statistic of prediction matrix M (repeats, n) or vector (n,) with founder weights w."""
    M = np.atleast_2d(M)
    pos = y == 1
    tp = ((M == 1) & pos) @ w
    fp = ((M == 1) & ~pos) @ w
    fn = ((M == 0) & pos) @ w
    p = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    r = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    if metric == "precision":
        v = p
    elif metric == "recall":
        v = r
    else:
        d = 0.25 * p + r
        v = np.divide(1.25 * p * r, d, out=np.zeros_like(d), where=d > 0)
    return float(v.mean())


def paired_bootstrap(A, B, y, metric="f05", n_boot=10_000, seed=42):
    """Difference stat(A) - stat(B) with a 95% percentile interval and a two-sided p-value, from
    resampling founders with replacement (both arms keep the same resampled founders)."""
    rng = np.random.RandomState(seed)
    n = len(y)
    one = np.ones(n)
    a, b = pooled_stat(A, y, one, metric), pooled_stat(B, y, one, metric)
    d = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.randint(0, n, n), minlength=n).astype(float)
        d[i] = pooled_stat(A, y, w, metric) - pooled_stat(B, y, w, metric)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return {"metric": metric, "a": a, "b": b, "delta": a - b, "ci95": [float(lo), float(hi)],
            "p": float(p), "n": int(n), "resamples": n_boot, "seed": seed}


def save_npz(path, arrays):
    """np.savez_compressed without the write time inside, so identical arrays give identical files."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, value in arrays.items():
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with zf.open(info, "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.asanyarray(value), allow_pickle=False)
