"""Shared machinery for the Figure 4 and Figure 5 sweeps.

One draw = a seeded subsample of successes and failures, on which the vote and the sweep elastic
net are fitted under the same outer folds (10 x 10, seeds 42 + repeat - 1), each tuned by 5 inner
folds. Draws run in parallel and are appended to a JSON-lines file, so a run can be resumed.
"""
from __future__ import annotations

import hashlib
import json
import time
from multiprocessing import Pool

import numpy as np

from rule_forest import elastic_net, metrics, vote

INNER = 5


def draw_seed(*parts):
    """Subsample seed from a SHA-256 of the arguments (independent of PYTHONHASHSEED)."""
    s = "|".join(str(p) for p in parts)
    return (int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**31 - 1)) or 1


def subsample(y, n_pos, n_neg, seed):
    """Sorted column indices of n_pos random successes and n_neg random failures."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if n_pos >= len(pos) and n_neg >= len(neg):
        return np.arange(len(y))
    rng = np.random.RandomState(seed)
    return np.sort(np.concatenate([rng.choice(pos, size=min(n_pos, len(pos)), replace=False),
                                   rng.choice(neg, size=min(n_neg, len(neg)), replace=False)]))


def mean_jaccard(sets):
    """Mean pairwise Jaccard overlap of a list of sets."""
    sets = [set(s) for s in sets]
    v = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:] if a | b]
    return float(np.mean(v)) if v else float("nan")


def compare(X, y, pools):
    """Vote and sweep elastic net on (X, y) with the given per-fold pools. Returns a summary."""
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    inner = max(2, min(INNER, n_pos, n_neg))
    folds = list(vote.outer_folds(y))
    v = vote.nested_cv(X, y, inner_splits=inner, pools=pools)
    e = elastic_net.nested_cv(X, y, "sweep", inner_splits=INNER, pools=pools, folds=folds)
    pv, pe = metrics.pooled(v), metrics.pooled(e)
    nz = np.array([r["coef"] for r in e]) != 0
    sizes = [len(p) for p in pools.values()]
    return {"vote_f05": pv["f05"], "en_f05": pe["f05"], "gap": pv["f05"] - pe["f05"],
            "vote_precision": pv["precision"], "vote_recall": pv["recall"],
            "en_precision": pe["precision"], "en_recall": pe["recall"],
            "vote_optimism": float(np.mean([r["train_f05"] for r in v]) - pv["f05"]),
            "en_optimism": float(np.mean([r["train_f05"] for r in e]) - pe["f05"]),
            "vote_selection_jaccard": mean_jaccard([r["selected"] for r in v]),
            "en_support_jaccard": mean_jaccard([np.where(row)[0] for row in nz]),
            "vote_N_median": float(np.median([r["N"] for r in v])),
            "pool_size": [int(min(sizes)), float(np.median(sizes)), int(max(sizes))]}


def run(tasks, work, init, out_path, jobs, fresh=False):
    """Evaluate the tasks not yet in out_path, `jobs` at a time, appending one line per draw.
    fresh=True discards out_path first, so every draw is recomputed."""
    if fresh and out_path.exists():
        out_path.unlink()
    done = set()
    if out_path.exists():
        done = {tuple(json.loads(l)["key"]) for l in out_path.read_text().splitlines() if l.strip()}
    todo = [t for t in tasks if tuple(t["key"]) not in done]
    print(f"{len(todo)} draws to run ({len(done)} already in {out_path.name})", flush=True)
    if not todo:
        return
    t0 = time.time()
    out_path.parent.mkdir(exist_ok=True)
    with Pool(min(jobs, len(todo)), initializer=init) as pool, open(out_path, "a") as fh:
        for i, r in enumerate(pool.imap_unordered(work, todo), 1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"[{i}/{len(todo)} {time.time() - t0:.0f}s] {r['key']} vote {r['vote_f05']:.3f} "
                  f"en {r['en_f05']:.3f}", flush=True)
    out_path.write_text("".join(json.dumps(r) + "\n" for r in read(out_path)))    # fixed line order


def read(out_path):
    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    return sorted(rows, key=lambda r: r["key"])
