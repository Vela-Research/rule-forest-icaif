"""Release-relative data loaders for the count-effect figures.

Every loader returns (X, y):
  X : binary question matrix (Q, F) = (n_questions, n_instances) -- the orientation the
      Rule Forest nested-CV expects.
  y : int {0,1} labels, shape (F,).

Inputs are all shipped under count_effect/data/ (+ the founder question matrices under
rrf_submission/precomputed/, via run_ablation.build_matrix for the deduped 22-question set).
Kiva probe scores are thresholded at 0.5 (locked); error/None rows dropped; de-duped by id.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RELEASE = HERE.parents[2]
SCORE_THRESHOLD = 0.5


def _load_jsonl_matrix(path: Path, id_key: str = "id"):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("scores"):          # None/empty -> errored probe call
                continue
            rows.append(r)
    seen = {}
    for r in rows:                            # de-dupe by id, first valid wins
        seen.setdefault(r.get(id_key), r)
    rows = list(seen.values())
    scores = np.asarray([r["scores"] for r in rows], dtype=float)      # (F, Q)
    y = np.asarray([int(r["label"]) for r in rows], dtype=int)         # (F,)
    X = (scores >= SCORE_THRESHOLD).astype(int).T.copy()              # (Q, F)
    return X, y


def load_founders():
    """Full 62-question / 9192-founder matrix (natural base rate 4.96%). Panel (b)/(c)."""
    ans = pd.read_parquet(DATA / "founders_answers.parquet")          # (9192, 62) YES/NO
    dat = pd.read_parquet(DATA / "founders_data.parquet")             # (9192, 2): summary, y
    assert len(ans) == len(dat), "founder answers/labels row mismatch"
    Xbin = ans.map(lambda v: 1 if str(v).strip().upper() == "YES" else 0).values.astype(int)
    y = (dat["y"].astype(str).str.strip().str.upper() == "YES").astype(int).values
    return Xbin.T.copy(), y                                            # (62, 9192), (9192,)


def load_kiva():
    """14-question / 5000-loan Kiva matrix (natural base rate 4.44%). Figure 5."""
    return _load_jsonl_matrix(DATA / "kiva_rq_scores_run1.jsonl", "id")


def load_founders_22q():
    """22-question DEDUPED founder test matrix at the real 1.9% operating rate. Panel (a).

    Built by the shipped filter_questions dedup chain (run_ablation.build_matrix), identical
    to the Table-2 Rule Forest matrix -> (22, ~8173), 158 positives."""
    sys.path.insert(0, str(RELEASE / "baselines" / "leakage_free_ablation"))
    import run_ablation as ra
    X, y = ra.build_matrix("gpt_4o_mini", "hamming", 0.15)
    return np.asarray(X), np.asarray(y)


if __name__ == "__main__":
    for name, fn in [("founders(62)", load_founders), ("kiva(14)", load_kiva),
                     ("founders_22q", load_founders_22q)]:
        X, y = fn()
        print(f"{name:14s} Q={X.shape[0]:2d} F={X.shape[1]:5d} n_pos={int(y.sum()):4d} "
              f"base={y.mean():.4f}")
