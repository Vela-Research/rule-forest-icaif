"""GPTree and RRM rows of Table 2, by re-scoring each method's frozen LLM output. No API calls.

GPTree (an LLM-built decision tree) and RRM (LLM rule mining) were each run three times
(generation seeds 42, 43, 44) on the question-generation founders. For the 8,204 evaluation
founders the release keeps each run's output:
  data/gptree/gptree_seed*_leaves.csv  the leaf of the fixed tree that each founder falls in
  data/rrm/rrm_seed*_scores.json       each founder's score from the fixed rule set
Each run is scored under 10 x 10 cross-validation (outer seeds 42 + repeat - 1). In every outer
fold only the data-dependent part is refitted on the training founders: GPTree's P(success | leaf)
and, for both methods, the score threshold that maximises training F0.5. Scores are pooled within
each repeat, so each method has 30 repeats (3 runs x 10); the table reports their mean and SD.

  python baselines/gptree_rrm/rescore.py   # -> results/gptree_rrm.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from rule_forest.metrics import best_cut, pooled, prf   # noqa: E402
from rule_forest.vote import outer_folds, record        # noqa: E402

DATA = HERE / "data"
SEEDS = [42, 43, 44]


def cross_validate(score_of_fold, y, seed):
    out = []
    for rep, k, tr, te in outer_folds(y):
        s_tr, s_te = score_of_fold(tr, te)
        pred = (s_te >= best_cut(y[tr], s_tr)).astype(np.int8)
        out.append(record(rep, k, te, pred, y[te], seed=seed))
    return out


def rrm():
    y = np.array(json.loads((DATA / "eval_labels.json").read_text()))
    recs = []
    for s in SEEDS:
        sc = json.loads((DATA / f"rrm/rrm_seed{s}_scores.json").read_text())
        scores = np.array([sc[str(i)] for i in range(len(y))], float)
        recs += cross_validate(lambda tr, te, v=scores: (v[tr], v[te]), y, s)
    return recs


def gptree():
    recs = []
    for s in SEEDS:
        df = pd.read_csv(DATA / f"gptree/gptree_seed{s}_leaves.csv")
        leaf = df["leaf_node_id"].to_numpy()
        y = (df["true_label"].astype(str).str.strip().str.lower() == "successful").astype(int).to_numpy()

        def score_of_fold(tr, te, leaf=leaf, y=y):
            rate = pd.Series(y[tr]).groupby(leaf[tr]).mean()
            base = y[tr].mean()
            return (np.array([rate.get(v, base) for v in leaf[tr]]), np.array([rate.get(v, base) for v in leaf[te]]))
        recs += cross_validate(score_of_fold, y, s)
    return recs


def summary(recs):
    per_fold = [prf(r["TP"], r["FP"], r["FN"]) for r in recs]
    return {"pooled": pooled(recs, group=["seed", "repeat"]),
            "per_fold_mean": dict(zip(("precision", "recall", "f05"), map(float, np.mean(per_fold, axis=0)))),
            "note": "pooled: mean and SD over 30 repeats (3 runs x 10). per_fold_mean: the mean over 300 folds "
                    "that the submitted paper reported."}


def main():
    y = np.array(json.loads((DATA / "eval_labels.json").read_text()))
    out = {"GPTree": summary(gptree()), "RRM": summary(rrm()), "founders": int(len(y)), "successes": int(y.sum())}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "gptree_rrm.json").write_text(json.dumps(out, indent=1) + "\n")
    for m in ("GPTree", "RRM"):
        p = out[m]["pooled"]
        print(f"{m:7s} P {p['precision']:.3f} ± {p['precision_sd']:.3f}  R {p['recall']:.3f} ± {p['recall_sd']:.3f}  "
              f"F0.5 {p['f05']:.3f} ± {p['f05_sd']:.3f}")


if __name__ == "__main__":
    main()
