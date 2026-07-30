"""Legit (leakage-free) question-generating-model ablation, computed from scratch.

Ignores the unreproducible published model-ablation CSVs. For each model we take
its FULL DISTINCT question set (exact-duplicate removal only — deterministic, no
similarity-dedup drift), then run the same leakage-free 10x10 vote (questions
ranked by TRAINING-fold F0.5 inside each fold; n_q/threshold tuned on train).
Per-repeat-pooled F0.5/Precision/Recall.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

from run_ablation import RRF, run_nested
from full_leakfree import agg_prf

META = {"Index", "Question", "Pass Rate", "Prec", "TP", "FP", "TN", "FN", "Rec", "F1", "F0.5",
        "Prec_Train", "Prec_Validation", "Prec_Test", "Prec_Mean", "Source", "Dataset", "SUCCESS_PROPORTION"}
SPECIAL = {"Founder Index", "Dataset", "Success", "SUCCESS_PROPORTION"}


def build_distinct_matrix(model):
    """All non-EXPERT prediction files -> stacked question rows -> drop exact-dup
    question vectors. Returns X (Q_distinct, F), y, raw_q, distinct_q."""
    pred_dir = RRF / "precomputed" / model / "test_predictions"
    files = sorted(f for f in pred_dir.glob("predictions_test_*anonymised.csv") if "EXPERT" not in f.name)
    # prefer the canonical merged file if present (gpt_4o_mini), else all set files
    allf = [f for f in files if f.name == "predictions_test_all_anonymised.csv"]
    use = allf if allf else [f for f in files if "test_all" not in f.name]
    qrows, founder_cols, y = [], None, None
    for f in use:
        d = pd.read_csv(f)
        fc = [c for c in d.columns if c not in META]
        if founder_cols is None:
            founder_cols = fc
            srow = d[d["Question"] == "Success"]
            y = pd.to_numeric(srow[founder_cols].iloc[0], errors="coerce").fillna(0).astype(int).values
        q = d[~d["Question"].astype(str).isin(SPECIAL)]
        qm = q[founder_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        qrows.append(qm)
    M = pd.concat(qrows, ignore_index=True)
    raw_q = len(M)
    M = M.drop_duplicates()  # exact-duplicate question vectors
    X = M.values
    return X, y, raw_q, X.shape[0]


def main():
    out = []
    for model in ["gpt_4o_mini", "gemini_2_5_pro", "claude_sonnet_4_5", "gpt_5_2"]:
        t0 = time.time()
        X, y, raw_q, Qd = build_distinct_matrix(model)
        pf = run_nested(X, y, "perfold")
        m = agg_prf(pf)
        m.update({"model": model, "raw_questions": int(raw_q), "distinct_questions": int(Qd),
                  "founders": int(X.shape[1]), "pos": int(y.sum()),
                  "nq_median": float(np.median([d["n_q"] for d in pf]))})
        out.append(m)
        print(f"{model:18s} Q_distinct={Qd:2d} (raw {raw_q}) F={X.shape[1]} pos={int(y.sum())} | "
              f"F0.5={m['f05']:.4f}±{m['f05_sd']:.4f} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"nq_med={m['nq_median']:.0f} [{time.time()-t0:.0f}s]", flush=True)
        Path("results_model_ablation_legit.json").write_text(json.dumps(out, indent=2))
    print("\nWrote results_model_ablation_legit.json", flush=True)


if __name__ == "__main__":
    main()
