"""Comprehensive LEAKAGE-FREE recomputation of the founder RRF results, computed
from scratch with per-training-fold question ranking. This is *our own* corrected
analysis (not an attempt to match the published cells).

Every RRF number = the pipeline's nested CV (select_best_hyperparams +
evaluate_test_fold) with ONE change: questions are ranked by TRAINING-fold F0.5
inside each fold instead of the global full-data F0.5 order. Metrics are
per-repeat-pooled F0.5/Precision/Recall (the figure's aggregation).

Covers ablation panels a-e + the main-table RRF row. Baselines (LogReg-EN,
Engineered ML, Embeddings, prompting) carry no F0.5-sort leak and are unchanged.
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import pandas as pd

import run_ablation as ra
from run_ablation import RRF, META, run_nested, _f05c
sys.path.insert(0, str(RRF))
from src.questions.question_filtering import construct_predictions_file_list, filter_questions


def build_matrix(model, mode, metric, threshold):
    pred_dir = RRF / "precomputed" / model / "test_predictions"
    args = argparse.Namespace(mode=mode, similarity_metric=metric,
                              similarity_threshold=threshold, optimise_for="f0.5", sort_by="f0.5")
    try:
        files = construct_predictions_file_list(args, "_anonymised", predictions_dir=pred_dir)
    except FileNotFoundError:
        avail = sorted(p.name for p in pred_dir.glob("predictions_test_set_*anonymised.csv")
                       if "EXPERT" not in p.name)
        files = avail[:1]  # single-batch models (gemini/claude/gpt5.2)
    df = filter_questions(pred_dir, files, "_anonymised", args, sort_by="f0.5")
    qr = df[~df["Question"].isin(["Founder Index", "Dataset", "Success"])].copy()
    fcols = [c for c in qr.columns if c not in META]
    qr[fcols] = qr[fcols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    X = qr[fcols].values
    y = pd.to_numeric(df[df["Question"] == "Success"][fcols].iloc[0], errors="coerce").fillna(0).astype(int).values
    return X, y


def agg_prf(per_fold):
    """per-repeat-pooled mean +/- SD for Precision / Recall / F0.5."""
    df = pd.DataFrame(per_fold)
    P, R, F = [], [], []
    for _, g in df.groupby("repeat"):
        tp, fp, fn = g.TP.sum(), g.FP.sum(), g.FN.sum()
        P.append(tp / (tp + fp) if tp + fp else 0.0)
        R.append(tp / (tp + fn) if tp + fn else 0.0)
        F.append(_f05c(tp, fp, fn))
    sd = lambda a: float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    return {"precision": float(np.mean(P)), "precision_sd": sd(P),
            "recall": float(np.mean(R)), "recall_sd": sd(R),
            "f05": float(np.mean(F)), "f05_sd": sd(F)}


def chronological_leakfree(model="gpt_4o_mini", metric="hamming", threshold=0.15):
    """Single chronological split: train=first 6362, test=last 1812. Questions ranked
    by TRAIN-only F0.5; n_q/thr tuned on train (inner CV); evaluated on the held-out tail."""
    from src.utils.scoring_utils import select_best_hyperparams, compute_weights, evaluate_test_fold
    X, y = build_matrix(model, "llm", metric, threshold)
    Q, F = X.shape
    fidx = np.arange(F)
    test_ids, train_ids = fidx[-1812:], fidx[:-1812]
    y_train = y[train_ids]
    order = np.argsort([ra.q_f05(y_train, X[q, train_ids]) for q in range(Q)])[::-1]
    Xo = X[order]
    bc, pa, rar, _ = select_best_hyperparams(Xo[:, train_ids], y_train, train_ids, 0, [0.0],
                                             list(range(1, 71)), Q, y, 10, 1, optimise_for="f0.5")
    e, nq, t = bc
    W, _ = compute_weights(0, e, pa, rar)
    preds, yte, tp, fp, tn, fn, *_ = evaluate_test_fold(Xo, W, nq, t, test_ids, y)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f05": _f05c(tp, fp, fn),
            "n_q": int(nq), "thr": int(t), "TP": int(tp), "FP": int(fp), "FN": int(fn), "single_fold": True}


CONFIGS = [
    # (panel, label, model, mode, metric, threshold)
    ("a", "LLM-Only", "gpt_4o_mini", "llm", "hamming", 0.15),
    ("a", "LLM+Expert", "gpt_4o_mini", "llm_expert", "hamming", 0.01),
    ("a", "Expert-Only", "gpt_4o_mini", "expert_only", "cosine-cluster", 0.85),
    ("b", "hamming 0.15", "gpt_4o_mini", "llm", "hamming", 0.15),
    ("b", "jaccard 0.90", "gpt_4o_mini", "llm", "jaccard", 0.90),
    ("b", "cosine 0.85", "gpt_4o_mini", "llm", "cosine-cluster", 0.85),
    ("c", "hamming 0.10", "gpt_4o_mini", "llm", "hamming", 0.10),
    ("c", "hamming 0.20", "gpt_4o_mini", "llm", "hamming", 0.20),
    ("e", "Gemini 2.5 Pro", "gemini_2_5_pro", "llm", "hamming", 0.10),
    ("e", "Claude Sonnet 4.5", "claude_sonnet_4_5", "llm", "hamming", 0.30),
    ("e", "GPT-5.2", "gpt_5_2", "llm", "hamming", 0.20),
]


def main():
    out = {"cells": [], "note": "leakage-free (per-training-fold question ranking); per-repeat-pooled metrics"}
    for panel, label, model, mode, metric, threshold in CONFIGS:
        t0 = time.time()
        try:
            X, y = build_matrix(model, mode, metric, threshold)
            pf = run_nested(X, y, "perfold")
            m = agg_prf(pf)
            m.update({"panel": panel, "label": label, "model": model, "mode": mode,
                      "metric": metric, "threshold": threshold, "Q": int(X.shape[0]),
                      "nq_median": float(np.median([d["n_q"] for d in pf]))})
        except Exception as ex:
            m = {"panel": panel, "label": label, "model": model, "error": str(ex)[:200]}
        out["cells"].append(m)
        print(f"[{panel}] {label:18s} F0.5={m.get('f05', float('nan')):.4f} "
              f"P={m.get('precision', float('nan')):.4f} R={m.get('recall', float('nan')):.4f} "
              f"Q={m.get('Q','?')} [{time.time()-t0:.0f}s]", flush=True)
        Path("results_full_leakfree.json").write_text(json.dumps(out, indent=2))
    # panel d chronological
    try:
        chron = chronological_leakfree()
        chron.update({"panel": "d", "label": "chronological (leakage-free)"})
        out["cells"].append(chron)
        print(f"[d] chronological      F0.5={chron['f05']:.4f} P={chron['precision']:.4f} "
              f"R={chron['recall']:.4f} n_q={chron['n_q']} (SINGLE FOLD)", flush=True)
    except Exception as ex:
        print(f"[d] chronological error: {ex}", flush=True)
    Path("results_full_leakfree.json").write_text(json.dumps(out, indent=2))
    print("\nWrote results_full_leakfree.json", flush=True)


if __name__ == "__main__":
    main()
