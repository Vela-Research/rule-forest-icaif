"""Leakage-free re-run of the founder RRF ablation panels (b, c, e).

The leak (verified): filter_questions sorts the question matrix by a precomputed
F0.5 column computed over ALL founders (incl. the held-out CV test split) and the
nested CV then sums the top-n_q in that GLOBAL order. Fix: re-rank questions by
TRAINING-fold F0.5 inside each fold; everything else (the dedup, the n_q/threshold
tuning via the actual scoring_utils functions) is unchanged.

We reuse the pipeline's own select_best_hyperparams + evaluate_test_fold so that
"global" mode reproduces the published numbers exactly (validation), and only the
question ORDER differs in "perfold" (leakage-free) mode.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

RRF = Path(__file__).resolve().parents[2] / "rrf_submission"
sys.path.insert(0, str(RRF))
from src.questions.question_filtering import construct_predictions_file_list, filter_questions  # noqa: E402
from src.utils.scoring_utils import (  # noqa: E402
    select_best_hyperparams, compute_weights, evaluate_test_fold, f_beta_score,
)

N_REPEATS = 10
N_SPLITS = 10
SCORE_THRESHOLDS = list(range(1, 71))
_BASE_META = ["Index", "Question", "Pass Rate", "Prec", "TP", "FP", "TN", "FN", "Rec", "F1",
              "F0.5", "Prec_Train", "Prec_Validation", "Prec_Test", "Prec_Mean"]
# Phantom-free by default: also exclude the all-NaN 'Source' tag that filter_questions
# injects. Left in, it becomes an all-negative phantom founder column (matrix F = 8174
# = 8173 real + 1 phantom) and shifts the founder-stratified CV partition. Removing it
# is the canonical correction (founder F0.5 0.105 -> 0.102). Set RRF_LEGACY_PHANTOM=1
# to KEEP it and reproduce the legacy phantom-included numbers (perfold 0.105 / global
# 0.118 = the original published anchor). See reproduction/PHANTOM_CORRECTION.md.
META = _BASE_META if os.environ.get("RRF_LEGACY_PHANTOM") == "1" else _BASE_META + ["Source"]


def build_matrix(model, metric, threshold):
    pred_dir = RRF / "precomputed" / model / "test_predictions"
    args = argparse.Namespace(mode="llm", similarity_metric=metric,
                              similarity_threshold=threshold, optimise_for="f0.5", sort_by="f0.5")
    try:
        files = construct_predictions_file_list(args, "_anonymised", predictions_dir=pred_dir)
    except FileNotFoundError:
        # single-file models (gemini/claude/gpt5.2): no canonical predictions_test_all
        avail = sorted(pred_dir.glob("predictions_test_set_*anonymised.csv"))
        files = [avail[0].name]
    df = filter_questions(pred_dir, files, "_anonymised", args, sort_by="f0.5")
    qr = df[~df["Question"].isin(["Founder Index", "Dataset", "Success"])].copy()
    fcols = [c for c in qr.columns if c not in META]
    qr[fcols] = qr[fcols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    X = qr[fcols].values  # (Q, F) in GLOBAL f0.5 order
    succ = df[df["Question"] == "Success"][fcols].iloc[0]
    y = pd.to_numeric(succ, errors="coerce").fillna(0).astype(int).values
    return X, y


def q_f05(y_tr, row):
    tp = int(((row == 1) & (y_tr == 1)).sum()); fp = int(((row == 1) & (y_tr == 0)).sum())
    fn = int(((row == 0) & (y_tr == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 0.0 if p + r == 0 else 1.25 * p * r / (0.25 * p + r)


def run_nested(X, y, mode):
    """mode='global' (rows kept in filter_questions order) or 'perfold' (re-rank by train F0.5)."""
    Q, F = X.shape
    fidx = np.arange(F)
    per_fold = []
    for repeat in range(1, N_REPEATS + 1):
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=42 + (repeat - 1))
        for outer_idx, (tr, te) in enumerate(skf.split(fidx, y), start=1):
            train_ids, test_ids = fidx[tr], fidx[te]
            y_train = y[train_ids]
            if mode == "perfold":
                order = np.argsort([q_f05(y_train, X[q, train_ids]) for q in range(Q)])[::-1]
            else:
                order = np.arange(Q)
            Xo = X[order, :]
            best_combo, prec_array, ratio_array, _ = select_best_hyperparams(
                Xo[:, train_ids], y_train, train_ids, 0, [0.0], SCORE_THRESHOLDS,
                Q, y, N_SPLITS, outer_idx + repeat, optimise_for="f0.5",
            )
            exp_opt, nq_opt, t_opt = best_combo
            W_full, _ = compute_weights(0, exp_opt, prec_array, ratio_array)
            preds, y_te, tp, fp, tn, fn, *_ = evaluate_test_fold(Xo, W_full, nq_opt, t_opt, test_ids, y)
            per_fold.append({"repeat": repeat, "fold": outer_idx, "n_q": nq_opt, "thr": t_opt,
                             "TP": tp, "FP": fp, "FN": fn,
                             "f05": f_beta_score(tp / (tp + fp) if tp + fp else 0.0,
                                                 tp / (tp + fn) if tp + fn else 0.0)})
    return per_fold


def _f05c(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 0.0 if p + r == 0 else 1.25 * p * r / (0.25 * p + r)


def agg_f05(per_fold):
    """Figure aggregation (notebook extract_metrics_from_file + compute_avg_metrics):
    per-repeat-pooled F0.5 = pool TP/FP/FN within each repeat, then mean over repeats."""
    df = pd.DataFrame(per_fold)
    per_rep = [_f05c(g.TP.sum(), g.FP.sum(), g.FN.sum()) for _, g in df.groupby("repeat")]
    return float(np.mean(per_rep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, help="JSON list of [model,metric,threshold]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--validate-global", action="store_true",
                    help="also run global mode to confirm reproduction")
    args = ap.parse_args()
    configs = json.loads(args.configs)
    results = []
    for model, metric, threshold in configs:
        t0 = time.time()
        X, y = build_matrix(model, metric, threshold)
        rec = {"model": model, "metric": metric, "threshold": threshold, "Q": int(X.shape[0]),
               "F": int(X.shape[1]), "pos": int(y.sum())}
        if args.validate_global:
            g = run_nested(X, y, "global")
            rec["repro_global_f05"] = agg_f05(g)
        pf = run_nested(X, y, "perfold")
        rec["leakfree_f05"] = agg_f05(pf)
        rec["leakfree_nq_median"] = float(np.median([d["n_q"] for d in pf]))
        results.append(rec)
        print(f"[{model} {metric} {threshold}] Q={X.shape[0]} "
              f"repro={rec.get('repro_global_f05')} leakfree={rec['leakfree_f05']:.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
