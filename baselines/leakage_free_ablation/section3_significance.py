"""SECTION 3 (paper 5.1) significance test.

Founder-level paired bootstrap of the F0.5 GAP that Table 2 actually reports -- the
per-repeat-pooled F0.5 (pool the folds within each repeat -> F0.5 per repeat -> mean over the
10 repeats). NO majority-vote "consensus" collapse anywhere (the previous version bootstrapped a
majority vote over the 10 repeats, which is not the reported metric and inflated some baselines).

Rule Forest: leakage-free gpt_4o_mini hamming-0.15 vote, questions ranked by TRAINING-fold F0.5;
the full 10-repeat x founder prediction matrix is regenerated here (cached in
results/rrf_lf_predmat.npz) and its per-repeat-pooled F0.5 == the Table-2 headline (~0.102).

Baselines:
  - o3 few-shot  : strongest baseline (Table-2 F0.5 ~0.088). Single-shot LLM prediction per
                   founder (no CV, so nothing to collapse).
  - Engineered ML: its own 10x10 per-repeat prediction matrix
                   (run ../traditional_ml_port/engineered_ml/run_engineered_ml.py first).

For each baseline: align to RRF by founder hash (dropping ambiguous 5-char prefixes), then
bootstrap Delta = F0.5(RRF) - F0.5(baseline) over founders (10,000 resamples, seed 42).
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from run_ablation import build_matrix, q_f05, RRF
from src.utils.scoring_utils import select_best_hyperparams, compute_weights, evaluate_test_fold
from src.questions.question_filtering import construct_predictions_file_list, filter_questions

HERE = RRF.parent / "baselines" / "leakage_free_ablation"
RESULTS = HERE / "results"
RRF_MAT_CACHE = RESULTS / "rrf_lf_predmat.npz"
TRAD = RRF.parent / "baselines" / "traditional_ml_port"
CONS = TRAD / "results" / "consensus_allrepeats.parquet"
ENGML_MAT = TRAD / "engineered_ml" / "engineered_ml_predmat.npy"
ENGML_CONS = TRAD / "engineered_ml" / "engineered_ml_consensus.csv"

META = {"Index", "Question", "Pass Rate", "Prec", "TP", "FP", "TN", "FN", "Rec", "F1", "F0.5",
        "Prec_Train", "Prec_Validation", "Prec_Test", "Prec_Mean", "Source"}
N_BOOT, SEED = 10000, 42


# ---- metric helpers ------------------------------------------------------------------------
def _f05(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    d = 0.25 * p + r
    return 1.25 * p * r / d if d > 0 else 0.0


def per_repeat_pooled(mat, y):
    """Table-2 metric: F0.5 pooled within each repeat, averaged over repeats. mat is (R, n)."""
    fs = []
    for r in range(mat.shape[0]):
        p = mat[r]
        fs.append(_f05(int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum()),
                      int(((p == 0) & (y == 1)).sum())))
    return float(np.mean(fs))


def f05_vec(pred, y):
    return _f05(int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()),
               int(((pred == 0) & (y == 1)).sum()))


def _stat(pred, y):
    return per_repeat_pooled(pred, y) if pred.ndim == 2 else f05_vec(pred, y)


def paired_boot(rrf, base, y, n=N_BOOT, seed=SEED):
    """rrf is (R, n); base is (R, n) or (n,). Bootstrap Delta = F0.5(rrf) - F0.5(base)."""
    rng = np.random.RandomState(seed)
    N = len(y)
    obs = _stat(rrf, y) - _stat(base, y)
    deltas = np.empty(n)
    for b in range(n):
        ix = rng.randint(0, N, N)
        yb = y[ix]
        rb = rrf[:, ix]
        bb = base[:, ix] if base.ndim == 2 else base[ix]
        deltas[b] = _stat(rb, yb) - _stat(bb, yb)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))
    return {"delta": float(obs), "ci95": [float(lo), float(hi)], "p": float(p),
            "rrf_f05": float(_stat(rrf, y)), "base_f05": float(_stat(base, y)), "n": int(N)}


# ---- RRF leakage-free per-repeat prediction matrix -----------------------------------------
def rrf_pred_matrix(model="gpt_4o_mini", thr=0.15):
    if RRF_MAT_CACHE.exists():
        z = np.load(RRF_MAT_CACHE)   # self-produced cache; plain int/string arrays (no pickle)
        return z["pred"], z["y"].astype(int), z["hashes"].astype(str)
    X, y = build_matrix(model, "hamming", thr)
    pred_dir = RRF / "precomputed" / model / "test_predictions"
    args = argparse.Namespace(mode="llm", similarity_metric="hamming", similarity_threshold=thr,
                              optimise_for="f0.5", sort_by="f0.5")
    files = construct_predictions_file_list(args, "_anonymised", predictions_dir=pred_dir)
    df = filter_questions(pred_dir, files, "_anonymised", args, sort_by="f0.5")
    qr = df[~df["Question"].isin(["Founder Index", "Dataset", "Success"])]
    hashes = np.array([c for c in qr.columns if c not in META])
    Q, F = X.shape
    pred = np.full((10, F), -1, dtype=int)
    for r in range(10):
        skf = StratifiedKFold(10, shuffle=True, random_state=42 + r)
        for oi, (tr, te) in enumerate(skf.split(np.arange(F), y), start=1):
            order = np.argsort([q_f05(y[tr], X[q, tr]) for q in range(Q)])[::-1]
            Xo = X[order]
            bc, pa, ra, _ = select_best_hyperparams(Xo[:, tr], y[tr], tr, 0, [0.0],
                                                    list(range(1, 71)), Q, y, 10, oi + r + 1, optimise_for="f0.5")
            e, nq, t = bc
            W, _ = compute_weights(0, e, pa, ra)
            preds, *_ = evaluate_test_fold(Xo, W, nq, t, te, y)
            pred[r, te] = preds.astype(int)
        print(f"  [RRF] repeat {r+1}/10 done", flush=True)
    RESULTS.mkdir(exist_ok=True)
    np.savez(RRF_MAT_CACHE, pred=pred, y=y.astype(int), hashes=hashes)
    return pred, y.astype(int), hashes


def _drop_amb(*hash_arrays):
    from collections import Counter
    bad = set()
    for h in hash_arrays:
        c = Counter(h)
        bad |= {k for k, v in c.items() if v > 1}
    return bad


def main():
    print("Regenerating leakage-free RRF 10x10 prediction matrix (cached after first run) ...", flush=True)
    rrf, y, hashes = rrf_pred_matrix()
    rrf_series = pd.Series(np.arange(len(hashes)), index=hashes)

    out = {"metric": "per-repeat-pooled F0.5 (Table-2), founder-level bootstrap, "
                     f"{N_BOOT} resamples, seed {SEED}", "pairs": {}}

    # --- baseline 1: o3 few-shot (single-shot LLM prediction) ---
    o3 = pd.read_parquet(CONS)
    o3 = o3[o3.model == "o3-fewshot"][["hash", "y", "consensus"]]
    amb = _drop_amb(hashes, o3["hash"].values)
    o3 = o3[~o3["hash"].isin(amb)].drop_duplicates("hash")
    keep = [h for h in o3["hash"] if h in rrf_series.index]
    o3 = o3.set_index("hash").loc[keep]
    cols = rrf_series.loc[keep].to_numpy()
    res = paired_boot(rrf[:, cols], o3["consensus"].to_numpy().astype(int), y[cols])
    out["pairs"]["o3_fewshot"] = res
    print(f"\nRRF vs o3 few-shot (strongest baseline)  n={res['n']}  "
          f"RRF={res['rrf_f05']:.4f} base={res['base_f05']:.4f}  "
          f"Delta={res['delta']:+.4f} CI[{res['ci95'][0]:+.4f},{res['ci95'][1]:+.4f}] p={res['p']:.4f}", flush=True)

    # --- baseline 2: Engineered ML (its own 10x10 per-repeat matrix) ---
    if ENGML_MAT.exists() and ENGML_CONS.exists():
        emat = np.load(ENGML_MAT)
        eh = pd.read_csv(ENGML_CONS)["hash"].astype(str).to_numpy()
        amb = _drop_amb(hashes, eh)
        e_series = pd.Series(np.arange(len(eh)), index=eh)
        common = [h for h in eh if (h not in amb) and (h in rrf_series.index)]
        common = list(dict.fromkeys(common))
        rc = rrf_series.loc[common].to_numpy(); ec = e_series.loc[common].to_numpy()
        res = paired_boot(rrf[:, rc], emat[:, ec], y[rc])
        out["pairs"]["engineered_ml"] = res
        print(f"RRF vs Engineered ML                     n={res['n']}  "
              f"RRF={res['rrf_f05']:.4f} base={res['base_f05']:.4f}  "
              f"Delta={res['delta']:+.4f} CI[{res['ci95'][0]:+.4f},{res['ci95'][1]:+.4f}] p={res['p']:.4f}", flush=True)
    else:
        print("(Engineered ML matrix not found; run engineered_ml/run_engineered_ml.py to include it.)", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "section3_significance.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RESULTS / 'section3_significance.json'}", flush=True)


if __name__ == "__main__":
    main()
