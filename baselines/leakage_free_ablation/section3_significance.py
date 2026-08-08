"""SECTION 3 re-run: paired bootstrap with the CORRECTED (leakage-free) RRF vs
Engineered ML (R1) and Embeddings (R2), on recall / precision / F0.5.

Leakage-free RRF = gpt_4o_mini hamming-0.15 vote with questions ranked by
TRAINING-fold F0.5 (the validated harness). Per-founder 10-repeat majority-vote
consensus, same scheme as the original §5.3 averaged bootstrap. R1/R2 consensus
carry no leak and are reused from results/consensus_allrepeats.parquet.
"""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score, precision_score, recall_score

from run_ablation import build_matrix, q_f05, RRF
from src.utils.scoring_utils import select_best_hyperparams, compute_weights, evaluate_test_fold

HERE = RRF.parent / "baselines" / "leakage_free_ablation"
CONS = RRF.parent / "baselines" / "traditional_ml_port" / "results" / "consensus_allrepeats.parquet"
RRF_LF_CACHE = HERE / "results" / "rrf_lf_consensus.parquet"


def leakfree_rrf_consensus(model="gpt_4o_mini", thr=0.15):
    X, y = build_matrix(model, "hamming", thr)
    # need founder hashes: rebuild column names
    import argparse
    from src.questions.question_filtering import construct_predictions_file_list, filter_questions
    META = {"Index", "Question", "Pass Rate", "Prec", "TP", "FP", "TN", "FN", "Rec", "F1", "F0.5",
            "Prec_Train", "Prec_Validation", "Prec_Test", "Prec_Mean"}
    if os.environ.get("RRF_LEGACY_PHANTOM") != "1":
        META = META | {"Source"}  # phantom-free founder hashes (canonical)
    pred_dir = RRF / "precomputed" / model / "test_predictions"
    args = argparse.Namespace(mode="llm", similarity_metric="hamming", similarity_threshold=thr,
                              optimise_for="f0.5", sort_by="f0.5")
    files = construct_predictions_file_list(args, "_anonymised", predictions_dir=pred_dir)
    df = filter_questions(pred_dir, files, "_anonymised", args, sort_by="f0.5")
    qr = df[~df["Question"].isin(["Founder Index", "Dataset", "Success"])]
    hashes = [c for c in qr.columns if c not in META]
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
        print(f"  [RRF-lf] repeat {r} done", flush=True)
    freq = pred.mean(0)
    cons = (freq >= 0.5).astype(int)
    return pd.DataFrame({"hash": hashes, "y": y, "rrf_lf": cons})


def paired_boot(y, a, b, n=10000, seed=42):
    rng = np.random.RandomState(seed)
    N = len(y)
    obs = {"F0.5": fbeta_score(y, a, beta=0.5, zero_division=0) - fbeta_score(y, b, beta=0.5, zero_division=0),
           "precision": precision_score(y, a, zero_division=0) - precision_score(y, b, zero_division=0),
           "recall": recall_score(y, a, zero_division=0) - recall_score(y, b, zero_division=0)}
    d = {k: [] for k in obs}
    for _ in range(n):
        idx = rng.randint(0, N, N)
        yt = y[idx]
        if yt.sum() < 1:
            continue
        at, bt = a[idx], b[idx]
        d["F0.5"].append(fbeta_score(yt, at, beta=0.5, zero_division=0) - fbeta_score(yt, bt, beta=0.5, zero_division=0))
        d["precision"].append(precision_score(yt, at, zero_division=0) - precision_score(yt, bt, zero_division=0))
        d["recall"].append(recall_score(yt, at, zero_division=0) - recall_score(yt, bt, zero_division=0))
    out = {}
    for k in obs:
        arr = np.array(d[k]); lo, hi = np.percentile(arr, [2.5, 97.5])
        p = min(1.0, 2 * min((arr <= 0).mean(), (arr >= 0).mean()))
        out[k] = {"delta": float(obs[k]), "ci": [float(lo), float(hi)], "p": float(p)}
    return out


def main():
    if RRF_LF_CACHE.exists():
        print(f"Loading cached leakage-free RRF consensus from {RRF_LF_CACHE}", flush=True)
        rrf = pd.read_parquet(RRF_LF_CACHE)
    else:
        print("Generating leakage-free RRF consensus (10x10) ...", flush=True)
        rrf = leakfree_rrf_consensus()
        rrf.to_parquet(RRF_LF_CACHE, index=False)
    cons = pd.read_parquet(CONS)  # has R1, R2 (+others) consensus by hash
    out = {"rrf_lf_marginal": {}, "pairs": {}}
    # align
    for base in ["R1", "R2"]:
        b = cons[cons.model == base][["hash", "y", "consensus"]].rename(columns={"consensus": "pred", "y": "yb"})
        amb = set(b["hash"][b["hash"].duplicated(keep=False)])  # ambiguous 5-char prefixes
        b = b[~b["hash"].isin(amb)].drop_duplicates("hash")
        r = rrf[~rrf["hash"].isin(amb)].drop_duplicates("hash")
        m = r.merge(b, on="hash", how="inner")
        assert (m["y"] == m["yb"]).all(), f"label mismatch {base}"
        y = m["y"].values.astype(int)
        a = m["rrf_lf"].values.astype(int)
        bp = m["pred"].values.astype(int)
        if not out["rrf_lf_marginal"]:
            out["rrf_lf_marginal"] = {"n": int(len(y)), "pos": int(y.sum()),
                                      "F0.5": float(fbeta_score(y, a, beta=0.5, zero_division=0)),
                                      "precision": float(precision_score(y, a, zero_division=0)),
                                      "recall": float(recall_score(y, a, zero_division=0)),
                                      "pred_pos": int(a.sum())}
        res = paired_boot(y, a, bp)
        res["base_marginal"] = {"F0.5": float(fbeta_score(y, bp, beta=0.5, zero_division=0)),
                                "precision": float(precision_score(y, bp, zero_division=0)),
                                "recall": float(recall_score(y, bp, zero_division=0))}
        res["n"] = int(len(y))
        out["pairs"][base] = res
        print(f"\nRRF-lf vs {base} (n={len(y)}):  "
              + "  ".join(f"{k}: d={v['delta']:+.4f} CI[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] p={v['p']:.4f}"
                          for k, v in res.items() if k in ("F0.5", "precision", "recall")), flush=True)
    (HERE / "results" / "section3_significance.json").write_text(json.dumps(out, indent=2))
    print(f"\nRRF-lf marginal: {out['rrf_lf_marginal']}")
    print("Wrote results/section3_significance.json", flush=True)


if __name__ == "__main__":
    main()
