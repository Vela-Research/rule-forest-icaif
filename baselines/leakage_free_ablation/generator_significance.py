"""Paired founder-level bootstrap for the GENERATOR comparison (paper 5.3, Table 3):
GPT-4o-mini vote (0.102) vs GPT-5.2 vote (0.064), and Claude Sonnet 4.5 (0.083).

Same methodology as section3_significance.py (paper 5.1): per-repeat-pooled F0.5, a 10,000
founder-level bootstrap, seed 42, aligned by founder hash (ambiguous 5-char prefixes dropped).
Reports Delta = F0.5(GPT-4o-mini) - F0.5(other), its 95% CI, and both the two-sided and one-sided
bootstrap p. Reuses only shipped release code + the shipped precomputed question matrices.

  python generator_significance.py   # -> results/generator_significance.json
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, str(Path(__file__).resolve().parent))
import section3_significance as S3
from run_ablation import build_matrix, q_f05, RRF, META
from src.utils.scoring_utils import select_best_hyperparams, compute_weights, evaluate_test_fold
from src.questions.question_filtering import construct_predictions_file_list, filter_questions

RESULTS = Path(__file__).resolve().parent / "results"


def _matrix_with_hashes(model, thr):
    """Same construction as run_ablation.build_matrix, but also return founder-column hashes.
    Uses the single-file fallback for models without a canonical predictions_test_all."""
    pred_dir = RRF / "precomputed" / model / "test_predictions"
    args = argparse.Namespace(mode="llm", similarity_metric="hamming",
                              similarity_threshold=thr, optimise_for="f0.5", sort_by="f0.5")
    try:
        files = construct_predictions_file_list(args, "_anonymised", predictions_dir=pred_dir)
    except FileNotFoundError:
        avail = sorted(pred_dir.glob("predictions_test_set_*anonymised.csv"))
        files = [avail[0].name]
    df = filter_questions(pred_dir, files, "_anonymised", args, sort_by="f0.5")
    qr = df[~df["Question"].isin(["Founder Index", "Dataset", "Success"])].copy()
    fcols = [c for c in qr.columns if c not in META]
    qr[fcols] = qr[fcols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    X = qr[fcols].values
    succ = df[df["Question"] == "Success"][fcols].iloc[0]
    y = pd.to_numeric(succ, errors="coerce").fillna(0).astype(int).values
    return X, y, np.array(fcols)


def build_pred_matrix(model, thr=0.15):
    """Leakage-free 10x10 per-founder prediction matrix for `model` (no shared cache)."""
    X, y, hashes = _matrix_with_hashes(model, thr)
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
    return pred, y.astype(int), hashes


def align_and_boot(pa, ha, ya, pb, hb, yb):
    sa = pd.Series(np.arange(len(ha)), index=ha)
    sb = pd.Series(np.arange(len(hb)), index=hb)
    amb = S3._drop_amb(ha, hb)
    hbset = set(hb)
    common = list(dict.fromkeys([h for h in ha if h not in amb and h in hbset]))
    ca = sa.loc[common].to_numpy(); cb = sb.loc[common].to_numpy()
    yca, ycb = ya[ca], yb[cb]
    assert np.array_equal(yca, ycb), "label mismatch after hash alignment"
    res = S3.paired_boot(pa[:, ca], pb[:, cb], yca)
    res["p_two_sided"] = res.pop("p")
    res["p_one_sided"] = res["p_two_sided"] / 2.0
    return res


def main():
    print("Building GPT-4o-mini prediction matrix (cached) ...", flush=True)
    p4, y4, h4 = S3.rrf_pred_matrix("gpt_4o_mini", 0.15)
    out = {"metric": "per-repeat-pooled F0.5, founder-level bootstrap 10,000 resamples, seed 42; "
                     "Delta = F0.5(gpt_4o_mini) - F0.5(other)", "pairs": {}}
    for model in ("gpt_5_2", "claude_sonnet_4_5"):
        print(f"Building {model} prediction matrix ...", flush=True)
        pm, ym, hm = build_pred_matrix(model, 0.15)
        res = align_and_boot(p4, h4, y4, pm, hm, ym)
        out["pairs"][f"gpt_4o_mini_vs_{model}"] = res
        print(f"\nGPT-4o-mini vs {model}  n={res['n']}  "
              f"4o={res['rrf_f05']:.4f} {model}={res['base_f05']:.4f}  Delta={res['delta']:+.4f}  "
              f"CI[{res['ci95'][0]:+.4f},{res['ci95'][1]:+.4f}]  "
              f"p_two={res['p_two_sided']:.4f} p_one={res['p_one_sided']:.4f}", flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "generator_significance.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RESULTS / 'generator_significance.json'}", flush=True)


if __name__ == "__main__":
    main()
