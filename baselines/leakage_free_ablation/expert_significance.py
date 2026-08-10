"""Paired founder-level bootstrap for the EXPERT-IN-THE-LOOP gain (paper 5.2):
Rule Forest (LLM-Only) vs Rule Forest + Expert.

Same methodology as section3_significance.py (paper 5.1): per-repeat-pooled metric, 10,000
founder-level bootstrap resamples, seed 42, aligned by founder hash (ambiguous 5-char prefixes
dropped). Both arms are the leakage-free full_leakfree.py panel-a operating points:
  LLM-Only    : mode 'llm',        hamming 0.15  (Q=22)  -> F0.5 0.102, P 0.109
  LLM+Expert  : mode 'llm_expert', hamming 0.01  (Q=48)  -> F0.5 0.125, P 0.164
Delta = (LLM+Expert) - (LLM-Only). Reports F0.5 AND precision (the section's headline).

  python expert_significance.py   # -> results/expert_significance.json
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, str(Path(__file__).resolve().parent))
import section3_significance as S3
from run_ablation import RRF, META, q_f05
from src.questions.question_filtering import construct_predictions_file_list, filter_questions
from src.utils.scoring_utils import select_best_hyperparams, compute_weights, evaluate_test_fold

RESULTS = Path(__file__).resolve().parent / "results"
N_BOOT, SEED = 10000, 42


def matrix_with_hashes(model, mode, metric, thr):
    """full_leakfree.build_matrix, but also return the founder-column hashes."""
    pred_dir = RRF / "precomputed" / model / "test_predictions"
    args = argparse.Namespace(mode=mode, similarity_metric=metric,
                              similarity_threshold=thr, optimise_for="f0.5", sort_by="f0.5")
    try:
        files = construct_predictions_file_list(args, "_anonymised", predictions_dir=pred_dir)
    except FileNotFoundError:
        avail = sorted(p.name for p in pred_dir.glob("predictions_test_set_*anonymised.csv")
                       if "EXPERT" not in p.name)
        files = avail[:1]
    df = filter_questions(pred_dir, files, "_anonymised", args, sort_by="f0.5")
    qr = df[~df["Question"].isin(["Founder Index", "Dataset", "Success"])].copy()
    fcols = [c for c in qr.columns if c not in META]
    qr[fcols] = qr[fcols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    X = qr[fcols].values
    y = pd.to_numeric(df[df["Question"] == "Success"][fcols].iloc[0], errors="coerce").fillna(0).astype(int).values
    return X, y, np.array(fcols)


def build_pred_matrix(X, y):
    """Leakage-free 10x10 per-founder prediction matrix (per-training-fold F0.5 ranking)."""
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
    return pred


def per_repeat_pooled_prf(mat, y):
    """Table-2 metric: pool TP/FP/FN within each repeat -> (precision, F0.5) per repeat -> mean."""
    P, Fv = [], []
    for r in range(mat.shape[0]):
        p = mat[r]
        tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
        P.append(tp / (tp + fp) if tp + fp else 0.0)
        Fv.append(S3._f05(tp, fp, fn))
    return float(np.mean(P)), float(np.mean(Fv))


def paired_boot(expert, base, y, n=N_BOOT, seed=SEED):
    """Paired founder bootstrap of Delta=(expert-base) for both precision and F0.5."""
    rng = np.random.RandomState(seed)
    N = len(y)
    p_e0, f_e0 = per_repeat_pooled_prf(expert, y)
    p_b0, f_b0 = per_repeat_pooled_prf(base, y)
    dP = np.empty(n); dF = np.empty(n)
    for b in range(n):
        ix = rng.randint(0, N, N); yb = y[ix]
        pe, fe = per_repeat_pooled_prf(expert[:, ix], yb)
        pb, fb = per_repeat_pooled_prf(base[:, ix], yb)
        dP[b] = pe - pb; dF[b] = fe - fb

    def summarise(dd, obs, e0, b0):
        lo, hi = np.percentile(dd, [2.5, 97.5])
        p = min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean()))
        return {"expert": e0, "base": b0, "delta": float(obs),
                "ci95": [float(lo), float(hi)], "p_two_sided": float(p)}
    return {"f05": summarise(dF, f_e0 - f_b0, f_e0, f_b0),
            "precision": summarise(dP, p_e0 - p_b0, p_e0, p_b0), "n": int(N)}


def main():
    print("Building LLM-Only prediction matrix (cached: rrf_lf_predmat.npz) ...", flush=True)
    base, yb, hb = S3.rrf_pred_matrix("gpt_4o_mini", 0.15)          # mode 'llm', hamming 0.15
    print("Building LLM+Expert prediction matrix (mode llm_expert, hamming 0.01) ...", flush=True)
    Xe, ye, he = matrix_with_hashes("gpt_4o_mini", "llm_expert", "hamming", 0.01)
    expert = build_pred_matrix(Xe, ye)

    sb = pd.Series(np.arange(len(hb)), index=hb)
    se = pd.Series(np.arange(len(he)), index=he)
    amb = S3._drop_amb(hb, he)
    hbset = set(hb)
    common = list(dict.fromkeys([h for h in he if h not in amb and h in hbset]))
    cb = sb.loc[common].to_numpy(); ce = se.loc[common].to_numpy()
    yc = ye[ce]
    assert np.array_equal(yc, yb[cb]), "label mismatch after hash alignment"

    res = paired_boot(expert[:, ce], base[:, cb], yc)
    res["metric"] = ("per-repeat-pooled F0.5 / precision, founder-level bootstrap "
                     f"{N_BOOT} resamples, seed {SEED}; Delta = (LLM+Expert) - (LLM-Only)")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "expert_significance.json").write_text(json.dumps(res, indent=2))

    for k in ("f05", "precision"):
        r = res[k]
        print(f"\n{k:9s}  LLM+Expert={r['expert']:.4f}  LLM-Only={r['base']:.4f}  "
              f"Delta={r['delta']:+.4f}  CI[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]  "
              f"p_two={r['p_two_sided']:.4f}", flush=True)
    print(f"\nn={res['n']} founders. Wrote {RESULTS / 'expert_significance.json'}", flush=True)


if __name__ == "__main__":
    main()
