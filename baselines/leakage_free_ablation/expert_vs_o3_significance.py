"""Paired founder-level bootstrap: Rule Forest + Expert vs o3 few-shot (strongest baseline overall).

Same methodology as section3_significance.py (per-repeat-pooled metric, 10,000 resamples, seed 42,
hash-aligned). The expert arm is the full_leakfree.py LLM+Expert panel (mode 'llm_expert', hamming
0.01; F0.5 0.125, precision 0.164); o3 few-shot is the single-shot prediction per founder from
consensus_allrepeats.parquet (no CV, so nothing to pool). Delta = (LLM+Expert) - (o3 few-shot).
Reports F0.5 AND precision.

  python expert_vs_o3_significance.py   # -> results/expert_vs_o3_significance.json
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import section3_significance as S3
from expert_significance import matrix_with_hashes, build_pred_matrix, per_repeat_pooled_prf

RESULTS = Path(__file__).resolve().parent / "results"
N_BOOT, SEED = 10000, 42


def prf_1d(vec, y):
    """Single-shot precision / F0.5 for o3 (one prediction per founder)."""
    p = np.asarray(vec, int)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    return prec, S3._f05(tp, fp, fn)


def paired_boot(expert, o3, y, n=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    N = len(y)
    pe0, fe0 = per_repeat_pooled_prf(expert, y)   # expert: (10, N) per-repeat-pooled
    po0, fo0 = prf_1d(o3, y)                       # o3: (N,) single shot
    dP = np.empty(n); dF = np.empty(n)
    for b in range(n):
        ix = rng.randint(0, N, N); yb = y[ix]
        pe, fe = per_repeat_pooled_prf(expert[:, ix], yb)
        po, fo = prf_1d(o3[ix], yb)
        dP[b] = pe - po; dF[b] = fe - fo

    def summarise(dd, obs, e0, b0):
        lo, hi = np.percentile(dd, [2.5, 97.5])
        p = min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean()))
        return {"expert": e0, "o3": b0, "delta": float(obs),
                "ci95": [float(lo), float(hi)], "p_two_sided": float(p)}
    return {"f05": summarise(dF, fe0 - fo0, fe0, fo0),
            "precision": summarise(dP, pe0 - po0, pe0, po0), "n": int(N)}


def main():
    print("Building LLM+Expert prediction matrix (mode llm_expert, hamming 0.01) ...", flush=True)
    Xe, ye, he = matrix_with_hashes("gpt_4o_mini", "llm_expert", "hamming", 0.01)
    expert = build_pred_matrix(Xe, ye)
    e_series = pd.Series(np.arange(len(he)), index=he)

    print("Loading o3 few-shot per-founder predictions (consensus_allrepeats.parquet) ...", flush=True)
    o3 = pd.read_parquet(S3.CONS)
    o3 = o3[o3.model == "o3-fewshot"][["hash", "y", "consensus"]]
    amb = S3._drop_amb(he, o3["hash"].values)
    o3 = o3[~o3["hash"].isin(amb)].drop_duplicates("hash")
    keep = [h for h in o3["hash"] if h in e_series.index]
    o3 = o3.set_index("hash").loc[keep]
    ce = e_series.loc[keep].to_numpy()
    y = ye[ce]
    assert np.array_equal(y, o3["y"].to_numpy().astype(int)), "label mismatch after hash alignment"

    res = paired_boot(expert[:, ce], o3["consensus"].to_numpy().astype(int), y)
    res["metric"] = ("per-repeat-pooled F0.5 / precision (expert) vs single-shot (o3), founder-level "
                     f"bootstrap {N_BOOT} resamples, seed {SEED}; Delta = (LLM+Expert) - (o3 few-shot)")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "expert_vs_o3_significance.json").write_text(json.dumps(res, indent=2))
    for k in ("f05", "precision"):
        r = res[k]
        print(f"\n{k:9s}  LLM+Expert={r['expert']:.4f}  o3={r['o3']:.4f}  Delta={r['delta']:+.4f}  "
              f"CI[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]  p_two={r['p_two_sided']:.4f}", flush=True)
    print(f"\nn={res['n']} founders. Wrote {RESULTS / 'expert_vs_o3_significance.json'}", flush=True)


if __name__ == "__main__":
    main()
