"""Founder results: Table 2, Table 3, the expert arm (Section 5.2) and equal-length shortlists.

Step 1, predict (slow, about 7 CPU-hours, almost all of it the elastic net). Every model is
fitted under the same 10 x 10 nested cross-validation (outer seeds 42 + repeat - 1) and its
test-fold predictions are saved to results/predictions/<arm>.npz:
  vote_*  Rule Forest. The candidate pool is rebuilt inside every outer training fold.
  en_*    The Table 2 elastic net on all of a generator's questions.
Step 2, report (seconds). Pooled scores, paired founder bootstraps (10,000 resamples, seed 42)
and shortlist precision, written to results/table2.json, table3.json, expert.json and
shortlists.json. Table 2 also reads results/gptree_rrm.json, engineered_ml.json and
embeddings.json, which the scripts in baselines/ write.

  python experiments/founders.py predict [--jobs 16]
  python experiments/founders.py report
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one thread per worker; set before numpy loads
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rule_forest import data, elastic_net, metrics, vote

RESULTS = ROOT / "results"
PRED = RESULTS / "predictions"

# arm: (generator, expert questions, pool settings)
VOTES = {
    "vote_hamming_0.15": ("gpt_4o_mini", False, dict(similarity="hamming", threshold=0.15)),
    "vote_hamming_0.10": ("gpt_4o_mini", False, dict(similarity="hamming", threshold=0.10)),
    "vote_hamming_0.20": ("gpt_4o_mini", False, dict(similarity="hamming", threshold=0.20)),
    "vote_jaccard_0.90": ("gpt_4o_mini", False, dict(similarity="jaccard", threshold=0.90)),
    "vote_cosine_0.85": ("gpt_4o_mini", False, dict(similarity="cosine", threshold=0.85)),
    "vote_claude_sonnet_4_5": ("claude_sonnet_4_5", False, dict(similarity="hamming", threshold=0.15)),
    "vote_gpt_5_2": ("gpt_5_2", False, dict(similarity="hamming", threshold=0.15)),
    "vote_expert": ("gpt_4o_mini", True, dict(similarity="hamming", threshold=0.01, cap_expert=True)),
}
ENS = {
    "en_gpt_4o_mini": ("gpt_4o_mini", False),
    "en_claude_sonnet_4_5": ("claude_sonnet_4_5", False),
    "en_gpt_5_2": ("gpt_5_2", False),
    "en_expert": ("gpt_4o_mini", True),
}
SHORTLISTS = [40, 60, 80, 120, 160, 200, 300, 402, 600, 800]


# ------------------------------------------------------------------ step 1: predictions
def _save(arm, recs, a, **extra):
    M = metrics.prediction_matrix(recs, len(a.y))
    arrays = {"pred": M, "y": a.y, "ids": a.ids}
    for key in ("votes", "prob"):
        if key in recs[0]:
            S = np.zeros(M.shape)
            for r in recs:
                S[r["repeat"] - 1, r["test"]] = r[key]
            arrays[key] = S
    folds = [(r["repeat"], r["fold"], r["TP"], r["FP"], r["FN"]) for r in recs]
    arrays["folds"] = np.array(folds)
    for k, v in extra.items():
        arrays[k] = v
    PRED.mkdir(parents=True, exist_ok=True)
    metrics.save_npz(PRED / f"{arm}.npz", arrays)


def predict_votes():
    for arm, (gen, ex, kw) in VOTES.items():
        a = data.load_founders(gen, expert=ex)
        t0 = time.time()
        recs = vote.nested_cv(a.X, a.y, source=a.source, **kw)
        _save(arm, recs, a, N=np.array([r["N"] for r in recs]), T=np.array([r["T"] for r in recs]),
              pool_size=np.array([len(r["pool"]) for r in recs]),
              selected=np.array([";".join(map(str, r["selected"])) for r in recs]),
              questions=np.array(a.questions))
        print(f"{arm:24s} pooled F0.5 {metrics.pooled(recs)['f05']:.4f} [{time.time() - t0:.0f}s]", flush=True)
    a = data.load_founders()
    Xr = data.load_reasked_blanks(a)
    recs = vote.nested_cv(Xr, a.y, similarity="hamming", threshold=0.15)
    _save("vote_hamming_0.15_reasked_blanks", recs, a)
    print(f"vote with re-asked blanks pooled F0.5 {metrics.pooled(recs)['f05']:.4f}", flush=True)


_SETS = {}


def _en_fold(task):
    arm, rep, k = task
    if arm not in _SETS:
        gen, ex = ENS[arm]
        _SETS[arm] = data.load_founders(gen, expert=ex)
    a = _SETS[arm]
    fold = [f for f in vote.outer_folds(a.y) if f[:2] == (rep, k)]
    r = elastic_net.nested_cv(a.X, a.y, "table2", folds=fold)[0]
    return arm, r


def predict_elastic_nets(jobs):
    tasks = [(arm, rep, k) for arm in ENS for rep in range(1, 11) for k in range(1, 11)]
    out = {arm: [] for arm in ENS}
    t0 = time.time()
    with Pool(jobs) as pool:
        for i, (arm, r) in enumerate(pool.imap_unordered(_en_fold, tasks), 1):
            out[arm].append(r)
            if i % 20 == 0:
                print(f"elastic net {i}/{len(tasks)} folds [{time.time() - t0:.0f}s]", flush=True)
    for arm, recs in out.items():
        recs.sort(key=lambda r: (r["repeat"], r["fold"]))
        gen, ex = ENS[arm]
        a = data.load_founders(gen, expert=ex)
        _save(arm, recs, a, coef=np.array([r["coef"] for r in recs]),
              threshold=np.array([r["threshold"] for r in recs]))
        print(f"{arm:24s} pooled F0.5 {metrics.pooled(recs)['f05']:.4f}", flush=True)


# ------------------------------------------------------------------ step 2: report
def load(arm):
    z = np.load(PRED / f"{arm}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def scores(arm):
    z = load(arm)
    recs = [{"repeat": int(r), "TP": int(tp), "FP": int(fp), "FN": int(fn)} for r, _, tp, fp, fn in z["folds"]]
    return metrics.pooled(recs)


def prompting(name, ids, y):
    p = data.load_prompting(name, ids)
    P, R, F = metrics.prf(int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum()),
                          int(((p == 0) & (y == 1)).sum()))
    return p, {"precision": P, "recall": R, "f05": F, "flagged": int(p.sum())}


def boot(A, B, y, metrics_=("f05", "precision", "recall")):
    return {m: metrics.paired_bootstrap(A, B, y, m) for m in metrics_}


def aligned(arm_a, arm_b):
    """Prediction matrices of two arms restricted to their common founders."""
    a, b = load(arm_a), load(arm_b)
    ia, ib = data.common_founders(list(a["ids"]), list(b["ids"]))
    assert np.array_equal(a["y"][ia], b["y"][ib])
    return a["pred"][:, ia], b["pred"][:, ib], a["y"][ia]


def table2(gpt):
    y, ids = gpt.y, gpt.ids
    rf = load("vote_hamming_0.15")
    assert np.array_equal(rf["y"], y) and list(rf["ids"]) == list(ids)
    o3, o3_row = prompting("few_shot_o3", ids, y)
    rows = {"rule_forest": scores("vote_hamming_0.15"), "elastic_net": scores("en_gpt_4o_mini")}
    base = {}
    for name in ("engineered_ml", "embeddings", "gptree_rrm"):
        p = RESULTS / f"{name}.json"
        if not p.exists():
            raise FileNotFoundError(f"{p} is missing; run the script in baselines/ that writes it (see README.md)")
        base[name] = json.loads(p.read_text())
    rows["engineered_ml"] = base["engineered_ml"]["pooled"]
    rows["embeddings"] = base["embeddings"]["pooled"]
    rows["o3_few_shot"] = o3_row
    rows["gpt_4o_mini_zero_shot"] = prompting("zero_shot_gpt_4o_mini", ids, y)[1]
    rows["gptree"] = base["gptree_rrm"]["GPTree"]["pooled"]
    rows["rrm"] = base["gptree_rrm"]["RRM"]["pooled"]
    en = load("en_gpt_4o_mini")
    flagged = rf["pred"].sum(1)
    hits = (rf["pred"] * (y == 1)).sum(1)
    N, T = rf["N"], rf["T"]
    sel = [s.split(";") for s in rf["selected"]]
    counts = {}
    for s in sel:
        for q in s:
            counts[int(q)] = counts.get(int(q), 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
    detail = {
        "founders": int(len(y)), "successes": int(y.sum()), "base_rate": float(y.mean()),
        "flagged_per_repeat_mean": float(flagged.mean()), "hits_per_repeat_mean": float(hits.mean()),
        "profiles_screened_at_random_for_the_same_hits": float(hits.mean() / y.mean()),
        "N_median": float(np.median(N)), "T_median": float(np.median(T)),
        "T_over_N_median": float(np.median(T / N)),
        "T_over_N_iqr": [float(np.percentile(T / N, 25)), float(np.percentile(T / N, 75))],
        "pool_size_range": [int(rf["pool_size"].min()), int(rf["pool_size"].max())],
        "pool_size_median": float(np.median(rf["pool_size"])),
        "folds_with_N_equal_to_pool_size": int((N == rf["pool_size"]).sum()),
        "elastic_net_nonzero_weights_median": float(np.median((en["coef"] != 0).sum(1))),
        "elastic_net_flagged_per_repeat_mean": float(en["pred"].sum(1).mean()),
        "questions_selected_most_often": [{"question": str(rf["questions"][q]), "folds": c} for q, c in top],
        "with_reasked_blanks": scores("vote_hamming_0.15_reasked_blanks"),
    }
    prompts = {name: prompting(name, ids, y)[1] for name in data.PROMPTING}
    intervals = {"rule_forest_minus_elastic_net": boot(rf["pred"], en["pred"], y),
                 "rule_forest_minus_o3_few_shot": boot(rf["pred"], o3, y),
                 "elastic_net_minus_o3_few_shot": boot(en["pred"], o3, y, ("f05",))}
    return {"rows": rows, "rule_forest_detail": detail, "intervals": intervals, "all_prompting_runs": prompts,
            "note": "Pooled scores, mean and SD over 10 repeats (30 for GPTree and RRM, 3 runs x 10). Rule Forest, "
                    "the elastic net and prompting use the 8,173 founders of the answer matrix; Engineered ML, "
                    "Embeddings, GPTree and RRM use all 8,204 evaluation profiles, so their folds differ. Prompting "
                    "rows are scored once, and a founder a run did not score counts as not flagged. Intervals are "
                    "paired founder bootstraps, 10,000 resamples, seed 42."}


def table3():
    cells = {arm: {**scores(arm), "pool_size_range": [int(load(arm)["pool_size"].min()), int(load(arm)["pool_size"].max())],
                   "N_median": float(np.median(load(arm)["N"]))}
             for arm in VOTES if arm != "vote_expert"}
    ens = {arm: scores(arm) for arm in ("en_gpt_4o_mini", "en_claude_sonnet_4_5", "en_gpt_5_2")}
    gens = {}
    for other in ("vote_gpt_5_2", "vote_claude_sonnet_4_5"):
        A, B, y = aligned("vote_hamming_0.15", other)
        gens[f"gpt_4o_mini_minus_{other[5:]}"] = {**metrics.paired_bootstrap(A, B, y, "f05")}
    vote_vs_en = {}
    for g in ("gpt_4o_mini", "claude_sonnet_4_5", "gpt_5_2"):
        v = "vote_hamming_0.15" if g == "gpt_4o_mini" else f"vote_{g}"
        A, B, y = aligned(v, f"en_{g}")
        vote_vs_en[g] = metrics.paired_bootstrap(A, B, y, "f05")
    return {"vote": cells, "elastic_net": ens, "generator_intervals": gens, "vote_minus_elastic_net": vote_vs_en,
            "note": "Similarity settings change only the redundancy step of the in-fold candidate pool. "
                    "Generator intervals use the founders common to both matrices."}


def expert(gpt):
    ex, rf = load("vote_expert"), load("vote_hamming_0.15")
    en58, en50 = load("en_expert"), load("en_gpt_4o_mini")
    y = gpt.y
    return {"expert_arm": {**scores("vote_expert"), "N_median": float(np.median(ex["N"])),
                           "pool_size_range": [int(ex["pool_size"].min()), int(ex["pool_size"].max())]},
            "rule_forest": scores("vote_hamming_0.15"),
            "elastic_net_58_questions": {**scores("en_expert"),
                                         "nonzero_weights_median": float(np.median((en58["coef"] != 0).sum(1)))},
            "intervals": {"expert_minus_rule_forest": boot(ex["pred"], rf["pred"], y),
                          "expert_minus_elastic_net_58": boot(ex["pred"], en58["pred"], y),
                          "elastic_net_58_minus_elastic_net_50": boot(en58["pred"], en50["pred"], y)},
            "note": "The expert arm adds the 8 expert questions to the 50 GPT-4o-mini questions, drops the last 8 "
                    "LLM questions that pass the floor (so the count before redundancy removal is unchanged), and uses Hamming 0.01, because at 0.15 the rarely-YES expert questions count as "
                    "near-duplicates of the strongest LLM questions. It therefore differs from Rule Forest in both "
                    "the questions and the redundancy threshold, and its pool is about twice as large."}


def fold_quotas(sizes, K, n):
    """Places per test fold in proportion to fold size, summing to exactly K (largest remainder)."""
    quota = np.array(sizes) * K / n
    base = np.floor(quota).astype(int)
    extra = np.argsort(-(quota - base), kind="stable")[: K - base.sum()]
    base[extra] += 1
    return base


def inclusion(scores_, folds, K, n):
    """Fractional shortlist membership (n,) for one repeat. Each test fold contributes its quota
    of K founders (fold_quotas), the highest-scoring ones; founders tied at the cut share the
    remaining places equally (the expected value of a random tie-break)."""
    inc = np.zeros(n)
    for te, left in zip(folds, fold_quotas([len(f) for f in folds], K, n)):
        s = scores_[te]
        for v in np.sort(np.unique(s))[::-1]:
            if left <= 0:
                break
            at = np.where(s == v)[0]
            take = min(left, len(at))
            inc[te[at]] = take / len(at)
            left -= take
    return inc


def shortlist_pr(inc, y, w=None):
    w = np.ones(len(y)) if w is None else w
    inc = np.atleast_2d(inc)
    tp = (inc * (y == 1)) @ w
    return float(np.mean(tp / (inc @ w))), float(np.mean(tp / ((y == 1) * w).sum()))


def shortlist_boot(A, B, y, n_boot=10_000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y)
    d = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.randint(0, n, n), minlength=n).astype(float)
        d[i] = shortlist_pr(A, y, w)[0] - shortlist_pr(B, y, w)[0]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"ci95": [float(lo), float(hi)], "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def shortlists(gpt):
    y, n = gpt.y, len(gpt.y)
    folds = {}
    for rep, _, _, te in vote.outer_folds(y):
        folds.setdefault(rep, []).append(te)
    inc = {}
    for arm, key in (("vote_hamming_0.15", "votes"), ("vote_expert", "votes"),
                     ("en_gpt_4o_mini", "prob"), ("en_expert", "prob")):
        S = load(arm)[key]
        inc[arm] = {K: np.array([inclusion(S[r], folds[r + 1], K, n) for r in range(10)]) for K in SHORTLISTS}
    o3 = data.load_prompting("few_shot_o3", gpt.ids).astype(float)
    n_yes = int(o3.sum())
    inc["o3_few_shot"] = {}
    for K in SHORTLISTS:     # o3 has no order inside its YES set: a random K of them, or all plus random NOs
        v = np.where(o3 == 1, min(K, n_yes) / n_yes, 0.0)
        if K > n_yes:
            v = np.where(o3 == 0, (K - n_yes) / (n - n_yes), v)
        inc["o3_few_shot"][K] = v[None, :]
    table = [{"K": K, **{arm: dict(zip(("precision", "recall"), shortlist_pr(inc[arm][K], y))) for arm in inc}}
             for K in SHORTLISTS]
    pairs = [("vote_hamming_0.15", "o3_few_shot"), ("vote_expert", "o3_few_shot"), ("en_gpt_4o_mini", "o3_few_shot"),
             ("vote_hamming_0.15", "en_gpt_4o_mini"), ("vote_expert", "en_gpt_4o_mini")]
    ints = []
    for a, b in pairs:
        for K in (60, 120, 402):
            pa, pb = shortlist_pr(inc[a][K], y)[0], shortlist_pr(inc[b][K], y)[0]
            ints.append({"a": a, "b": b, "K": K, "precision_a": pa, "precision_b": pb, "delta": pa - pb,
                         **shortlist_boot(inc[a][K], inc[b][K], y)})
    return {"o3_flagged": n_yes, "table": table, "intervals": ints,
            "note": "K founders per repeat (of 8,173), taken fold by fold in proportion to fold size. "
                    "Precision and recall are pooled within each repeat and averaged over repeats. The intervals "
                    "are 15 comparisons with no correction for multiplicity."}


def write(name, obj):
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=1) + "\n")


def report():
    gpt = data.load_founders()
    write("table2", table2(gpt))
    write("table3", table3())
    write("expert", expert(gpt))
    write("shortlists", shortlists(gpt))
    t2 = json.loads((RESULTS / "table2.json").read_text())
    for k, r in t2["rows"].items():
        print(f"{k:24s} P {r['precision']:.3f} R {r['recall']:.3f} F0.5 {r['f05']:.3f}")
    i = t2["intervals"]["rule_forest_minus_elastic_net"]["f05"]
    print(f"Rule Forest - elastic net F0.5 {i['delta']:+.3f} [{i['ci95'][0]:+.3f}, {i['ci95'][1]:+.3f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["predict", "report"])
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--votes-only", action="store_true", help="skip the slow elastic-net predictions")
    a = ap.parse_args()
    if a.step == "predict":
        predict_votes()
        if not a.votes_only:
            predict_elastic_nets(a.jobs)
    else:
        report()


if __name__ == "__main__":
    main()
