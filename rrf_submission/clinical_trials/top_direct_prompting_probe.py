"""TOP contamination probe: direct LLM prompting on the 627 Phase-I test trials.

Why: the TOP test trials are real, named, pre-2022 trials whose outcomes are public and
in every LLM's pretraining corpus. RRF's win over HINT (PR-AUC 0.638 vs 0.567) is open to
a "the LLM just recalled the outcomes" attack, because there is NO direct-prompting
baseline on TOP (unlike the founder side). This probe bounds that: if direct prompting —
which has MAXIMAL opportunity to exploit memorised outcomes — does NOT beat RRF, then RRF's
margin is not explained by contamination.

We ask the same model RRF uses to answer questions (gpt-4o-mini) to predict Phase-I success
directly from the trial's diseases/drugs/eligibility text, zero-shot and few-shot, returning
a 0-100 success probability. We report PR-AUC / ROC-AUC / F1 vs RRF (0.638 / 0.596 / 0.655)
and the always-positive floor (F1 0.712, PR-AUC 0.553).

Needs OPENAI_API_KEY. ~627 calls/setting, gpt-4o-mini, a few $ total.

    export OPENAI_API_KEY=...
    python rrf_submission/clinical_trials/top_direct_prompting_probe.py --shots 0
    python rrf_submission/clinical_trials/top_direct_prompting_probe.py --shots 5
"""
from __future__ import annotations
import argparse, asyncio, json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score

# raw TOP Phase-I test trials with text (diseases/drugs/criteria/label); 627 trials, 55.3% positive.
# This is the public TOP benchmark test split (Fu et al., 2022); place that CSV here to run the probe.
DEFAULT_TEST = Path(__file__).resolve().parent / "data" / "phase_I_test_raw.csv"
OUT = Path(__file__).resolve().parent / "results" / "top_direct_prompting_probe.json"

SYS = ("You are an expert in clinical drug development. You will be shown a Phase I clinical "
       "trial (its diseases, drugs, and eligibility criteria). Estimate the probability that "
       "the trial SUCCEEDS at Phase I (i.e., advances). Rely only on the trial design shown; "
       "do NOT use outside knowledge of this specific trial's real-world outcome. "
       "Respond with ONLY an integer 0-100 (probability of success).")


def trial_text(row: pd.Series, max_crit: int = 1200) -> str:
    dis = str(row.get("diseases", ""))[:300]
    drg = str(row.get("drugs", ""))[:300]
    crit = str(row.get("criteria", ""))[:max_crit]
    return f"Diseases: {dis}\nDrugs: {drg}\nEligibility criteria (truncated): {crit}"


def build_fewshot(df: pd.DataFrame, k: int, seed: int = 0) -> str:
    if k <= 0:
        return ""
    rng = np.random.RandomState(seed)
    pos = df[df.label == 1].sample(min(k // 2 + k % 2, int((df.label == 1).sum())), random_state=seed)
    neg = df[df.label == 0].sample(min(k // 2, int((df.label == 0).sum())), random_state=seed)
    ex = pd.concat([pos, neg]).sample(frac=1, random_state=seed)
    out = ["Here are labelled examples:"]
    for _, r in ex.iterrows():
        out.append(trial_text(r, max_crit=400) + f"\nOutcome: {'SUCCESS' if r.label==1 else 'FAILURE'}\n")
    return "\n".join(out) + "\n---\n"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-csv", default=str(DEFAULT_TEST))
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--fewshot-source", default=None,
                    help="CSV to draw few-shot examples from (default: the val split, to avoid test leakage)")
    args = ap.parse_args()

    from openai import AsyncOpenAI
    client = AsyncOpenAI()

    df = pd.read_csv(args.test_csv)
    assert "label" in df.columns, f"no label col in {list(df.columns)}"
    y = df["label"].astype(int).values
    print(f"loaded {len(df)} trials, {y.mean():.3f} positive", flush=True)

    fewshot = ""
    if args.shots > 0:
        src = pd.read_csv(args.fewshot_source) if args.fewshot_source else df.sample(frac=1, random_state=99)
        # if drawing from test itself, hold those rows out would be ideal; prefer a val source
        fewshot = build_fewshot(src, args.shots)

    sem = asyncio.Semaphore(args.concurrency)
    probs: dict[int, float] = {}

    async def one(i: int):
        async with sem:
            prompt = fewshot + "Trial to score:\n" + trial_text(df.iloc[i])
            for attempt in range(4):
                try:
                    r = await client.chat.completions.create(
                        model=args.model, temperature=0.0, max_tokens=6,
                        messages=[{"role": "system", "content": SYS},
                                  {"role": "user", "content": prompt}])
                    txt = r.choices[0].message.content or ""
                    m = re.search(r"\d+", txt)
                    probs[i] = (min(100, max(0, int(m.group()))) / 100.0) if m else 0.5
                    return
                except Exception as e:
                    if attempt == 3:
                        print(f"  trial {i} failed: {e}", flush=True); probs[i] = 0.5; return
                    await asyncio.sleep(4 * (attempt + 1))

    t0 = time.time()
    await asyncio.gather(*[one(i) for i in range(len(df))])
    idx = sorted(probs)
    p = np.array([probs[i] for i in idx]); yy = y[idx]
    yp = (p >= 0.5).astype(int)
    res = {
        "model": args.model, "shots": args.shots, "n": int(len(yy)), "prevalence": float(yy.mean()),
        "pr_auc": float(average_precision_score(yy, p)),
        "roc_auc": float(roc_auc_score(yy, p)),
        "f1_at_0.5": float(f1_score(yy, yp, zero_division=0)),
        "precision_at_0.5": float(precision_score(yy, yp, zero_division=0)),
        "recall_at_0.5": float(recall_score(yy, yp, zero_division=0)),
        "pred_pos_rate": float(yp.mean()),
        "reference_RRF": {"pr_auc": 0.638, "roc_auc": 0.596, "f1": 0.655},
        "reference_always_positive": {"pr_auc": round(float(yy.mean()), 3), "f1": 0.712},
        "wall_seconds": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\n=> direct prompting ({args.shots}-shot): PR-AUC {res['pr_auc']:.3f} / ROC-AUC {res['roc_auc']:.3f}"
          f"  vs RRF 0.638 / 0.596. "
          + ("BOUNDS contamination (prompting <= RRF)." if res['pr_auc'] <= 0.638
             else "WARNING: prompting >= RRF — investigate."))


if __name__ == "__main__":
    asyncio.run(main())
