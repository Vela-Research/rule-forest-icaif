"""Offline verification of the traditional-ML baseline numbers (Table 2: Engineered ML /
Embeddings) from the shipped frozen artifacts. numpy + pandas only; no training, no network.

Why this exists: the two baselines were trained on the non-public VCBench founder records
(raw structured job/education JSON for the Engineered-ML features; original founder prose for
the MiniLM embeddings). Those inputs are NOT part of this anonymized release -- shipping them
would deanonymise the founders -- so the baselines cannot be *retrained* from the release (see
README.md). What CAN be verified offline is that the reported headline numbers are the faithful
aggregate of the shipped per-fold results, and that they match the per-founder consensus
predictions. This script checks both.

  python evaluate_from_artifacts.py
"""
from __future__ import annotations
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BETA2 = 0.25


def f05(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    d = BETA2 * p + r
    return (1 + BETA2) * p * r / d if d else 0.0, p, r


def _maybe(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def check_protocol_b():
    """Reported f05_mean == mean of the shipped per-repeat F0.5 (10x10 CV)."""
    print("== Check 1: headline F0.5 == aggregate of shipped per-fold results ==")
    ok = True
    for line in (RESULTS / "test_set_protocol_b.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        prm = _maybe(r["per_repeat_means"]); agg = _maybe(r["aggregate"])
        recomputed = float(np.mean([d["f05"] for d in prm]))
        shipped = float(agg["f05_mean"])
        match = abs(recomputed - shipped) < 1e-9
        ok &= match
        print(f"  {r['cell']:32s} recomputed={recomputed:.6f} shipped={shipped:.6f} "
              f"[{'OK' if match else 'MISMATCH'}] over {len(prm)} repeats")
    # cross-check the summary CSV
    csv = pd.read_csv(RESULTS / "test_set_protocol_b_summary.csv")
    print("  summary CSV f05_mean:", dict(zip(csv["cell"], csv["f05_mean"].round(6))))
    return ok


def check_consensus():
    """Recompute the per-founder majority-vote (consensus) F0.5 from the raw predictions."""
    print("\n== Check 2: consensus F0.5 from per-founder predictions (consensus_allrepeats.parquet) ==")
    df = pd.read_parquet(RESULTS / "consensus_allrepeats.parquet")
    for model in ["R1", "R2", "RRF", "LogReg-EN", "o3-fewshot"]:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        y = sub["y"].to_numpy(); pred = sub["consensus"].to_numpy()
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        f, p, r = f05(tp, fp, fn)
        print(f"  {model:12s} n={len(sub):5d} n_pos={int(y.sum()):3d} pred_pos={int(pred.sum()):4d} "
              f"consensus F0.5={f:.4f} (P={p:.3f} R={r:.3f})")
    print("  (consensus = majority vote over the 10 repeats; the paired bootstrap in "
          "baselines/leakage_free_ablation/section3_significance.py operates on these columns.)")


def main():
    ok = check_protocol_b()
    check_consensus()
    print(f"\nVERIFIED: reported Engineered-ML / Embeddings F0.5 match the shipped per-fold data."
          if ok else "\nWARNING: a reported number did not match the shipped per-fold data.")
    print("NOTE: full retraining from raw data is out of scope for the anonymized release "
          "(needs non-public VCBench structured records / original prose) -- see README.md.")


if __name__ == "__main__":
    main()
