"""Producer for results_model_panel.json (Table 2 question-generating-model panel).

For each question-generating model we report, on the same binary question matrix:
  - rrf_f05: the leakage-free RRF vote F0.5 (hamming-0.15 dedup, questions ranked by
    TRAINING-fold F0.5 inside each fold, n_q/threshold tuned on train; 10x10, per-repeat
    pooled) -- identical to run_ablation.py's leakfree_f05.
  - en_f05: the elastic-net logistic-regression baseline F0.5, averaged over the shipped
    10x10 cross-validation folds in precomputed/<model>/benchmark_results/.
Writes results_model_panel.json, which the reproduction notebook reads for the model panel.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

from run_ablation import RRF, build_matrix, run_nested, agg_f05

MODELS = {
    "gpt_4o_mini": "GPT-4o-mini",
    "claude_sonnet_4_5": "Claude Sonnet 4.5",
    "gpt_5_2": "GPT-5.2",
}


def en_f05(model: str) -> float:
    p = RRF / "precomputed" / model / "benchmark_results" / f"benchmark_{model}.csv"
    return float(pd.read_csv(p)["Test F0.5"].mean())


def main():
    out = {}
    for model, label in MODELS.items():
        X, y = build_matrix(model, "hamming", 0.15)
        rrf = agg_f05(run_nested(X, y, "perfold"))
        out[model] = {
            "label": label,
            "Q": int(X.shape[0]),
            "rrf_f05": rrf,
            "en_f05": en_f05(model),
            "n_founders": int(X.shape[1]),
            "phantom_free": True,
        }
        print(f"{model:18s} Q={X.shape[0]:2d} rrf_f05={rrf:.4f} en_f05={out[model]['en_f05']:.4f}", flush=True)
    Path("results_model_panel.json").write_text(json.dumps(out, indent=2))
    print("Wrote results_model_panel.json", flush=True)


if __name__ == "__main__":
    main()
