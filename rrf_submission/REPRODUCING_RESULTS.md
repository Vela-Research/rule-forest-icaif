# Reproducing Paper Results

This guide explains how to reproduce the figures and tables from the paper.

## Prerequisites

```bash
pip install -r requirements.txt
```

## Option A: Run the Notebook (Recommended)

Open and run all cells in `notebooks/paper_reproduction.ipynb`:

```bash
jupyter notebook notebooks/paper_reproduction.ipynb
```

This is the fastest way to reproduce the paper results. The notebook reads precomputed CSV artifacts plus the leakage-free harness outputs under `baselines/leakage_free_ablation/` (`results_full_leakfree.json`, `results_model_panel.json`), and generates all figures and tables. It does not make any API calls or require external access — it uses only standard Python libraries (pandas, numpy, matplotlib, tabulate).

## Option B: Offline Primary Model Verification — DEPRECATED (leaky)

> **Deprecated: do not use for paper numbers.** `scripts/reproduce_paper.py`
> (and `scripts/04_primary_model.py`, which it wraps) reproduce the **superseded,
> leaky** RRF operating point (per-fold-pooled ranking with a phantom `Source`
> column), which reports F0.5 ≈ 0.117. That number does **not** match the paper.
> The leakage-free, phantom-free RRF headline is **F0.5 = 0.102**, produced by the
> harness in `baselines/leakage_free_ablation/` (`full_leakfree.py` →
> `results_full_leakfree.json`) and consumed by the notebook in Option A. Leak
> correction history: **0.118 → 0.105 → 0.102**. The leaky
> `precomputed/*/ablation_results/nested_cv_*.csv` inputs this option relied on
> have been removed from this release, so it no longer runs.

## Option C: End-to-End with LLM API Calls

To re-run the full pipeline from scratch (requires an OpenAI API key and incurs API costs):

```bash
export OPENAI_API_KEY="sk-..."

# Step 1: Generate questions
python scripts/01_question_generation.py --provider openai --model gpt-4o-mini

# Step 2: Generate predictions
python scripts/02_test_questions.py --provider openai --model gpt-4o-mini --mode questions

# Step 3: Run primary model evaluation
python scripts/04_primary_model.py
```

**Note:** LLM outputs are stochastic, so exact numbers will differ slightly from the paper. The precomputed artifacts in Option A ensure exact reproduction. Step 3 (`scripts/04_primary_model.py`) computes the RRF operating point with the **superseded, leaky** protocol (F0.5 ≈ 0.117); the leakage-free headline (F0.5 = 0.102) is produced by `baselines/leakage_free_ablation/` and read by the Option A notebook.

## Why Precomputed Artifacts?

RRF uses LLMs for question generation and scoring. LLM outputs are **non-deterministic** and **API-dependent**: the same prompt can produce different responses across runs, and results depend on model versions that may change over time. The repository ships CSV snapshots of all LLM-generated outputs so that the paper results can be reproduced exactly without requiring API access.

## Artifact Map

The table below maps each section of the paper reproduction notebook to the precomputed artifacts it reads. Section names correspond to markdown headers in the notebook.

Because the per-fold-pooled `precomputed/*/ablation_results/nested_cv_*.csv` files leaked information across folds (and carried a phantom `Source` column), they have been **removed** from this release. Every RRF number the notebook shows is now read from the leakage-free, phantom-free harness in `baselines/leakage_free_ablation/`. Leak correction history for the RRF headline: **0.118 → 0.105 → 0.102**.

| Notebook Section | Artifact Path(s) | Paper Output |
|---|---|---|
| **Setup** | `baselines/leakage_free_ablation/results_full_leakfree.json`, `baselines/leakage_free_ablation/results_model_panel.json` (leakage-free sources) | — |
| **Optimisation Target Comparison** | — (alt-objective F1/F2/Precision sweep has no leakage-free re-run; cell prints a restricted-data note) | Table: optimisation target comparison |
| **Main Results: RRF vs Baselines** | RRF row: `baselines/leakage_free_ablation/results_full_leakfree.json` (panel a "LLM-Only", leakage-free F0.5 = 0.102; supersedes the leaky nested_cv 0.117). Non-RRF rows (clean, unchanged): `precomputed/gpt_4o_mini/benchmark_results/benchmark_gpt_4o_mini.csv`, `precomputed/gpt_4o_mini/vanilla/vanilla_*.csv` (6 files) | Table: main results |
| **Question-Generating Model Comparison** | `baselines/leakage_free_ablation/results_model_panel.json` (per-model leakage-free RRF F0.5 and Elastic-Net F0.5; regenerate with `python make_model_panel.py` from `baselines/leakage_free_ablation/`) | Table: cross-model comparison |
| **Combined Ablation Figure** | `baselines/leakage_free_ablation/results_full_leakfree.json` (panel a question source; panel b similarity metric; panel c Hamming threshold) + `results_model_panel.json` (panel e question-generating model). Panel d chronological split omitted — requires non-public founding dates, no leakage-free re-run shipped | Figure: 5-panel ablation figure |
| **Optimisation Target Ablation (Detailed)** | — (alt-objective sweep has no leakage-free re-run; cell prints a restricted-data note) | Table: detailed ablation metrics |
| **Hyperparameter Sensitivity** | — (grid-search + per-fold artifacts are pre-correction/leaky; no leakage-free source shipped, so the cell prints a restricted-data note and the figure is omitted) | Figure: 2-panel hyperparameter sensitivity |
| **Clinical Trials: TOP Phase I Results** | `clinical_trials/results/phase_I/primary_results/primary_model_repeat_metrics.csv`, `clinical_trials/results/phase_I/logreg_results/logreg_repeat_metrics.csv` | Table: clinical trial results |
| **Clinical Trials: Operating-Point Diagnostics** | Same as above | Table: operating-point diagnostics (appendix) |
| **Clinical Trials: Recall–Specificity Pareto** | `clinical_trials/results/phase_I/ablation_results/beta_sweep_summary.csv` | Figure: recall–specificity Pareto (appendix) |

## Regression Tests

To verify that precomputed artifacts have not been inadvertently modified:

```bash
python -m unittest tests.test_precomputed_regression -v
```

This checks file checksums against a golden manifest (`tests/regression_manifest.json`).
