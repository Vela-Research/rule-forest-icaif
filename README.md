# Rule Forest — Reproduction Package

Anonymized code and data for *"Rule Forest: Interpretable Ensembles of LLM-Generated
Questions for Rare-Outcome Prediction."* Everything needed to reproduce the paper's
tables and figures **offline** (no API key, no network) from precomputed LLM outputs is
included.

## Layout
```
LICENSE  requirements.txt
rrf_submission/
  PROMPTS.md                 # exact question-generation prompt templates (verbatim)
  src/                       # scoring + question-filtering modules + llms/prompts.py
  precomputed/               # anonymized LLM question-response matrices + baseline outputs
  data/                      # anonymized founder dataset (9,192 profiles)
  notebooks/paper_reproduction.ipynb
baselines/
  leakage_free_ablation/     # canonical leakage-free RRF harness (reproduces F0.5 = 0.102)
  traditional_ml_port/       # Engineered-ML and Embeddings baseline outputs
baselines_llm/
  gptree/  rrm/              # GPTree / RRM per-seed artifacts (single-split runs)
  rescore/                   # 10x10 re-scoring of GPTree/RRM -> paper's Table 2 rows (offline)
count_effect/                # Figures 4 & 5 (founder count-effect + Kiva), self-contained
```

## Quickstart
Requires **Python >= 3.11**.
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # offline core needs only: numpy pandas scipy scikit-learn matplotlib pyarrow
```

## Reproduce every headline number (offline, no API key)

**Founder table (Rule Forest + ablations + expert-in-the-loop):**
```bash
cd baselines/leakage_free_ablation
python full_leakfree.py        # -> results_full_leakfree.json  (Rule Forest F0.5=0.102; ablations; expert 0.125)
python make_model_panel.py     # -> results_model_panel.json    (per-generator vote/elastic-net panel)
python section3_significance.py
```
**Engineered ML / Embeddings baselines:** `baselines/traditional_ml_port/results/test_set_protocol_b_summary.csv`.

**GPTree / RRM baselines (Table 2), by re-scoring the frozen trees/rules under 10x10 CV:**
```bash
cd baselines_llm/rescore && python rescore_gptree_rrm.py   # GPTree ~0.066, RRM ~0.076 (see rescore/README.md)
```
**Figures 4 & 5:**
```bash
cd count_effect && python make_figure4.py && python make_figure5.py   # -> figures/*.pdf
```

## Paper -> artifact map
| Paper item | Value | Where |
|---|---|---|
| Table 2 — Rule Forest (LLM + vote) | 0.109 / 0.082 / **0.102** | `baselines/leakage_free_ablation/results_full_leakfree.json` (panel a, LLM-Only) |
| Table 2 — Engineered ML / Embeddings | 0.086 / 0.069 | `baselines/traditional_ml_port/results/test_set_protocol_b_summary.csv` |
| Table 2 — LogReg (elastic net) | 0.080 | `rrf_submission/precomputed/gpt_4o_mini/benchmark_results/benchmark_gpt_4o_mini.csv` |
| Table 2 — prompting (o3 few-shot / GPT-4o-mini zero-shot) | 0.088 / 0.086 | `rrf_submission/precomputed/gpt_4o_mini/vanilla/` |
| Table 2 — GPTree / RRM | 0.066 / 0.076 | `baselines_llm/rescore/` (run `rescore_gptree_rrm.py`) |
| Table 3 — similarity / threshold ablations | 0.102 / 0.106 / 0.097 ; 0.090 / 0.102 / 0.081 | `results_full_leakfree.json` (panels b, c) |
| Table 3 — generator (vote): GPT-4o-mini / Claude / GPT-5.2 | 0.102 / 0.083 / 0.064 | `results_model_panel.json` |
| Table 3 — generator (elastic net): GPT-4o-mini / Claude / GPT-5.2 | 0.080 / 0.061 / 0.041 | `results_model_panel.json` |
| Expert-in-the-loop | precision 0.109->0.164 ; F0.5 0.102->0.125 | `results_full_leakfree.json` (panel a) |
| Figure 4 — founder count-effect (rho=-0.48) | founders 1.9% / 5% + reliability | `count_effect/make_figure4.py` |
| Figure 5 — Kiva generalisation | vote vs elastic net vs base rate | `count_effect/make_figure5.py` |

## Notes on reproducibility
- **Leakage-free harness.** The canonical Rule Forest numbers come from
  `baselines/leakage_free_ablation/`, which ranks candidate questions **within each training
  fold** (documented correction history 0.118 -> 0.105 -> 0.102).
- **Offline core vs. API pipeline.** All paper numbers reproduce offline from the shipped
  precomputed LLM outputs. Regenerating those outputs from scratch (question generation +
  scoring) requires an LLM API key; the offline reproduction does not.
- **Prompt templates.** The exact question-generation prompts are in
  `rrf_submission/PROMPTS.md` (readable) and `rrf_submission/src/llms/prompts.py` (literal
  source). The anonymised two-stage template produced the shipped `precomputed/` question
  matrices.
- **GPTree / RRM.** `baselines_llm/{gptree,rrm}/*_results.json` are the single-held-out-split
  runs; the paper's Table 2 numbers come from the 10x10 re-scoring in `baselines_llm/rescore/`
  (reuses the frozen tree / rule set, no API calls).
- **Restricted-data items.** The chronological-split robustness check needs per-founder founding
  dates that are not part of this anonymized release.

## Data
- **Founders** (`rrf_submission/data/founders_summary_data_anonymised.csv`): 9,192 U.S. companies
  founded 2010-2016 as anonymized natural-language summaries. The held-out evaluation set is 8,204
  founders with 158 positives (1.93% base rate).
- **Kiva microloans** (`count_effect/data/kiva_rq_scores_run1.jsonl`): 5,000 loans (222 positive,
  4.44% base rate); the task is to predict which loans fail to reach their funding goal.
