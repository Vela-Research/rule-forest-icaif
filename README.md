# Random Rule Forest (RRF) — Reproduction Package

Anonymized code and data accompanying the paper *"Random Rule Forest (RRF):
Interpretable and Manageable Ensembles of LLM-Generated Questions for Predicting
Success from Unstructured Data."* Everything needed to reproduce the paper's tables
and figures **offline** (no API key, no network) from precomputed LLM outputs is
included.

## Layout
```
requirements.txt
rrf_submission/
  src/                       # scoring + question-filtering modules
  precomputed/               # anonymized LLM question-response matrices + baseline outputs
  data/                      # anonymized founder dataset (9,192 profiles)
  clinical_trials/           # TOP Phase-I benchmark pipeline, data, and results
  notebooks/paper_reproduction.ipynb
baselines/
  leakage_free_ablation/     # canonical leakage-free RRF harness (reproduces F0.5 = 0.102)
  traditional_ml_port/       # Engineered-ML and Embeddings baseline outputs
baselines_llm/               # GPTree / Policy Induction / RRM transplant results (per seed)
```

## Quickstart
Requires **Python ≥ 3.11** (the pinned dependencies and type hints need it).
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook rrf_submission/notebooks/paper_reproduction.ipynb   # all tables + figures, offline
```
The offline reproduction needs only `numpy pandas scipy scikit-learn matplotlib
pyarrow`; the rest of `requirements.txt` is for regenerating LLM outputs from scratch.

## Reproducing the headline numbers (offline, no API key)
Founder RRF result — the **leakage-free harness**:
```bash
cd baselines/leakage_free_ablation
python full_leakfree.py        # -> results_full_leakfree.json  (RRF F0.5=0.102; ablations; expert 0.125)
python run_ablation.py --configs '[["gpt_4o_mini","hamming",0.15]]' --out /tmp/verify.json  # leakfree_f05 = 0.1019
python section3_significance.py  # Section 5.3 paired bootstrap
```
Clinical trials (TOP Phase-I, Table 3):
```bash
cd rrf_submission
python clinical_trials/pipeline/09_primary_model.py  --config clinical_trials/configs/primary_model_locked.json  --force   # RRF    0.638 / 0.596 / 0.655
python clinical_trials/pipeline/11_logreg_baseline.py --config clinical_trials/configs/logreg_baseline_locked.json --force  # LogReg 0.632 / 0.594 / 0.641
```

## Paper → artifact map
| Paper item | Value | Where |
|---|---|---|
| Table 1 — RRF | 0.109 / 0.082 / 0.102 | `baselines/leakage_free_ablation/results_full_leakfree.json` (panel a, LLM-Only) |
| Table 1 — Engineered ML / Embeddings | 0.086 / 0.069 | `baselines/traditional_ml_port/results/test_set_protocol_b_summary.csv` |
| Table 1 — LogReg (EN) | 0.080 | `rrf_submission/precomputed/gpt_4o_mini/benchmark_results/benchmark_gpt_4o_mini.csv` |
| Table 1 — prompting (o3, GPT-4o-mini) | 0.088 / 0.086 | `rrf_submission/precomputed/gpt_4o_mini/vanilla/` |
| Table 1 — GPTree / Policy Induction / RRM | 0.040 / 0.058 / ~0.05 | `baselines_llm/{gptree,policy_induction,rrm}/` (per-seed) |
| Table 2 — similarity / threshold ablations | 0.102 / 0.106 / 0.097 ; 0.090 / 0.102 / 0.081 | `results_full_leakfree.json` (panels b, c) |
| Table 2 — question-generating model | 0.102 / 0.083 / 0.081 / 0.064 | `results_model_panel.json` |
| Expert-in-the-loop | precision 0.109→0.164 ; F0.5 0.102→0.125 | `results_full_leakfree.json` (panel a) |
| Table 3 — TOP Phase-I | RRF 0.638/0.596/0.655 ; LogReg 0.632/0.594/0.641 | `rrf_submission/clinical_trials/results/phase_I/` |

## Notes on reproducibility
- **Leakage-free harness.** The canonical RRF numbers come from
  `baselines/leakage_free_ablation/`, which ranks candidate questions **within each
  training fold**. (An earlier variant ranked questions by a score computed over the
  full dataset; the harness here is the corrected, leakage-free version. Set
  `RRF_LEGACY_PHANTOM=1` only if you want to reproduce the legacy anchor.)
- **Offline core vs. API pipeline.** All paper numbers reproduce offline from the
  shipped precomputed LLM outputs. Regenerating those outputs from scratch (question
  generation + scoring) requires an LLM API key; the offline reproduction does not.
- **Restricted-data items.** The chronological-split robustness check requires
  per-founder founding dates that are not part of this anonymized release. GPTree,
  Policy Induction, and RRM are external methods (see the paper's references); their
  per-seed result JSONs are included here, but regenerating them requires those
  methods' own implementations plus an API key.
- **Contamination probe.** `rrf_submission/clinical_trials/top_direct_prompting_probe.py`
  (needs an API key and the public TOP Phase-I test set) supports the memorization
  check discussed in the paper; its output is in
  `rrf_submission/clinical_trials/results/top_direct_prompting_probe.json`.

## Data
- **Founders** (`rrf_submission/data/founders_summary_data_anonymised.csv`): 9,192
  U.S. companies founded 2010–2016, transformed into anonymized natural-language
  summaries. The held-out evaluation set is 8,204 founders with 158 positives
  (1.93% base rate).
- **Clinical trials**: the public TOP Phase-I benchmark (Fu et al., 2022),
  represented as question-response matrices under `rrf_submission/clinical_trials/`.
