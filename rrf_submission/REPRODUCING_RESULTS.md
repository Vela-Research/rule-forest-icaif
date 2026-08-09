# Reproducing Paper Results

Step-by-step guide to reproduce every table and figure in *"Rule Forest"* **offline**
(no API key, no network) from the shipped precomputed LLM outputs. All paths are relative
to the release root. See the top-level `README.md` for a one-line summary of each command.

## Prerequisites
```bash
pip install -r requirements.txt   # offline core: numpy pandas scipy scikit-learn matplotlib pyarrow
```

## 1. Founder table (Rule Forest, ablations, expert-in-the-loop)
The canonical numbers come from the **leakage-free harness**, which ranks candidate questions
within each training fold (leak-correction history 0.118 -> 0.105 -> 0.102):
```bash
cd baselines/leakage_free_ablation
python full_leakfree.py        # -> results_full_leakfree.json
python make_model_panel.py     # -> results_model_panel.json
python section3_significance.py
python generator_significance.py   # -> results/generator_significance.json (paper 5.3 generator-comparison bootstrap)
```
- Rule Forest **0.109 / 0.082 / 0.102** — `results_full_leakfree.json` panel a ("LLM-Only").
- Similarity ablation **0.102 / 0.106 / 0.097** and Hamming-threshold ablation **0.090 / 0.102 / 0.081** — panels b, c.
- Expert-in-the-loop (precision **0.109 -> 0.164**, F0.5 **0.102 -> 0.125**) — panel a ("LLM+Expert").
- Per-generator vote / elastic-net panel (GPT-4o-mini 0.102/0.080, Claude 0.083/0.061, GPT-5.2 0.064/0.041) — `results_model_panel.json`.
- Generator-comparison significance (paper 5.3): GPT-4o-mini vs GPT-5.2 Delta=**+0.038**, 95% CI [-0.01, +0.09], two-sided p **0.14** (one-sided 0.07); vs Claude Delta=+0.020, p=0.47 — `generator_significance.py` / `results/generator_significance.json`.

## 2. Classical baselines (Engineered ML, Embeddings)
```bash
cd baselines/traditional_ml_port/engineered_ml && python run_engineered_ml.py   # Engineered ML 0.066
```
Engineered ML **0.066** reproduces from the anonymised prose with the core requirements alone.
Embeddings **0.047** is shipped as `baselines/traditional_ml_port/embeddings/embeddings_result.json`
(re-encoding the prose needs `training/requirements-training.txt`: torch / xgboost / sentence-transformers).

## 3. GPTree / RRM (Table 2)
The paper scores GPTree and RRM under the **same 10x10 nested CV** as Rule Forest, reusing each
method's frozen artifact (tree / mined rules) and re-fitting only the data-dependent part per fold:
```bash
cd baselines_llm/rescore && python rescore_gptree_rrm.py   # GPTree 0.068, RRM 0.080
```
See `baselines_llm/rescore/README.md`. (The single-split runs in `baselines_llm/{gptree,rrm}/*_results.json`
report a different, lower headline; the paper uses the 10x10 re-scoring.)

## 4. Figures 4 & 5
```bash
cd count_effect
python make_figure4.py   # founder count-effect (1.9% / 5% / reliability); rho = -0.517
python make_figure5.py   # Kiva generalisation (vote vs elastic net as the base rate falls)
```
See `count_effect/README.md` for the cache-regeneration (slow) path.

## Why precomputed artifacts?
Rule Forest uses an LLM for question generation and scoring, which is non-deterministic and
API-dependent. The release ships CSV/JSON snapshots of all LLM outputs so the paper numbers
reproduce exactly without API access. Regenerating those snapshots from scratch needs an LLM API
key and is out of scope for offline reproduction.

## Artifact map
| Paper item | Artifact | Value |
|---|---|---|
| Rule Forest (Table 2) | `baselines/leakage_free_ablation/results_full_leakfree.json` (panel a) | 0.109 / 0.082 / 0.102 |
| Engineered ML / Embeddings (Table 2) | `baselines/traditional_ml_port/engineered_ml/run_engineered_ml.py` ; `embeddings/embeddings_result.json` | 0.066 / 0.047 |
| LogReg elastic net (Table 2) | `rrf_submission/precomputed/gpt_4o_mini/benchmark_results/benchmark_gpt_4o_mini.csv` | 0.080 |
| Prompting o3 / GPT-4o-mini (Table 2) | `rrf_submission/precomputed/gpt_4o_mini/vanilla/` | 0.088 / 0.086 |
| GPTree / RRM (Table 2) | `baselines_llm/rescore/rescore_gptree_rrm.py` | 0.068 / 0.080 |
| Similarity / threshold ablations (Table 3) | `results_full_leakfree.json` (panels b, c) | 0.102/0.106/0.097 ; 0.090/0.102/0.081 |
| Generator ablation (Table 3) | `results_model_panel.json` | GPT-4o-mini 0.102/0.080, Claude 0.083/0.061, GPT-5.2 0.064/0.041 |
| Figure 4 (founder count-effect) | `count_effect/make_figure4.py` | rho = -0.517 |
| Figure 5 (Kiva) | `count_effect/make_figure5.py` | vote vs elastic net vs base rate |

## Founder dataset split & counts

The 9,192-profile founder dataset is split three ways (defined by `founder_source_mapping.csv` in the
experiment repo; not shipped here):

| Split | Count | Role |
|---|---|---|
| Question generation | 497 | founders shown to the LLM to generate candidate questions (~20 per prompt) |
| Validation | 491 | held-out set for the question precision filter (drop questions with validation-set precision < 0.02) |
| Evaluation | 8,204 | the reported test set (158 positives, 1.9% base rate) |

`497 + 491 + 8,204 = 9,192`. Two things to know for exact reproduction:

- **Unique founders = 9,183.** Nine `founder_uuid`s are assigned to two splits each; five of those straddle
  validation∩evaluation, so the validation-based question filter incidentally saw 5 of the 8,204 evaluation
  founders (0.06%). Because validation is used only to discard near-zero-precision questions, this has no
  material effect on any reported number.
- **The scored evaluation matrix here has 8,173 founder columns, not 8,204.** Anonymisation truncates each
  `founder_uuid` to a 5-char hash; 54 evaluation founders share 27 such prefixes (−27), and 4 more have no
  scored response (−4), giving `8,204 − 27 − 4 = 8,173`. All 158 positives survive; base rate and F0.5
  (0.102) are unchanged. So a reviewer reproducing from this release will see 8,173 evaluation founders.

## Question generation chain (240 -> 62 -> 48 -> 22)

The paper's question counts map to the release as follows:

| Stage | Count | Where it lives in the release |
|---|---|---|
| Generated candidates (25 prompts x ~10 YES/NO) | 240 | prompt template in `PROMPTS.md` / `src/llms/prompts.py` |
| After semantic de-duplication (MiniLM cosine > 0.9) | 62 | the 62-question founder matrix `count_effect/data/founders_answers.parquet` (9,192 x 62) |
| After the validation-set precision screen (14 pruned to save API cost) | 48 | screen on the 491 validation founders (`src/questions/question_filtering.py`) |
| Final voting pool (Hamming-0.15 dedup + F0.5 rank) | 22 | the 22-question matrix `baselines/leakage_free_ablation/full_leakfree.py` builds and votes over (`Q=22`) |

Across the 100 outer folds the vote selects a median of 8 questions with threshold 6
(`run_ablation.py`, `leakfree_nq_median`).

## Notes
- The reproduction notebook (`notebooks/paper_reproduction.ipynb`) covers the founder Rule Forest and
  prompting rows; the full Table 2 (classical + GPTree/RRM) and Figures 4-5 are produced by the scripts
  above.
