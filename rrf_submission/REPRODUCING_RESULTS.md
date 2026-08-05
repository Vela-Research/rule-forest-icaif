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
```
- Rule Forest **0.109 / 0.082 / 0.102** — `results_full_leakfree.json` panel a ("LLM-Only").
- Similarity ablation **0.102 / 0.106 / 0.097** and Hamming-threshold ablation **0.090 / 0.102 / 0.081** — panels b, c.
- Expert-in-the-loop (precision **0.109 -> 0.164**, F0.5 **0.102 -> 0.125**) — panel a ("LLM+Expert").
- Per-generator vote / elastic-net panel (GPT-4o-mini 0.102/0.080, Claude 0.083/0.061, GPT-5.2 0.064/0.041) — `results_model_panel.json`.

## 2. Classical baselines (Engineered ML, Embeddings)
```
baselines/traditional_ml_port/results/test_set_protocol_b_summary.csv
```
Row `R1-LR-ENet` -> Engineered ML **0.086**; row `R2-XGB` -> Embeddings **0.069**.

## 3. GPTree / RRM (Table 2)
The paper scores GPTree and RRM under the **same 10x10 nested CV** as Rule Forest, reusing each
method's frozen artifact (tree / mined rules) and re-fitting only the data-dependent part per fold:
```bash
cd baselines_llm/rescore && python rescore_gptree_rrm.py   # GPTree ~0.066, RRM ~0.076
```
See `baselines_llm/rescore/README.md`. (The single-split runs in `baselines_llm/{gptree,rrm}/*_results.json`
report a different, lower headline; the paper uses the 10x10 re-scoring.)

## 4. Figures 4 & 5
```bash
cd count_effect
python make_figure4.py   # founder count-effect (1.9% / 5% / reliability); rho = -0.48
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
| Engineered ML / Embeddings (Table 2) | `baselines/traditional_ml_port/results/test_set_protocol_b_summary.csv` | 0.086 / 0.069 |
| LogReg elastic net (Table 2) | `rrf_submission/precomputed/gpt_4o_mini/benchmark_results/benchmark_gpt_4o_mini.csv` | 0.080 |
| Prompting o3 / GPT-4o-mini (Table 2) | `rrf_submission/precomputed/gpt_4o_mini/vanilla/` | 0.088 / 0.086 |
| GPTree / RRM (Table 2) | `baselines_llm/rescore/rescore_gptree_rrm.py` | 0.066 / 0.076 |
| Similarity / threshold ablations (Table 3) | `results_full_leakfree.json` (panels b, c) | 0.102/0.106/0.097 ; 0.090/0.102/0.081 |
| Generator ablation (Table 3) | `results_model_panel.json` | GPT-4o-mini 0.102/0.080, Claude 0.083/0.061, GPT-5.2 0.064/0.041 |
| Figure 4 (founder count-effect) | `count_effect/make_figure4.py` | rho = -0.48 |
| Figure 5 (Kiva) | `count_effect/make_figure5.py` | vote vs elastic net vs base rate |

## Notes
- The reproduction notebook (`notebooks/paper_reproduction.ipynb`) covers the founder Rule Forest and
  prompting rows; the full Table 2 (classical + GPTree/RRM) and Figures 4-5 are produced by the scripts
  above.
- The chronological-split robustness check needs per-founder founding dates that are not part of this
  anonymized release.
