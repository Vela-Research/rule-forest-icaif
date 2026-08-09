# Traditional-ML baselines (Table 2: Engineered ML / Embeddings)

Two classical non-LLM baselines the paper compares Rule Forest against, evaluated on the
8,204-founder evaluation set under the paper's protocol (10×10 repeated nested CV, per-fold
F0.5-optimal threshold, per-repeat-pooled F0.5).

## Engineered ML — reproducible from the anonymised prose (`engineered_ml/`)

`engineered_ml/` re-applies the engineered-features **method** to this dataset: ~30 deterministic
features parsed from each anonymised founder summary with keyword/pattern rules
(`build_features.py`), standardised, and fed to a class-balanced elastic-net logistic regression
(`run_engineered_ml.py`). No LLM, no network, no external data — reproducible from the shipped
`03_full_cross_validation_test_data_anonymised.csv` alone.

```bash
cd engineered_ml && python run_engineered_ml.py     # -> engineered_ml_result.json (+ features + consensus)
```

**Result:** Precision 0.071 ± 0.013, Recall 0.055 ± 0.020, **F0.5 0.066 ± 0.014** (33 features,
8,204 founders, 158 positives).

Note on feature design: the feature set was hand-designed from domain knowledge and the summary
vocabulary; per-feature label association was inspected on the full set during design (no
label-based pruning), so the value may be mildly optimistic as a baseline — in the conservative
direction for Rule Forest, which leads it comfortably. The cross-validation itself is leakage-free
(features are deterministic per-founder functions; scaler/model/threshold fit on train folds only).

## Embeddings

`results/` reports the Embeddings baseline (XGBoost on `all-MiniLM-L6-v2` prose embeddings) at
**F0.5 0.069**. This is also reproducible in principle from the shipped anonymised prose (encode
summaries with MiniLM → XGBoost) but is shipped here as a frozen result; a self-contained
producer is not yet included (it needs `sentence-transformers`/`torch`/`xgboost`; see
`training/requirements-training.txt`, kept out of the main requirements to keep the install lean).

## Shipped result artifacts + offline verification

- `results/test_set_protocol_b.jsonl`, `..._summary.csv` — the original per-fold results for both
  rows (the old Engineered-ML producer was not retained; `engineered_ml/` supersedes it).
- `results/consensus_allrepeats.parquet` — per-founder consensus predictions per model; the paired
  bootstrap in `../leakage_free_ablation/section3_significance.py` operates on these.
- `evaluate_from_artifacts.py` — recomputes the frozen headline F0.5 from the shipped per-fold data.
