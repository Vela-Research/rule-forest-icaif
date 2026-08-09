# Traditional-ML baselines (Table 2: Engineered ML / Embeddings)

Two classical non-LLM baselines the paper compares Rule Forest against, evaluated under the
paper's protocol (**10×10 repeated nested CV on the 8,204-founder evaluation set**, reported at
the fold-tuned F0.5 threshold):

| Table-2 row | Method | F0.5 |
|---|---|---|
| Engineered ML | LR-ElasticNet on 114 hand-crafted structured features | **0.086** |
| Embeddings | XGBoost on `all-MiniLM-L6-v2` (384-d) founder-prose embeddings | **0.069** |

## What ships here

- `results/test_set_protocol_b.jsonl` — the full per-repeat / per-fold results (thresholds,
  F0.5, precision, recall, ROC/PR-AUC) for both methods across the 10×10 CV.
- `results/test_set_protocol_b_summary.csv` — the aggregated headline row per method (the
  0.086 / 0.069 above are `f05_mean`).
- `results/consensus_allrepeats.parquet` — per-founder consensus (majority-vote-over-repeats)
  predictions for R1, R2, and the LLM methods; this is what
  `../leakage_free_ablation/section3_significance.py` bootstraps.
- `evaluate_from_artifacts.py` — **offline verification** (numpy/pandas only): recomputes the
  reported F0.5 from the shipped per-fold data and from the per-founder consensus predictions.

```bash
python evaluate_from_artifacts.py     # confirms 0.086 / 0.069 match the shipped per-fold data
```

## Data-availability limitation (why these are shipped as frozen results)

Both baselines are trained on the **non-public VCBench founder records**, which are **not part
of this anonymized release**:

- **Engineered ML** parses 114 features from the *raw structured* founder fields — job history
  JSON, education JSON, IPO / acquisition records, industry, company sizes. The release ships
  only anonymized natural-language summaries (`founder_uuid, summary, startup_success`), not
  these raw records; publishing them would de-anonymize the founders.
- **Embeddings** encodes the *original* (non-anonymized) founder prose with MiniLM.

So neither row can be **retrained** from this release — it is a restricted-data item, like the
other non-public inputs noted in the top-level README. What is verifiable offline is that the
reported numbers are the faithful aggregate of the shipped per-fold results and match the
per-founder consensus predictions (`evaluate_from_artifacts.py`).

## Method detail (for independent reproduction on the raw VCBench data)

**Engineered ML.** 114 deterministic features in nine groups: career structure (8), role
flags/counts (26), role-seniority trajectory (5), career tenure (8), company-size trajectory
(3), industry alignment (4), education (11), founder/exit signals (6), and a top-20 industry
one-hot (~20). Model: `StandardScaler` → `LogisticRegression(penalty="elasticnet",
solver="saga", class_weight="balanced")`, with `(C, l1_ratio)` chosen by inner 3-fold
`GridSearchCV` under `neg_log_loss`; the decision threshold is the F0.5-optimal cut on pooled
out-of-fold scores.

**Embeddings.** `all-MiniLM-L6-v2` (frozen, L2-normalized, 384-d) sentence embeddings of the
founder prose → XGBoost, hyper-parameters chosen by inner 3-fold `RandomizedSearchCV` (30
trials) under `neg_log_loss`; same F0.5-optimal OOF threshold rule.

Reproducing these from the raw VCBench CSVs additionally requires
`sentence-transformers` / `torch` / `xgboost` — see `training/requirements-training.txt`. These
are deliberately **not** in the top-level `requirements.txt`: they are unnecessary for every
offline reproduction the release actually supports (all Rule Forest tables/figures + the
verification above), and would make a clean install substantially heavier.
