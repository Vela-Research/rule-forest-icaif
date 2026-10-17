# Baselines

The Table 2 rows that are neither Rule Forest nor its elastic net. Each script writes one file to
`results/`, which `experiments/founders.py report` reads. Direct prompting needs no script here:
its predictions are in `data/founders/prompting/` and `experiments/founders.py` scores them.

| Folder | Table 2 row | Command | Output |
|---|---|---|---|
| `gptree_rrm/` | GPTree, RRM | `python baselines/gptree_rrm/rescore.py` | `results/gptree_rrm.json` |
| `engineered_ml/` | Engineered ML | `python baselines/engineered_ml/run.py` | `results/engineered_ml.json` |
| `embeddings/` | Embeddings | `python baselines/embeddings/run.py` | `results/embeddings.json` |

- **GPTree and RRM** were run with an LLM on the question-generation founders, three times each.
  `gptree_rrm/data/` keeps each run's output for the 8,204 evaluation founders, and
  `rescore.py` scores it under the paper's cross-validation without calling an LLM.
  `gptree_rrm/single_split_runs/` records the original runs' settings and single-split scores,
  for reference only.
- **Engineered ML** parses about 30 features from each profile with keyword rules and fits a
  class-balanced elastic net. It needs only `requirements.txt`.
- **Embeddings** fits XGBoost on frozen all-MiniLM-L6-v2 embeddings of each profile. The
  embeddings are shipped (`embeddings/embeddings_minilm.npy`), so `run.py` needs only xgboost
  (`pip install -r baselines/embeddings/requirements.txt`; on macOS also `brew install libomp`).
  Re-encoding the profiles with `build_embeddings.py` also needs sentence-transformers and torch,
  which that file lists separately.

All three use the same cross-validation design as Rule Forest (10 repeats of stratified 10-fold,
seeds 42 + repeat - 1) and the same pooled scoring. They score all 8,204 evaluation founders, not
the 8,173 of the answer matrix, so their folds are not the same founder sets.
