# Count-effect figures (paper Figures 4 & 5)

Self-contained reproduction of the two "When counting beats weighting" figures. Reads only files
under `data/`; no network, no API key, no dependency on any other part of the release.

## Reproduce

```bash
pip install numpy pandas pyarrow scipy matplotlib
python make_figure4.py    # -> figures/mech_final.pdf     (founder count-effect: 1.9% / 5% / reliability)
python make_figure5.py    # -> figures/kiva_D2_final.pdf  (Kiva: vote vs elastic net as the base rate falls)
```

## What each figure reads

**Figure 4** (`make_figure4.py`)
- panel (a) 1.9% operating rate: `data/cache/panel_a_1p9.json` (cached nested-CV means).
- panel (b) 5% pool: means computed live from `data/cache/results_founders.jsonl` (points 15, 456) and
  `data/cache/panelb_newpoints.jsonl` (points 120, 240, 360).
- panel (c) reliability scatters: computed live from `data/founders_answers.parquet` +
  `data/founders_data.parquet` (fixed seeds; the panel titles' rank correlations recompute each run).
- Spearman `rho = -0.48` (text): `data/cache/summary_founders.json`.

**Figure 5** (`make_figure5.py`)
- everything live from `data/cache/kiva_D2.jsonl` (10x10 nested-CV draws; 95% CIs and the base-rate top
  axis are recomputed).

## Regenerating the caches from scratch (slow path)

The cached JSONL files hold the 10x10 nested-CV sweeps, which take hours. The producing scripts and the
minimal Rule Forest scoring primitives they need live in `regenerate/` (see `regenerate/README.md`). In brief:

- `data/cache/kiva_D2.jsonl` from `data/kiva_rq_scores_run1.jsonl`:
  `python regenerate/kiva_D2_sweep.py && python regenerate/kiva_D2_k93.py`
- `data/cache/panelb_newpoints.jsonl` and `results_founders.jsonl` from the founder matrix:
  `PYTHONHASHSEED=0 python regenerate/run_panelb_points.py` (the `results_founders.jsonl` seeding uses
  Python's builtin `hash()`, so exact-byte reproduction needs `PYTHONHASHSEED=0`; the draw means are
  unaffected either way).
- `data/cache/panel_a_1p9.json` from the 22-question founder evaluation matrices:
  `python regenerate/same_axis_realcount.py`.

The outer nested cross-validation is deterministic everywhere (`StratifiedKFold(random_state=42+repeat-1)`).
