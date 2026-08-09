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
- Spearman `rho = -0.517` (text): `data/cache/summary_founders.json`.

**Figure 5** (`make_figure5.py`)
- everything live from `data/cache/kiva_D2.jsonl` (10x10 nested-CV draws; 95% CIs and the base-rate top
  axis are recomputed).

## Regenerating the caches from scratch

The cached JSONL/JSON files hold the nested-CV sweeps behind Figures 4 and 5. The producing
scripts and the scoring primitives they need are self-contained under `regenerate/` (see
`regenerate/README.md`). All are **deterministic** (sha256 subsample seeds — no
`PYTHONHASHSEED` dependency) and parallel (`--nproc N`):

- `data/cache/kiva_D2.jsonl` (Figure 5) from `data/kiva_rq_scores_run1.jsonl`:
  `python regenerate/kiva_D2_sweep.py`
- `data/cache/panel_a_1p9.json` (Figure 4a) from the 22-question founder matrix:
  `python regenerate/same_axis_realcount.py`
- `data/cache/results_founders.jsonl` + `panelb_newpoints.jsonl` (Figure 4b) from the founder matrix:
  `python regenerate/run_panelb_points.py`   (heavy; ~1 hour)
- `data/cache/summary_founders.json` (Figure 4 caption ρ) from `results_founders.jsonl`:
  `python regenerate/analyze_founders.py`

The outer nested cross-validation is deterministic everywhere (`StratifiedKFold(random_state=42+repeat-1)`).
