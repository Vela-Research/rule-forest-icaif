# Regenerating the count-effect caches

These scripts regenerate every cache under `../data/cache/` that Figures 4 and 5 read.
They are **deterministic** (all subsample seeds come from a sha256 of the arguments, so no
`PYTHONHASHSEED` dependency) and **parallel** (one process per core; pass `--nproc N`).
Re-running a script reproduces its cache exactly.

Each eval is the paired **unit-vote vs elastic-net** comparison on the same binary question
matrix under a 10×10 repeated nested CV (panel (a) uses 5×8), with the inner model-selection
CV fixed at 5 folds for **both** arms. The Rule Forest arm reuses the shipped scoring
primitives (`baselines/leakage_free_ablation/run_ablation.py`,
`rrf_submission/src/utils/scoring_utils.py`); the elastic-net primitives are inlined in
`sweep_core.py`, so `count_effect/` depends on nothing under `baselines/prevalence_sweep/`.

| Script | Regenerates | Figure | Grid | Approx cost |
|---|---|---|---|---|
| `kiva_D2_sweep.py` | `kiva_D2.jsonl` | 5 | positives thinned to [15,30,60,93,130,180,222], all negatives kept | minutes |
| `same_axis_realcount.py` | `panel_a_1p9.json` | 4(a) | 22Q deduped matrix, counts [15,30,60,100,140,158] @ 1.9% | minutes |
| `run_panelb_points.py` | `results_founders.jsonl`, `panelb_newpoints.jsonl` | 4(b) | full 62Q matrix: natural(456) + npos_nat sweep + [120,240,360] | ~1 hour |
| `analyze_founders.py` | `summary_founders.json` (ρ) | 4 caption | reads `results_founders.jsonl` | seconds |

Typical order:
```bash
python kiva_D2_sweep.py            # Figure 5
python same_axis_realcount.py      # Figure 4(a)
python run_panelb_points.py        # Figure 4(b)  (heavy)
python analyze_founders.py         # Figure 4 caption ρ  (after run_panelb_points)
```
Then redraw the figures from the refreshed caches:
```bash
cd .. && python make_figure4.py && python make_figure5.py
```

Shared modules: `sweep_core.py` (paired evaluator + subsamplers + inlined EN primitives) and
`loaders.py` (release-relative data loaders). Figure 4 panel (c) is computed live inside
`make_figure4.py` from the shipped parquets and needs no cache.
