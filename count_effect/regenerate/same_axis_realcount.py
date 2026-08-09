"""Regenerate count_effect/data/cache/panel_a_1p9.json (Figure 4a).

Held-out F0.5 of the equal-weight vote vs elastic net at the REAL 1.9% founder operating
rate, as the number of training successes varies. Nested CV (n_repeats=5, n_splits=8) on the
22-question deduplicated founder evaluation matrix (the Table-2 Rule Forest matrix). For each
count < full, positives are thinned to `count` and negatives added to hold the natural 1.9%
prevalence (DRAWS seeded draws, averaged); the full count is the whole matrix (single draw).

Deterministic (sha256 seeds), parallel across cores.
  python same_axis_realcount.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, time, argparse
from pathlib import Path
from multiprocessing import Pool
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import loaders, sweep_core as sc

OUT = HERE.parent / "data" / "cache" / "panel_a_1p9.json"
COUNTS = [15, 30, 60, 100, 140, 158]
N_REPEATS, N_SPLITS = 5, 8
DRAWS = 10

_X = _Y = None
_PREV = None
def _init():
    global _X, _Y, _PREV
    _X, _Y = loaders.load_founders_22q()
    _PREV = float(_Y.mean())            # natural ~1.93%


def _work(task):
    count, draw = task
    if count >= int(_Y.sum()):
        Xs, ys = _X, _Y
    else:
        Xs, ys = sc.sub_fixed_prev(_X, _Y, count, _PREV, sc.det_seed("panel_a", count, draw))
    r = sc.paired_eval(Xs, ys, N_REPEATS, N_SPLITS)
    return {"count": count, "draw": draw, "vote": r["rrf_f05"], "en": r["en_f05"]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nproc", type=int, default=14)
    args = ap.parse_args()
    _, y = loaders.load_founders_22q(); npos_full = int(y.sum())
    tasks = [(c, d) for c in COUNTS for d in range(1 if c >= npos_full else DRAWS)]
    print(f"panel_a: {len(tasks)} evals over counts {COUNTS} (n_repeats={N_REPEATS}, "
          f"n_splits={N_SPLITS}), nproc={args.nproc}", flush=True)
    t0 = time.time()
    with Pool(min(args.nproc, len(tasks)), initializer=_init) as pool:
        rows = list(pool.imap_unordered(_work, tasks))
    vote = [round(float(np.mean([r["vote"] for r in rows if r["count"] == c])), 4) for c in COUNTS]
    en = [round(float(np.mean([r["en"] for r in rows if r["count"] == c])), 4) for c in COUNTS]
    out = {
        "_provenance": (f"Figure 4(a): held-out F0.5 at the real 1.9% operating rate vs number "
                        f"of training successes. Nested CV (n_repeats={N_REPEATS}, "
                        f"n_splits={N_SPLITS}) on the 22-question deduplicated founder evaluation "
                        f"matrix; positives thinned to each count with negatives added to hold "
                        f"~1.9% prevalence, {DRAWS} seeded draws averaged (full count = whole "
                        f"matrix). Regenerate with regenerate/same_axis_realcount.py."),
        "counts": COUNTS, "vote": vote, "elastic_net": en,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT} in {time.time()-t0:.0f}s", flush=True)
    for c, v, e in zip(COUNTS, vote, en):
        print(f"  count={c:3d} vote={v:.4f} en={e:.4f} gap={v-e:+.4f}", flush=True)


if __name__ == "__main__":
    main()
