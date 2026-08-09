"""Regenerate count_effect/data/cache/kiva_D2.jsonl (Figure 5).

Kiva-D2 design: keep ALL negatives, thin the positive class to `count`, so the base rate
falls from 4.44% (full) toward the founder 1.9% rate (count=93). For each count, DRAWS
independent seeded subsamples are evaluated under the paired unit-vote-vs-elastic-net
10x10 nested CV; count=222 (all positives) is a single deterministic draw.

Deterministic (sha256 seeds) and parallel across cores. Overwrites the cache atomically.
  python kiva_D2_sweep.py            # full grid
  python kiva_D2_sweep.py --nproc 8
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

OUT = HERE.parent / "data" / "cache" / "kiva_D2.jsonl"
PLOT_K = [15, 30, 60, 93, 130, 180, 222]   # 93 = the 1.9% founder-rate point; 222 = full data
DRAWS = 10

_X = _Y = None
def _init():
    global _X, _Y
    _X, _Y = loaders.load_kiva()


def _work(task):
    count, draw = task
    if count >= int(_Y.sum()):                 # all positives -> deterministic single draw
        Xs, ys = _X, _Y
    else:
        Xs, ys = sc.sub_all_neg(_X, _Y, count, sc.det_seed("kiva_D2", count, draw))
    t0 = time.time()
    r = sc.paired_eval(Xs, ys, 10, 10)
    return {"count": count, "draw": draw, "rrf_f05": r["rrf_f05"], "en_f05": r["en_f05"],
            "gap": r["gap"], "n": r["n"], "n_pos": r["n_pos"], "prevalence": r["prevalence"],
            "secs": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=14)
    args = ap.parse_args()
    _, y = loaders.load_kiva(); npos_full = int(y.sum())
    tasks = [(c, d) for c in PLOT_K for d in range(1 if c >= npos_full else DRAWS)]
    print(f"kiva_D2: {len(tasks)} evals over counts {PLOT_K}, nproc={args.nproc}", flush=True)
    t0 = time.time()
    with Pool(min(args.nproc, len(tasks)), initializer=_init) as pool:
        rows = [r for r in pool.imap_unordered(_work, tasks) if r]
    rows.sort(key=lambda r: (r["count"], r["draw"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {OUT} ({len(rows)} rows) in {time.time()-t0:.0f}s", flush=True)
    import collections
    by = collections.defaultdict(list)
    for r in rows: by[r["count"]].append(r)
    for k in PLOT_K:
        v = [x["rrf_f05"] for x in by[k]]; e = [x["en_f05"] for x in by[k]]
        print(f"  count={k:3d} nd={len(v):2d} vote={np.mean(v):.4f} en={np.mean(e):.4f} "
              f"gap={np.mean(v)-np.mean(e):+.4f} prev={np.mean([x['prevalence'] for x in by[k]]):.4f}", flush=True)


if __name__ == "__main__":
    main()
