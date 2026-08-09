"""Regenerate the FOUNDER count-effect caches for Figure 4(b):
  - results_founders.jsonl : natural full-pool point (n_pos=456) + the npos_nat sweep
                             (vary positive COUNT at natural 4.96% prevalence, log grid to 15).
  - panelb_newpoints.jsonl : the intermediate founder points (n_pos = 120, 240, 360).

Full 62-question / 9192-founder matrix. Paired unit-vote-vs-elastic-net 10x10 nested CV,
inner=5 selection, 10 draws/point (natural = single deterministic draw). Deterministic
(sha256 seeds), parallel across cores. This is the heavy driver (large matrices).
  python run_panelb_points.py --nproc 14
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

CACHE = HERE.parent / "data" / "cache"
OUT_MAIN = CACHE / "results_founders.jsonl"
OUT_NEW = CACHE / "panelb_newpoints.jsonl"
DRAWS = 10
REPEATS, SPLITS = 10, 10
CAP_N, NPOS_FLOOR, K_MAIN_FOUNDERS = 2200, 15, 4
NEWPOINTS = [120, 240, 360]

_X = _Y = None
_BR = None
def _init():
    global _X, _Y, _BR
    _X, _Y = loaders.load_founders()
    _BR = float(_Y.mean())


def geomgrid(lo, hi, k):
    if hi <= lo:
        return [int(lo)]
    return [int(v) for v in np.unique(np.round(np.geomspace(lo, hi, k)).astype(int))]


def _grid():
    npos_full = 456
    cap = max(min(int(round(0.9 * npos_full)), int(round(CAP_N * 0.0496))), NPOS_FLOOR + 5)
    return geomgrid(NPOS_FLOOR, cap, K_MAIN_FOUNDERS)


def _work(task):
    kind, val, draw = task
    if kind == "natural":
        Xs, ys = _X, _Y
    else:
        seed = sc.det_seed("founders", kind, val, draw)
        Xs, ys = sc.sub_fixed_prev(_X, _Y, val, _BR, seed)
        if int(ys.sum()) < 2 or int((ys == 0).sum()) < 2:
            return None
    t0 = time.time()
    r = sc.paired_eval(Xs, ys, REPEATS, SPLITS)
    base = {"n": r["n"], "n_pos": r["n_pos"], "prevalence": r["prevalence"],
            "rrf_f05": r["rrf_f05"], "en_f05": r["en_f05"], "gap": r["gap"],
            "n_repeats": r["n_repeats"], "n_splits": r["n_splits"],
            "inner_splits": r["inner_splits"], "secs": round(time.time() - t0, 1)}
    if kind == "newpoint":
        return ("new", {"count": int(val), "draw": draw, **base})
    if kind == "natural":
        return ("main", {"dataset": "founders", "display": "Founders", "arm": "natural",
                         "point_id": "full", "pidx": 0, "draw": draw, "target_npos": 456,
                         "target_prev": _BR, "base_rate": _BR, **base})
    return ("main", {"dataset": "founders", "display": "Founders", "arm": "npos_nat",
                     "point_id": f"npos{val}", "pidx": val, "draw": draw,
                     "target_npos": int(val), "target_prev": _BR, "base_rate": _BR, **base})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nproc", type=int, default=14)
    args = ap.parse_args()
    grid = _grid()
    tasks = [("natural", 456, 0)]
    tasks += [("npos_nat", n, d) for n in grid for d in range(DRAWS)]
    tasks += [("newpoint", c, d) for c in NEWPOINTS for d in range(DRAWS)]
    print(f"founder panel: {len(tasks)} evals | npos_nat grid={grid} | newpoints={NEWPOINTS} "
          f"| nproc={args.nproc}", flush=True)
    t0 = time.time()
    main_rows, new_rows = [], []
    with Pool(min(args.nproc, len(tasks)), initializer=_init) as pool:
        for res in pool.imap_unordered(_work, tasks):
            if res is None:
                continue
            (main_rows if res[0] == "main" else new_rows).append(res[1])
            done = len(main_rows) + len(new_rows)
            if done % 10 == 0:
                print(f"  ... {done}/{len(tasks)} done ({time.time()-t0:.0f}s)", flush=True)
    main_rows.sort(key=lambda r: (r["arm"], r["pidx"], r["draw"]))
    new_rows.sort(key=lambda r: (r["count"], r["draw"]))
    OUT_MAIN.write_text("\n".join(json.dumps(r) for r in main_rows) + "\n")
    OUT_NEW.write_text("\n".join(json.dumps(r) for r in new_rows) + "\n")
    print(f"wrote {OUT_MAIN.name} ({len(main_rows)}) + {OUT_NEW.name} ({len(new_rows)}) "
          f"in {time.time()-t0:.0f}s", flush=True)
    # summary
    def mean(rows, k): return float(np.mean([r[k] for r in rows])) if rows else float("nan")
    nat = [r for r in main_rows if r["arm"] == "natural"]
    print(f"  natural: n_pos={nat[0]['n_pos']} vote={nat[0]['rrf_f05']:.4f} en={nat[0]['en_f05']:.4f} "
          f"gap={nat[0]['gap']:+.4f}", flush=True)
    for n in grid:
        rs = [r for r in main_rows if r["point_id"] == f"npos{n}"]
        print(f"  npos{n:<4d} vote={mean(rs,'rrf_f05'):.4f} en={mean(rs,'en_f05'):.4f} "
              f"gap={mean(rs,'rrf_f05')-mean(rs,'en_f05'):+.4f}", flush=True)
    for c in NEWPOINTS:
        rs = [r for r in new_rows if r["count"] == c]
        print(f"  new{c:<4d} vote={mean(rs,'rrf_f05'):.4f} en={mean(rs,'en_f05'):.4f} "
              f"gap={mean(rs,'rrf_f05')-mean(rs,'en_f05'):+.4f}", flush=True)


if __name__ == "__main__":
    main()
