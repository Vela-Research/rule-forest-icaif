"""Embeddings row of Table 2. Frozen sentence embeddings of the anonymised profiles, then XGBoost.

build_embeddings.py encodes each evaluation founder's summary with all-MiniLM-L6-v2 (384
dimensions, L2-normalised; the result is shipped as embeddings_minilm.npy). This script fits an
imbalance-weighted XGBoost classifier under the paper's 10 x 10 cross-validation (outer seeds
42 + repeat - 1): depth, trees and learning rate are chosen by 3 inner folds and the threshold
maximises training F0.5. Scores are pooled within each repeat.

Needs xgboost (pip install -r baselines/embeddings/requirements.txt), which needs the OpenMP
runtime (on macOS: brew install libomp).

  python baselines/embeddings/run.py   # -> results/embeddings.json
"""
import os, warnings
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, argparse
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVAL = ROOT / "data" / "founders" / "profiles" / "evaluation.csv"
EMB = HERE / "embeddings_minilm.npy"
BETA2 = 0.25
N_REPEATS, N_SPLITS, INNER = 10, 10, 3
# inner grid over (max_depth, n_estimators, learning_rate); scale_pos_weight set per fold.
GRID = [(md, ne, lr) for md in (3, 4, 6) for ne in (200, 400) for lr in (0.05, 0.1)]


def f05c(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (0.0, 0.0, 0.0) if p + r == 0 else (1.25 * p * r / (0.25 * p + r), p, r)


def sweep_thr(y, s):
    best_t, best = float(s.min()) - 1.0, -1.0
    for t in np.unique(s):
        yp = (s >= t).astype(int)
        tp = int(((yp == 1) & (y == 1)).sum()); fp = int(((yp == 1) & (y == 0)).sum())
        fn = int(((yp == 0) & (y == 1)).sum())
        f = f05c(tp, fp, fn)[0]
        if f > best:
            best, best_t = f, float(t)
    return best_t


def _xgb(md, ne, lr, spw):
    return XGBClassifier(max_depth=md, n_estimators=ne, learning_rate=lr, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                         tree_method="hist", n_jobs=1, random_state=0, verbosity=0)


def tune_hp(Xtr, ytr):
    spw = float((ytr == 0).sum()) / max(1, int((ytr == 1).sum()))
    inner = max(2, min(INNER, int(min((ytr == 1).sum(), (ytr == 0).sum()))))
    skf = StratifiedKFold(inner, shuffle=True, random_state=0)
    best, bestg = -1.0, GRID[0]
    for (md, ne, lr) in GRID:
        sc = []
        for itr, iva in skf.split(Xtr, ytr):
            if ytr[itr].sum() < 1 or ytr[iva].sum() < 1:
                continue
            clf = _xgb(md, ne, lr, spw).fit(Xtr[itr], ytr[itr])
            thr = sweep_thr(ytr[itr], clf.predict_proba(Xtr[itr])[:, 1])
            yp = (clf.predict_proba(Xtr[iva])[:, 1] >= thr).astype(int)
            tp = int(((yp == 1) & (ytr[iva] == 1)).sum()); fp = int(((yp == 1) & (ytr[iva] == 0)).sum())
            fn = int(((yp == 0) & (ytr[iva] == 1)).sum())
            sc.append(f05c(tp, fp, fn)[0])
        m = float(np.mean(sc)) if sc else -1.0
        if m > best:
            best, bestg = m, (md, ne, lr)
    return bestg, spw


_X = _Y = None
def _init(X, y):
    global _X, _Y
    _X, _Y = X, y


def _fold(task):
    rep, tr, te = task
    Xtr, ytr, Xte, yte = _X[tr], _Y[tr], _X[te], _Y[te]
    (md, ne, lr), spw = tune_hp(Xtr, ytr)
    clf = _xgb(md, ne, lr, spw).fit(Xtr, ytr)
    thr = sweep_thr(ytr, clf.predict_proba(Xtr)[:, 1])
    yp = (clf.predict_proba(Xte)[:, 1] >= thr).astype(int)
    return {"repeat": rep,
            "TP": int(((yp == 1) & (yte == 1)).sum()),
            "FP": int(((yp == 1) & (yte == 0)).sum()),
            "FN": int(((yp == 0) & (yte == 1)).sum())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nproc", type=int, default=os.cpu_count())
    args = ap.parse_args()
    d = pd.read_csv(EVAL)
    X = np.load(EMB).astype(np.float32)
    y = d["startup_success"].astype(int).to_numpy()
    assert X.shape[0] == len(y), "embedding/label row mismatch"
    print(f"embeddings: {X.shape[0]} founders x {X.shape[1]}-d, {int(y.sum())} positives", flush=True)

    tasks = []
    for rep in range(1, N_REPEATS + 1):
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=42 + (rep - 1))
        for tr, te in skf.split(X, y):
            tasks.append((rep, tr, te))
    with Pool(min(args.nproc, len(tasks)), initializer=_init, initargs=(X, y)) as pool:
        rows = pool.map(_fold, tasks)

    per_rep = []
    for rep in range(1, N_REPEATS + 1):
        g = [r for r in rows if r["repeat"] == rep]
        f, p, r = f05c(sum(x["TP"] for x in g), sum(x["FP"] for x in g), sum(x["FN"] for x in g))
        per_rep.append({"repeat": rep, "f05": f, "precision": p, "recall": r})
    pooled = {}
    for k in ("precision", "recall", "f05"):
        v = [r[k] for r in per_rep]
        pooled[k], pooled[k + "_sd"] = float(np.mean(v)), float(np.std(v, ddof=1))
    result = {"method": "Embeddings: XGBoost on all-MiniLM-L6-v2 profile embeddings",
              "founders": int(X.shape[0]), "successes": int(y.sum()), "dimensions": int(X.shape[1]),
              "pooled": pooled, "per_repeat": per_rep}
    out = ROOT / "results" / "embeddings.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"Embeddings  P {pooled['precision']:.3f} R {pooled['recall']:.3f} F0.5 {pooled['f05']:.3f} "
          f"± {pooled['f05_sd']:.3f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
