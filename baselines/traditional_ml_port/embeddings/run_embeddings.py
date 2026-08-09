"""Embeddings baseline for the founder eval set, end to end and offline: frozen MiniLM
sentence embeddings of the anonymised prose (build_embeddings.py) -> XGBoost, evaluated under
the paper's protocol (10x10 repeated nested CV, class-imbalance-weighted, per-fold F0.5-optimal
threshold, per-repeat-pooled F0.5). Reproducible from the shipped anonymised prose alone.

Needs xgboost (+ its OpenMP runtime libomp). On macOS, `brew install libomp`, or point
DYLD_LIBRARY_PATH at a bundled copy (e.g. torch/lib) as done at the top of this file.

  python run_embeddings.py       # -> embeddings_result.json (+ consensus)
"""
import os, warnings
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
# best-effort: make xgboost's libomp findable via a bundled copy (torch), if present
try:
    import torch as _t
    _lib = os.path.join(os.path.dirname(_t.__file__), "lib")
    os.environ["DYLD_LIBRARY_PATH"] = _lib + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")
except Exception:
    pass
import sys, json, argparse
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
REL = HERE.parents[2]
EVAL = REL / "rrf_submission" / "precomputed" / "03_full_cross_validation_test_data_anonymised.csv"
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
    return {"repeat": rep, "te": te.tolist(), "preds": yp.tolist(),
            "TP": int(((yp == 1) & (yte == 1)).sum()),
            "FP": int(((yp == 1) & (yte == 0)).sum()),
            "FN": int(((yp == 0) & (yte == 1)).sum())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nproc", type=int, default=14)
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

    predmat = np.full((N_REPEATS, X.shape[0]), -1, dtype=int)
    for r in rows:
        predmat[r["repeat"] - 1, np.asarray(r["te"], int)] = np.asarray(r["preds"], int)
    vote_freq = predmat.mean(0); consensus = (vote_freq >= 0.5).astype(int)
    pd.DataFrame({"founder_uuid": d["founder_uuid"].values,
                 "hash": d["founder_uuid"].astype(str).str[:5].values,
                 "y": y, "vote_freq": vote_freq, "consensus": consensus}
                ).to_csv(HERE / "embeddings_consensus.csv", index=False)

    per_rep = []
    for rep in range(1, N_REPEATS + 1):
        g = [r for r in rows if r["repeat"] == rep]
        tp = sum(r["TP"] for r in g); fp = sum(r["FP"] for r in g); fn = sum(r["FN"] for r in g)
        f, p, r = f05c(tp, fp, fn)
        per_rep.append({"repeat": rep, "f05": f, "precision": p, "recall": r})
    F = np.array([r["f05"] for r in per_rep]); P = np.array([r["precision"] for r in per_rep])
    R = np.array([r["recall"] for r in per_rep])
    result = {"method": "Embeddings (XGBoost on all-MiniLM-L6-v2 prose embeddings)",
              "n_founders": int(X.shape[0]), "emb_dim": int(X.shape[1]), "n_positives": int(y.sum()),
              "protocol": "10x10 repeated nested CV, imbalance-weighted XGBoost, per-fold "
                          "F0.5-optimal threshold, per-repeat-pooled F0.5",
              "f05_mean": float(F.mean()), "f05_std": float(F.std(ddof=1)),
              "precision_mean": float(P.mean()), "precision_std": float(P.std(ddof=1)),
              "recall_mean": float(R.mean()), "recall_std": float(R.std(ddof=1)),
              "per_repeat": per_rep}
    (HERE / "embeddings_result.json").write_text(json.dumps(result, indent=2))
    ctp = int(((consensus == 1) & (y == 1)).sum()); cfp = int(((consensus == 1) & (y == 0)).sum())
    cfn = int(((consensus == 0) & (y == 1)).sum()); cf, cp, cr = f05c(ctp, cfp, cfn)
    print(f"\nEmbeddings  F0.5 = {F.mean():.4f} +/- {F.std(ddof=1):.4f}  "
          f"(P={P.mean():.4f} R={R.mean():.4f})   [per-repeat-pooled, Table-2 format]", flush=True)
    print(f"  consensus F0.5 = {cf:.4f} (P={cp:.4f} R={cr:.4f}) pred_pos={int(consensus.sum())}", flush=True)
    print("wrote embeddings_result.json + embeddings_consensus.csv", flush=True)


if __name__ == "__main__":
    main()
