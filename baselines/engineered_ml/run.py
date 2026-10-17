"""Engineered ML row of Table 2. No LLM and no network.

About 30 deterministic features are parsed from each anonymised founder summary with keyword and
pattern rules (build_features.py). Features that do not vary on the 497 question-generation
founders are dropped, so the evaluation labels play no part in choosing them. The features are
standardised and fed to a class-balanced elastic-net logistic regression, evaluated under the
paper's 10 x 10 cross-validation (outer seeds 42 + repeat - 1): C is chosen by 5 inner folds and
the threshold maximises training F0.5. Scores are pooled within each repeat.

The feature rules were written by hand with the summaries' vocabulary in view, so this baseline
may be mildly optimistic.

  python baselines/engineered_ml/run.py   # -> results/engineered_ml.json
"""
import os, warnings
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, argparse
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_features import build_features

ROOT = HERE.parents[1]
EVAL = ROOT / "data" / "founders" / "profiles" / "evaluation.csv"
GEN = ROOT / "data" / "founders" / "profiles" / "generation.csv"
BETA2 = 0.25
N_REPEATS, N_SPLITS, INNER = 10, 10, 5
CS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


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


def _en(C):
    return LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=C, solver="saga",
                              max_iter=5000, tol=1e-3, class_weight="balanced", random_state=0)


def tune_C(Xtr, ytr):
    inner = max(2, min(INNER, int(min((ytr == 1).sum(), (ytr == 0).sum()))))
    skf = StratifiedKFold(inner, shuffle=True, random_state=0)
    best, bestC = -1.0, CS[0]
    for C in CS:
        sc = []
        for itr, iva in skf.split(Xtr, ytr):
            if ytr[itr].sum() < 1 or ytr[iva].sum() < 1:
                continue
            clf = _en(C).fit(Xtr[itr], ytr[itr])
            thr = sweep_thr(ytr[itr], clf.predict_proba(Xtr[itr])[:, 1])
            yp = (clf.predict_proba(Xtr[iva])[:, 1] >= thr).astype(int)
            tp = int(((yp == 1) & (ytr[iva] == 1)).sum()); fp = int(((yp == 1) & (ytr[iva] == 0)).sum())
            fn = int(((yp == 0) & (ytr[iva] == 1)).sum())
            sc.append(f05c(tp, fp, fn)[0])
        m = float(np.mean(sc)) if sc else -1.0
        if m > best:
            best, bestC = m, C
    return bestC


_X = _Y = None
def _init(X, y):
    global _X, _Y
    _X, _Y = X, y


def _fold(task):
    rep, tr, te = task
    Xtr, ytr, Xte, yte = _X[tr], _Y[tr], _X[te], _Y[te]
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    clf = _en(tune_C(Xtr, ytr)).fit(Xtr, ytr)
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
    Xdf = build_features(d["summary"])
    y = d["startup_success"].astype(int).to_numpy()
    # keep the features that vary on the generation founders (never the evaluation labels)
    gen = pd.read_csv(GEN)
    Xgen = build_features(gen["summary"])
    keep = [c for c in Xgen.columns if Xgen[c].nunique() > 1]
    dropped = [c for c in Xgen.columns if c not in keep]
    Xdf = Xdf[keep]
    X = Xdf.to_numpy(float)
    print(f"engineered_ml: {X.shape[0]} eval founders, {X.shape[1]} features "
          f"(selected on the 497 generation founders; dropped {dropped}), "
          f"{int(y.sum())} positives", flush=True)

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
    result = {"method": "Engineered ML: class-balanced elastic net on deterministic prose features",
              "founders": int(X.shape[0]), "successes": int(y.sum()), "features": keep,
              "features_dropped_as_constant_on_generation_founders": dropped,
              "pooled": pooled, "per_repeat": per_rep}
    out = ROOT / "results" / "engineered_ml.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"Engineered ML  P {pooled['precision']:.3f} R {pooled['recall']:.3f} F0.5 {pooled['f05']:.3f} "
          f"± {pooled['f05_sd']:.3f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
