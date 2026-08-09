"""Engineered-ML baseline for the founder eval set, end to end and fully offline.

Re-applies the engineered-features method to THIS dataset: ~30 deterministic prose features
(build_features.py) -> StandardScaler -> elastic-net logistic regression, evaluated under the
paper's protocol (10x10 repeated nested CV, class-balanced, per-fold F0.5-optimal threshold,
per-repeat-pooled F0.5). Reproducible from the shipped anonymised founder prose alone -- no
LLM, no network, no VCBench, no non-public data.

  python run_engineered_ml.py            # -> engineered_ml_result.json + engineered_features.csv
"""
import os, warnings
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
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

REL = HERE.parents[2]
EVAL = REL / "rrf_submission" / "precomputed" / "03_full_cross_validation_test_data_anonymised.csv"
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
    return {"repeat": rep, "te": te.tolist(), "preds": yp.tolist(),
            "TP": int(((yp == 1) & (yte == 1)).sum()),
            "FP": int(((yp == 1) & (yte == 0)).sum()),
            "FN": int(((yp == 0) & (yte == 1)).sum())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nproc", type=int, default=14)
    args = ap.parse_args()
    d = pd.read_csv(EVAL)
    Xdf = build_features(d["summary"])
    y = d["startup_success"].astype(int).to_numpy()
    # drop zero-variance columns (e.g. constant role_founder) before scaling
    keep = [c for c in Xdf.columns if Xdf[c].nunique() > 1]
    dropped = [c for c in Xdf.columns if c not in keep]
    Xdf = Xdf[keep]
    X = Xdf.to_numpy(float)
    print(f"engineered_ml: {X.shape[0]} founders, {X.shape[1]} features "
          f"(dropped zero-variance: {dropped}), {int(y.sum())} positives", flush=True)

    # save the anonymised feature matrix for inspection / reuse
    out = pd.concat([d[["founder_uuid"]].reset_index(drop=True), Xdf.reset_index(drop=True),
                     pd.Series(y, name="startup_success")], axis=1)
    out.to_csv(HERE / "engineered_features.csv", index=False)

    tasks = []
    for rep in range(1, N_REPEATS + 1):
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=42 + (rep - 1))
        for tr, te in skf.split(X, y):
            tasks.append((rep, tr, te))
    with Pool(min(args.nproc, len(tasks)), initializer=_init, initargs=(X, y)) as pool:
        rows = pool.map(_fold, tasks)

    # per-founder 10-repeat OOF predictions -> consensus (majority vote), for the paired bootstrap
    predmat = np.full((N_REPEATS, X.shape[0]), -1, dtype=int)
    for r in rows:
        predmat[r["repeat"] - 1, np.asarray(r["te"], int)] = np.asarray(r["preds"], int)
    vote_freq = predmat.mean(0)
    consensus = (vote_freq >= 0.5).astype(int)
    cons_df = pd.DataFrame({"founder_uuid": d["founder_uuid"].values,
                            "hash": d["founder_uuid"].astype(str).str[:5].values,
                            "y": y, "vote_freq": vote_freq, "consensus": consensus})
    cons_df.to_csv(HERE / "engineered_ml_consensus.csv", index=False)

    per_rep = []
    for rep in range(1, N_REPEATS + 1):
        g = [r for r in rows if r["repeat"] == rep]
        tp = sum(r["TP"] for r in g); fp = sum(r["FP"] for r in g); fn = sum(r["FN"] for r in g)
        f, p, r = f05c(tp, fp, fn)
        per_rep.append({"repeat": rep, "f05": f, "precision": p, "recall": r})
    f05 = np.array([r["f05"] for r in per_rep])
    prec = np.array([r["precision"] for r in per_rep]); rec = np.array([r["recall"] for r in per_rep])
    result = {"method": "Engineered ML (elastic-net on deterministic prose features)",
              "n_founders": int(X.shape[0]), "n_features": int(X.shape[1]), "features": keep,
              "n_positives": int(y.sum()), "base_rate": float(y.mean()),
              "protocol": "10x10 repeated nested CV, class-balanced elastic-net, per-fold "
                          "F0.5-optimal threshold, per-repeat-pooled F0.5",
              "f05_mean": float(f05.mean()), "f05_std": float(f05.std(ddof=1)),
              "precision_mean": float(prec.mean()), "recall_mean": float(rec.mean()),
              "per_repeat": per_rep}
    (HERE / "engineered_ml_result.json").write_text(json.dumps(result, indent=2))
    ctp = int(((consensus == 1) & (y == 1)).sum()); cfp = int(((consensus == 1) & (y == 0)).sum())
    cfn = int(((consensus == 0) & (y == 1)).sum())
    cf, cp, cr = f05c(ctp, cfp, cfn)
    print(f"\nEngineered ML  F0.5 = {f05.mean():.4f} +/- {f05.std(ddof=1):.4f}  "
          f"(P={prec.mean():.4f} R={rec.mean():.4f})   [per-repeat-pooled, Table-2 format]", flush=True)
    print(f"  consensus (majority-vote) F0.5 = {cf:.4f} (P={cp:.4f} R={cr:.4f}) pred_pos={int(consensus.sum())}"
          f"   [for the paired bootstrap]", flush=True)
    print(f"wrote engineered_ml_result.json + engineered_features.csv + engineered_ml_consensus.csv", flush=True)


if __name__ == "__main__":
    main()
