"""Reproduce the paper's Table 2 GPTree and RRM rows by re-scoring each method's frozen
LLM-generated artifacts under the same 10x10 nested cross-validation used for Rule Forest.
No LLM / API calls -- reuses the fitted tree (per-founder leaf assignment) and the mined rule
set (per-founder frozen score), and only re-fits the data-dependent part inside each fold:
  GPTree: leaf -> P(success) refit on each training fold; threshold tuned per fold.
  RRM   : per-founder score frozen; only the threshold is tuned per fold.
Protocol (identical to the RRF reference): StratifiedKFold(n=10, shuffle, random_state=42+rep-1),
10 repeats; F0.5 = 1.25*p*r/(0.25*p+r); Table reports mean +/- SD over the 100 folds, averaged
over the 3 generation seeds. Self-contained: reads only files under ./data.
  python rescore_gptree_rrm.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
HERE = Path(__file__).resolve().parent; DATA = HERE/"data"
SEEDS = [42, 43, 44]

def f05c(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0.0; r = tp/(tp+fn) if tp+fn else 0.0
    return 0.0 if p+r == 0 else 1.25*p*r/(0.25*p+r)
def pr(tp, fp, fn):
    return (tp/(tp+fp) if tp+fp else 0.0), (tp/(tp+fn) if tp+fn else 0.0)
def best_thr(sc, y, grid):
    best = (-1.0, grid[0])
    for t in grid:
        pred = (sc >= t).astype(int)
        tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
        s = f05c(tp,fp,fn)
        if s > best[0]: best = (s, t)
    return best[1]

def cv_perfold(score_of_fold, y, n_rep=10, n_spl=10):
    """Return per-fold precision, recall, F0.5 arrays (length n_rep*n_spl)."""
    P,R,F = [],[],[]
    for rep in range(1, n_rep+1):
        skf = StratifiedKFold(n_spl, shuffle=True, random_state=42+(rep-1))
        for tr, te in skf.split(np.zeros(len(y)), y):
            s_tr, s_te = score_of_fold(tr, te)
            thr = best_thr(s_tr, y[tr], np.unique(s_tr))
            pred = (s_te >= thr).astype(int)
            tp=int(((pred==1)&(y[te]==1)).sum()); fp=int(((pred==1)&(y[te]==0)).sum()); fn=int(((pred==0)&(y[te]==1)).sum())
            p,r = pr(tp,fp,fn); P.append(p); R.append(r); F.append(f05c(tp,fp,fn))
    return np.array(P), np.array(R), np.array(F)

def report(name, per_seed):  # per_seed: list of (P,R,F) arrays
    P = np.concatenate([s[0] for s in per_seed]); R = np.concatenate([s[1] for s in per_seed]); F = np.concatenate([s[2] for s in per_seed])
    print(f"  {name:7s}  P={P.mean():.3f}+/-{P.std(ddof=1):.3f}   R={R.mean():.3f}+/-{R.std(ddof=1):.3f}   F0.5={F.mean():.3f}+/-{F.std(ddof=1):.3f}")

y_rrm = np.array(json.loads((DATA/"eval_labels.json").read_text()))
print(f"eval set: n={len(y_rrm)} pos={int(y_rrm.sum())} rate={y_rrm.mean():.4f}\n(10x10 nested CV re-scoring, mean +/- SD over 100 folds, pooled across 3 seeds)\n")

# RRM: frozen score, threshold refit per fold
rrm = []
for s in SEEDS:
    sc = json.loads((DATA/f"rrm/rrm_seed{s}_scores.json").read_text())
    scores = np.array([sc[str(i)] for i in range(len(y_rrm))], float)
    rrm.append(cv_perfold(lambda tr,te: (scores[tr], scores[te]), y_rrm))
report("RRM", rrm)

# GPTree: leaf prob refit per fold, threshold refit per fold
gp = []
for s in SEEDS:
    df = pd.read_csv(DATA/f"gptree/gptree_seed{s}_leaves.csv")
    leaf = df["leaf_node_id"].values
    yg = (df["true_label"].astype(str).str.strip().str.lower()=="successful").astype(int).values
    def sof(tr, te, leaf=leaf, yg=yg):
        d = pd.DataFrame({"leaf": leaf[tr], "y": yg[tr]}); lp = d.groupby("leaf")["y"].mean(); base = yg[tr].mean()
        return np.array([lp.get(l, base) for l in leaf[tr]]), np.array([lp.get(l, base) for l in leaf[te]])
    gp.append(cv_perfold(sof, yg))
report("GPTree", gp)
print("\nTable 2 rows (P / R / F0.5):  GPTree 0.066 / 0.095 / 0.068    RRM 0.079 / 0.102 / 0.080")
