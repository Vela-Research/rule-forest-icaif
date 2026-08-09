"""Core paired unit-vote-vs-elastic-net F0.5 evaluation + subsampling for the count-effect
figures. Release-self-contained port of the experiment-side sweep_core.

Reuses the shipped Rule Forest reference implementation:
  - unit-vote (RRF)  = rank questions by in-fold F0.5, keep top-N, predict positive if >=T
    fire; (N,T) tuned by nested CV for F0.5 -- via the shipped scoring_utils building blocks
    (select_best_hyperparams / compute_weights / evaluate_test_fold), identical to
    baselines/leakage_free_ablation/run_ablation.run_nested("perfold").
  - elastic-net (EN) = elastic-net logistic regression on the same matrix, nested-CV tuned C,
    in-fold F0.5 threshold.
Both arms share identical outer StratifiedKFold seeds (paired) and the same per-repeat-pooled
F0.5 aggregation (run_ablation.agg_f05).

INNER MODEL-SELECTION CV: the *outer* protocol is the pre-registered 10x10 repeated nested CV
(same seeds 42+repeat-1, same per-fold question re-ranking as run_nested). The inner
model-selection CV is INNER_SPLITS=5 folds, applied uniformly to BOTH arms (the unit-vote's
(N,T) selection AND the elastic-net's C selection). inner folds != outer folds is standard
nested CV; applied identically to every point so gap comparisons stay fair.

DETERMINISM: subsample seeds come from det_seed() (sha256-based) -- reproducible with no
PYTHONHASHSEED dependency. The EN primitives (_en/sweep_thr/f05c) are inlined here so
count_effect/ needs nothing from baselines/prevalence_sweep/.
"""
from __future__ import annotations
import sys, hashlib
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

# --- wire up the shipped reference implementation (release-relative) -------------------------
RELEASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RELEASE / "baselines" / "leakage_free_ablation"))
sys.path.insert(0, str(RELEASE / "rrf_submission"))
import run_ablation as _ra                                     # noqa: E402
from run_ablation import agg_f05, q_f05                        # noqa: E402
from src.utils.scoring_utils import (                          # noqa: E402
    select_best_hyperparams, compute_weights, evaluate_test_fold, f_beta_score,
)

INNER_SPLITS = 5   # inner selection folds; UNIFORM across BOTH arms (unit-vote & EN)


# --- inlined EN primitives (verbatim from the reference fig_a_rrf_weight_sweep) --------------
def f05c(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 0.0 if p + r == 0 else 1.25 * p * r / (0.25 * p + r)


def sweep_thr(y, scores):
    """F0.5-optimal cut over candidate score values (in-fold, on train)."""
    best_thr, best = float(scores.min()) - 1.0, -1.0
    for t in np.unique(scores):
        yp = (scores >= t).astype(int)
        tp = int(((yp == 1) & (y == 1)).sum()); fp = int(((yp == 1) & (y == 0)).sum())
        fn = int(((yp == 0) & (y == 1)).sum())
        s = f05c(tp, fp, fn)
        if s > best:
            best, best_thr = s, float(t)
    return best_thr


def _en(C):
    return LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=C, solver="saga",
                              max_iter=5000, random_state=0)


def det_seed(*parts):
    """Deterministic subsample seed (sha256 of the arguments). Reproducible with no
    PYTHONHASHSEED dependency, unlike the experiment-side builtin hash()."""
    s = "|".join(str(p) for p in parts)
    return (int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**31 - 1)) or 1


def tune_C_inner(Xtr, ytr, inner_splits=INNER_SPLITS, Cs=(1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    inner = max(2, min(inner_splits, int(min((ytr == 1).sum(), (ytr == 0).sum()))))
    skf = StratifiedKFold(inner, shuffle=True, random_state=0)
    best, bestC = -1.0, Cs[0]
    for C in Cs:
        sc = []
        for itr, iva in skf.split(Xtr, ytr):
            if ytr[itr].sum() < 1 or ytr[iva].sum() < 1:
                continue
            clf = _en(C).fit(Xtr[itr], ytr[itr])
            thr = sweep_thr(ytr[itr], clf.predict_proba(Xtr[itr])[:, 1])
            yp = (clf.predict_proba(Xtr[iva])[:, 1] >= thr).astype(int)
            tp = int(((yp == 1) & (ytr[iva] == 1)).sum()); fp = int(((yp == 1) & (ytr[iva] == 0)).sum())
            fn = int(((yp == 0) & (ytr[iva] == 1)).sum())
            sc.append(f05c(tp, fp, fn))
        m = float(np.mean(sc)) if sc else -1.0
        if m > best:
            best, bestC = m, C
    return bestC


def run_en_inner(Xf, y, n_rep, n_splits, inner_splits=INNER_SPLITS):
    """EN outer loop; inner C-selection CV decoupled to `inner_splits`. Same outer
    StratifiedKFold seeding (42+rep-1) as the unit-vote arm -> paired. Xf is (instances, questions)."""
    per = []
    for rep in range(1, n_rep + 1):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=42 + (rep - 1))
        for fold, (tr, te) in enumerate(skf.split(Xf, y), 1):
            Xtr, ytr, Xte, yte = Xf[tr], y[tr], Xf[te], y[te]
            clf = _en(tune_C_inner(Xtr, ytr, inner_splits)).fit(Xtr, ytr)
            thr = sweep_thr(ytr, clf.predict_proba(Xtr)[:, 1])
            yp = (clf.predict_proba(Xte)[:, 1] >= thr).astype(int)
            per.append({"repeat": rep, "fold": fold,
                        "TP": int(((yp == 1) & (yte == 1)).sum()),
                        "FP": int(((yp == 1) & (yte == 0)).sum()),
                        "FN": int(((yp == 0) & (yte == 1)).sum())})
    return per


def run_nested_inner(X, y, n_repeats, n_splits, inner_splits=INNER_SPLITS):
    """run_ablation.run_nested(X, y, 'perfold') OUTER loop, inner (N,T) CV decoupled to
    `inner_splits`. Reuses the shipped select_best_hyperparams / compute_weights /
    evaluate_test_fold and the identical outer StratifiedKFold seeding (42 + repeat-1)."""
    Q, F = X.shape
    fidx = np.arange(F)
    score_thresholds = list(range(1, Q + 1))
    inner = max(2, min(inner_splits, int(min((y == 1).sum(), (y == 0).sum()))))
    per_fold = []
    for repeat in range(1, n_repeats + 1):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=42 + (repeat - 1))
        for outer_idx, (tr, te) in enumerate(skf.split(fidx, y), start=1):
            train_ids, test_ids = fidx[tr], fidx[te]
            y_train = y[train_ids]
            order = np.argsort([q_f05(y_train, X[q, train_ids]) for q in range(Q)])[::-1]
            Xo = X[order, :]
            best_combo, prec_array, ratio_array, _ = select_best_hyperparams(
                Xo[:, train_ids], y_train, train_ids, 0, [0.0], score_thresholds,
                Q, y, inner, outer_idx + repeat, optimise_for="f0.5",
            )
            exp_opt, nq_opt, t_opt = best_combo
            W_full, _ = compute_weights(0, exp_opt, prec_array, ratio_array)
            preds, y_te, tp, fp, tn, fn, *_ = evaluate_test_fold(Xo, W_full, nq_opt, t_opt, test_ids, y)
            per_fold.append({"repeat": repeat, "fold": outer_idx, "n_q": nq_opt, "thr": t_opt,
                             "TP": tp, "FP": fp, "FN": fn,
                             "f05": f_beta_score(tp / (tp + fp) if tp + fp else 0.0,
                                                 tp / (tp + fn) if tp + fn else 0.0)})
    return per_fold


def paired_eval(X, y, n_repeats=10, n_splits=10):
    """Unit-vote and EN on the SAME nested CV over matrix X (Q,F), labels y (F,)."""
    Q, F = X.shape
    n_pos = int(y.sum()); n_neg = int((y == 0).sum())
    ns = max(2, int(min(n_splits, min(n_pos, n_neg))))
    rrf = agg_f05(run_nested_inner(X, y, n_repeats, ns))
    en = agg_f05(run_en_inner(X.T, y, n_repeats, ns))
    return {"rrf_f05": float(rrf), "en_f05": float(en), "gap": float(rrf - en),
            "n": int(F), "n_pos": n_pos, "prevalence": float(y.mean()),
            "n_repeats": n_repeats, "n_splits": ns, "inner_splits": INNER_SPLITS}


# --- subsamplers.  X is (Q, F); columns are instances. --------------------------------------
def sub_fixed_prev(X, y, n_pos, prev, seed):
    """Keep n_pos positives, add negatives to hold prevalence `prev`."""
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    tp = rng.choice(pos, size=min(n_pos, len(pos)), replace=False)
    n_neg = min(int(round(len(tp) * (1 - prev) / prev)), len(neg))
    keep = np.sort(np.concatenate([tp, rng.choice(neg, size=n_neg, replace=False)]))
    return X[:, keep], y[keep]


def sub_all_neg(X, y, n_pos, seed):
    """Keep ALL negatives, subsample positives to n_pos (the Kiva-D2 design)."""
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    tp = rng.choice(pos, size=min(n_pos, len(pos)), replace=False)
    keep = np.sort(np.concatenate([tp, neg]))
    return X[:, keep], y[keep]
