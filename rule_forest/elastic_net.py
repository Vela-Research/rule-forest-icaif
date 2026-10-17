"""The learned-weight baseline. Elastic-net logistic regression on the same YES/NO answers.

Two configurations, both fitted on training founders only:

table2  The paper's Table 2 and Table 3 model. LogisticRegressionCV with the elastic-net penalty
        (saga), l1_ratio in {0.1, 0.5, 0.7, 0.9, 0.95, 1.0} and 10 values of C, chosen by
        5-fold ROC AUC on the training founders. The decision threshold is the value in
        {0.05, ..., 0.95} (50 steps) that maximises F0.5 on the training founders.
sweep   The faster model used for the Figure 4 and Figure 5 sweeps, where the vote also uses 5
        inner folds. l1_ratio 0.5, and C in {0.001, 0.01, 0.1, 1, 10} chosen by mean F0.5 over 5
        inner folds (each refitting the model and its threshold on the inner training part). The
        final threshold is the training probability that maximises F0.5.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold

from .metrics import best_cut, prf
from .vote import outer_folds, record

L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 0.95, 1.0]
THRESHOLDS = np.linspace(0.05, 0.95, 50)
SWEEP_CS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def proba(model, X):
    """P(success) computed once per distinct answer pattern, so examples with identical answers
    get exactly the same probability whatever the array layout (a threshold placed at one
    example's probability then never splits a group of identical examples)."""
    patterns, inverse = np.unique(X, axis=0, return_inverse=True)
    return model.predict_proba(patterns)[:, 1][inverse.ravel()]


def _quiet():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)


def table2_fit(Xtr, ytr):
    """Fit the Table 2 model. Returns (model, threshold)."""
    _quiet()
    m = LogisticRegressionCV(cv=5, penalty="elasticnet", solver="saga", l1_ratios=L1_RATIOS,
                             max_iter=5000, scoring="roc_auc", n_jobs=1, random_state=42).fit(Xtr, ytr)
    p = proba(m, Xtr)
    best_t, best_s = 0.5, 0.0
    for t in THRESHOLDS:
        s = fbeta_score(ytr, (p >= t).astype(int), beta=0.5, zero_division=0)
        if s > best_s:
            best_s, best_t = s, t
    return m, best_t


def _sweep_model(C):
    return LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=C, solver="saga", max_iter=5000, random_state=0)


def sweep_fit(Xtr, ytr, inner_splits=5):
    """Fit the sweep model: C by inner-fold F0.5, then refit. Returns (model, threshold, C)."""
    _quiet()
    k = max(2, min(inner_splits, int(min((ytr == 1).sum(), (ytr == 0).sum()))))
    skf = StratifiedKFold(k, shuffle=True, random_state=0)
    best, best_C = -1.0, SWEEP_CS[0]
    for C in SWEEP_CS:
        s = []
        for itr, iva in skf.split(Xtr, ytr):
            if ytr[itr].sum() < 1 or ytr[iva].sum() < 1:
                continue
            m = _sweep_model(C).fit(Xtr[itr], ytr[itr])
            t = best_cut(ytr[itr], proba(m, Xtr[itr]))
            yp = proba(m, Xtr[iva]) >= t
            s.append(prf(int((yp & (ytr[iva] == 1)).sum()), int((yp & (ytr[iva] == 0)).sum()),
                         int((~yp & (ytr[iva] == 1)).sum()))[2])
        mean = float(np.mean(s)) if s else -1.0
        if mean > best:
            best, best_C = mean, C
    m = _sweep_model(best_C).fit(Xtr, ytr)
    return m, best_cut(ytr, proba(m, Xtr)), best_C


def nested_cv(X, y, model="table2", n_repeats=10, n_splits=10, inner_splits=5, pools=None, folds=None):
    """Outer repeated K-fold for the elastic net on answers X (Q, F), the vote's orientation.
    `pools` maps (repeat, fold) to the question rows the model may use in that fold."""
    Xf = X.T.astype(float)
    out = []
    for rep, k, tr, te in (folds or outer_folds(y, n_repeats, n_splits)):
        cols = np.arange(X.shape[0]) if pools is None else np.asarray(pools[(rep, k)])
        Xtr, Xte = Xf[tr][:, cols], Xf[te][:, cols]
        if model == "table2":
            m, t = table2_fit(Xtr, y[tr])
            C = None
        else:
            m, t, C = sweep_fit(Xtr, y[tr], inner_splits)
        p_tr, p_te = proba(m, Xtr), proba(m, Xte)
        coef = np.zeros(X.shape[0])
        coef[cols] = m.coef_.ravel()
        hit = p_tr >= t
        yt = y[tr]
        out.append(record(rep, k, te, (p_te >= t).astype(np.int8), y[te], prob=p_te, threshold=float(t),
                          coef=coef, C=C, train_f05=prf(int((hit & (yt == 1)).sum()), int((hit & (yt == 0)).sum()),
                                                        int((~hit & (yt == 1)).sum()))[2]))
    return out
