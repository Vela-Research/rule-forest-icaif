import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, fbeta_score, matthews_corrcoef

def f_beta_score(p, r, beta=0.5):
    b2 = beta * beta
    return 0.0 if (p + r) == 0 else (1 + b2) * p * r / (b2 * p + r)

def compute_weighted_cumulative_scores(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Computes cumulative weighted scores over questions for each founder.

    Args:
        X (np.ndarray): Binary prediction matrix of shape (Q, F)
        W (np.ndarray): Weight vector of shape (Q,)

    Returns:
        np.ndarray: Cumulative score matrix of shape (Q, F)
    """
    Q, F = X.shape
    S = np.zeros((Q, F))
    running = np.zeros(F)

    for q in range(Q):
        running += W[q] * X[q, :]
        S[q, :] = running.copy()

    return S

def compute_weights(weighting, exponent, prec_array, ratio_array):
    """
    Compute question weights based on precision scores and weighting method.

    Args:
        weighting (str): 'adaboost' or numeric string ('0', '1')
        exponent (float or str): exponent value or 'adaboost'
        prec_array (np.ndarray): precision scores per question
        ratio_array (np.ndarray): normalized precision ratios

    Returns:
        tuple: (weight array W, label)
    """
    if weighting == 'adaboost':
        epsilon_q = 1 - prec_array
        epsilon_q = np.clip(epsilon_q, 1e-6, 0.999)
        W = 0.5 * np.log((1 - epsilon_q) / epsilon_q)
        W = np.clip(W, -2, 2)
        W = W - W.mean() + 1
        label = 'adaboost'
    else:
        W = ratio_array ** exponent
        W = W / W.mean()
        label = exponent
    return W, label

def select_best_hyperparams(
    X_train, y_train, train_ids, weighting, exponents,
    score_thresholds, Q, success_values, n_splits, outer_seed,
    optimise_for="f0.5"  # Options: "precision", "f0.5", "f1", "f2"
):
    prec_array = np.zeros(Q, dtype=float)
    for qi in range(Q):
        preds_q = X_train[qi, :]
        tp = ((preds_q == 1) & (y_train == 1)).sum()
        fp = ((preds_q == 1) & (y_train == 0)).sum()
        prec_array[qi] = tp / (tp + fp) if tp + fp > 0 else 0.0

    mean_prec_train = prec_array.mean()
    ratio_array = (prec_array / mean_prec_train) if mean_prec_train > 0 else np.ones(Q)

    best_combo = None
    best_score = -1.0
    inner_skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=outer_seed)
    weighting_options = ['adaboost'] if weighting == 'adaboost' else exponents

    for w in weighting_options:
        W, label = compute_weights(weighting, w, prec_array, ratio_array)
        scores_train = compute_weighted_cumulative_scores(X_train, W)

        for n_q in range(1, Q + 1):
            for t in score_thresholds:
                fold_scores = []

                for _, (_, iv) in enumerate(inner_skf.split(train_ids, y_train)):
                    val_ids = train_ids[iv]
                    pos = np.where(np.isin(train_ids, val_ids))[0]
                    preds = (scores_train[n_q - 1, pos] >= t).astype(int)
                    yv = success_values[val_ids]

                    tp_i = ((preds == 1) & (yv == 1)).sum()
                    fp_i = ((preds == 1) & (yv == 0)).sum()
                    fn_i = ((preds == 0) & (yv == 1)).sum()

                    p = tp_i / (tp_i + fp_i) if tp_i + fp_i > 0 else 0.0
                    r = tp_i / (tp_i + fn_i) if tp_i + fn_i > 0 else 0.0

                    if optimise_for == "precision":
                        fold_scores.append(p)
                    elif optimise_for == "f1":
                        fold_scores.append(fbeta_score(yv, preds, beta=1.0, zero_division=0))
                    elif optimise_for == "f2":
                        fold_scores.append(fbeta_score(yv, preds, beta=2.0, zero_division=0))
                    elif optimise_for == "f0.5":
                        fold_scores.append(f_beta_score(p, r))  # custom function
                    else:
                        raise ValueError(f"Unknown optimise_for metric: {optimise_for}")

                mean_score = np.mean(fold_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_combo = (label, n_q, t)

    return best_combo, prec_array, ratio_array, best_score

def evaluate_test_fold(X, W, n_q, threshold, test_ids, success_values):
    scores_full = compute_weighted_cumulative_scores(X, W)
    preds_test = (scores_full[n_q - 1, test_ids] >= threshold).astype(int)
    y_test = success_values[test_ids]

    tp = ((preds_test == 1) & (y_test == 1)).sum()
    fp = ((preds_test == 1) & (y_test == 0)).sum()
    tn = ((preds_test == 0) & (y_test == 0)).sum()
    fn = ((preds_test == 0) & (y_test == 1)).sum()
    p_out = tp / (tp + fp) if tp + fp > 0 else 0.0
    r_out = tp / (tp + fn) if tp + fn > 0 else 0.0
    f05_out = f_beta_score(p_out, r_out)
    f05_out_sk = fbeta_score(y_test, preds_test, beta=0.5, zero_division=0)
    f1_out = f1_score(y_test, preds_test, zero_division=0)
    f2_out = fbeta_score(y_test, preds_test, beta=2.0, zero_division=0)
    mcc_out = matthews_corrcoef(y_test, preds_test)    

    return preds_test, y_test, tp, fp, tn, fn, p_out, r_out, f05_out, f05_out_sk, f1_out, f2_out, mcc_out