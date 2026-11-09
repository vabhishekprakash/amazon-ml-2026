"""Cross-validation harness.
run_cv trains a model per fold, returns OOF predictions + test predictions
(averaged across folds) + per-fold scores. Works for regression & classification.
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.metrics import get_metric
from src.data import inverse_target


def run_cv(make_model, X, y, X_test=None, groups_for_stratify=None, n_splits=None, seed=None,
           fit_kwargs=None, predict_fn=None):
    """
    make_model: callable () -> fresh model with .fit/.predict (predict_proba used for clf if available)
    X, X_test: anything sliceable by integer index array (np array / scipy sparse / pd DataFrame via .iloc handled)
    Returns dict: oof, test_pred, fold_scores, mean, std
    """
    n_splits = n_splits or C.N_SPLITS
    seed = seed or C.SEED
    fit_kwargs = fit_kwargs or {}
    metric = get_metric(C.METRIC)

    import pandas as pd
    def take(A, idx):
        if A is None:
            return None
        if isinstance(A, pd.DataFrame) or isinstance(A, pd.Series):
            return A.iloc[idx]
        return A[idx]

    y = np.asarray(y)
    if C.TASK == "classification":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        strat = y if groups_for_stratify is None else groups_for_stratify
        splits = splitter.split(np.zeros(len(y)), strat)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(np.zeros(len(y)))

    oof = np.zeros(len(y), dtype=float) if C.TASK == "regression" else np.empty(len(y), dtype=object)
    test_preds = []
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(splits):
        t0 = time.time()
        model = make_model()
        model.fit(take(X, tr_idx), y[tr_idx], **fit_kwargs)

        if predict_fn is not None:
            va_pred = predict_fn(model, take(X, va_idx))
        else:
            va_pred = model.predict(take(X, va_idx))
        oof[va_idx] = va_pred

        # score in ORIGINAL target space
        score = metric(inverse_target(y[va_idx]), inverse_target(np.asarray(va_pred)))
        fold_scores.append(score)

        if X_test is not None:
            tp = predict_fn(model, X_test) if predict_fn is not None else model.predict(X_test)
            test_preds.append(np.asarray(tp))
        print(f"fold {fold}: {C.METRIC}={score:.5f}  ({time.time()-t0:.1f}s)", flush=True)

    result = {
        "oof": oof,
        "fold_scores": fold_scores,
        "mean": float(np.mean(fold_scores)),
        "std": float(np.std(fold_scores)),
        "test_pred": None,
    }
    if test_preds:
        if C.TASK == "regression":
            result["test_pred"] = np.mean(test_preds, axis=0)
        else:
            # majority vote across folds (string-safe)
            import pandas as pd
            stacked = pd.DataFrame(np.vstack([p.astype(str) for p in test_preds]))
            result["test_pred"] = stacked.mode(axis=0).iloc[0].values
    print(f"CV {C.METRIC}: {result['mean']:.5f} ± {result['std']:.5f}")
    return result
