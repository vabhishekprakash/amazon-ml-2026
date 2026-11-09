"""Metric registry. Add the official metric here on Day 1 if it's not already present.
Every metric: fn(y_true, y_pred) -> float. LOWER_IS_BETTER says which direction wins.
"""
import numpy as np


def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1e-9, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmsle(y_true, y_pred):
    y_true = np.clip(np.asarray(y_true, float), 0, None)
    y_pred = np.clip(np.asarray(y_pred, float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def f1_macro(y_true, y_pred):
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro"))


def f1_micro(y_true, y_pred):
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="micro"))


def _id_set(ids):
    """Parse an id list into a set of strings.

    Accepts a comma-separated string (the on-disk format), any iterable of ids,
    None, or a pandas NaN. Blank entries are dropped.
    """
    if ids is None:
        return set()
    if isinstance(ids, float) and np.isnan(ids):
        return set()
    if isinstance(ids, str):
        return {tok.strip() for tok in ids.split(",") if tok.strip()}
    return {str(tok).strip() for tok in ids if str(tok).strip()}


def f05_entity(true_ids, pred_ids):
    """F0.5 for one Source 1 entity: 1.25*P*R / (0.25*P + R).

    Precision and recall are computed over the predicted and true id sets for
    that entity. Predicting nothing for an entity with no true matches scores
    1.0; every other case with no overlap scores 0.0.
    """
    truth, pred = _id_set(true_ids), _id_set(pred_ids)
    if not truth and not pred:
        return 1.0
    if not truth or not pred:
        return 0.0
    hits = len(truth & pred)
    if hits == 0:
        return 0.0
    precision, recall = hits / len(pred), hits / len(truth)
    return float(1.25 * precision * recall / (0.25 * precision + recall))


def macro_f05(true_map, pred_map, entity_ids=None):
    """Mean of f05_entity over Source 1 entities.

    true_map and pred_map map a Source 1 entity id to its id list. Scoring runs
    over entity_ids when given, otherwise over the keys of true_map; an entity
    missing from pred_map counts as an empty prediction.
    """
    keys = list(true_map) if entity_ids is None else list(entity_ids)
    if not keys:
        raise ValueError("no entities to score")
    return float(np.mean([f05_entity(true_map.get(k), pred_map.get(k)) for k in keys]))


REGISTRY = {
    "smape": smape,
    "rmse": rmse,
    "mae": mae,
    "rmsle": rmsle,
    "accuracy": accuracy,
    "f1_macro": f1_macro,
    "f1_micro": f1_micro,
}

LOWER_IS_BETTER = {
    "smape": True, "rmse": True, "mae": True, "rmsle": True,
    "accuracy": False, "f1_macro": False, "f1_micro": False,
}


def get_metric(name):
    if name not in REGISTRY:
        raise KeyError(f"Unknown metric '{name}'. Known: {list(REGISTRY)}")
    return REGISTRY[name]
