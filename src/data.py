"""Data loading helpers. Robust to big files: downcasts dtypes, optional row sampling."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def load_train(nrows=None) -> pd.DataFrame:
    df = pd.read_csv(C.TRAIN_FILE, nrows=nrows)
    return _downcast(df)


def load_test(nrows=None) -> pd.DataFrame:
    df = pd.read_csv(C.TEST_FILE, nrows=nrows)
    return _downcast(df)


def load_sample_sub() -> pd.DataFrame:
    return pd.read_csv(C.SAMPLE_SUB_FILE)


def combined_text(df: pd.DataFrame) -> pd.Series:
    """Concatenate the configured text columns into one string per row."""
    cols = [c for c in C.TEXT_COLS if c in df.columns]
    if not cols:
        raise ValueError(f"None of TEXT_COLS {C.TEXT_COLS} found in columns {list(df.columns)[:20]}")
    out = df[cols[0]].fillna("").astype(str)
    for c in cols[1:]:
        out = out + " \n " + df[c].fillna("").astype(str)
    return out


def get_target(df: pd.DataFrame) -> np.ndarray:
    y = df[C.TARGET_COL].values
    if C.TASK == "regression":
        y = y.astype(float)
        if C.LOG_TARGET:
            y = np.log1p(np.clip(y, 0, None))
    return y


def inverse_target(y: np.ndarray) -> np.ndarray:
    """Undo target transform before scoring/submitting."""
    if C.TASK == "regression" and C.LOG_TARGET:
        return np.expm1(y)
    return y
