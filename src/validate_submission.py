"""Validate a submission file against sample_submission BEFORE uploading.
Run: python src/validate_submission.py outputs/preds/submission_text.csv
Exit code 0 = safe to upload.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

def main(path):
    sub, sample = pd.read_csv(path), pd.read_csv(C.SAMPLE_SUB_FILE)
    errs, warns = [], []
    if list(sub.columns) != list(sample.columns):
        errs.append(f"columns {list(sub.columns)} != expected {list(sample.columns)}")
    if len(sub) != len(sample):
        errs.append(f"row count {len(sub)} != expected {len(sample)}")
    if sub.isna().any().any():
        errs.append(f"NaNs present: {sub.isna().sum().to_dict()}")
    if C.ID_COL in sub.columns and C.ID_COL in sample.columns:
        if not sub[C.ID_COL].astype(str).equals(sample[C.ID_COL].astype(str)):
            if set(sub[C.ID_COL].astype(str)) == set(sample[C.ID_COL].astype(str)):
                warns.append("ids match as a set but ORDER differs — reorder to sample order unless rules say otherwise")
            else:
                errs.append("id values differ from sample_submission")
        if sub[C.ID_COL].duplicated().any():
            errs.append("duplicate ids")
    tcols = [c for c in sample.columns if c != C.ID_COL]
    for c in tcols:
        if c in sub.columns and pd.api.types.is_numeric_dtype(sample[c]):
            if not pd.api.types.is_numeric_dtype(sub[c]):
                errs.append(f"column '{c}' should be numeric")
            elif C.TASK == "regression" and (sub[c] < 0).any():
                warns.append(f"column '{c}' has negative values — sensible for a price/count target?")
    for w in warns: print(f"WARN: {w}")
    if errs:
        for e in errs: print(f"FAIL: {e}")
        sys.exit(1)
    print(f"OK: {path} is safe to upload ({len(sub)} rows)")

if __name__ == "__main__":
    main(sys.argv[1])
