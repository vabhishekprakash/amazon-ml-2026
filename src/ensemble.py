"""Ensemble N submission CSVs (same format). Mean or rank-mean, optional weights.
Run: python src/ensemble.py out.csv a.csv b.csv c.csv [--ranks] [--weights 2,1,1]
Rank-mean is robust when models' output scales differ. For classification use --vote.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

def main():
    p = argparse.ArgumentParser()
    p.add_argument("out"); p.add_argument("subs", nargs="+")
    p.add_argument("--ranks", action="store_true"); p.add_argument("--vote", action="store_true")
    p.add_argument("--weights", default=None)
    a = p.parse_args()
    dfs = [pd.read_csv(s) for s in a.subs]
    base = dfs[0].copy()
    tcols = [c for c in base.columns if c != C.ID_COL]
    for df in dfs[1:]:
        assert list(df.columns) == list(base.columns) and len(df) == len(base), "submission shapes differ"
        assert df[C.ID_COL].astype(str).equals(base[C.ID_COL].astype(str)), "id order differs — fix first"
    w = np.array([float(x) for x in a.weights.split(",")]) if a.weights else np.ones(len(dfs))
    w = w / w.sum()
    for c in tcols:
        cols = np.vstack([df[c].values for df in dfs])
        if a.vote:
            from scipy import stats
            base[c] = stats.mode(cols.astype(str), axis=0, keepdims=False).mode
        elif a.ranks:
            ranks = np.vstack([pd.Series(x).rank(pct=True).values for x in cols])
            blend_rank = (ranks * w[:, None]).sum(0)
            # map blended rank back onto the value distribution of the best (first) model
            order = np.argsort(np.argsort(blend_rank))
            base[c] = np.sort(cols[0])[order]
        else:
            base[c] = (cols * w[:, None]).sum(0)
    base.to_csv(a.out, index=False)
    print(f"wrote {a.out} from {len(dfs)} subs (weights={w.round(3)})")

if __name__ == "__main__":
    main()
