"""One-command EDA report. Run: python src/eda.py [--sample 50000]
Prints shapes, dtypes, missing %, target stats, text length stats, duplicates,
train/test overlap on id and on text, and saves basic figures to outputs/figs/.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.data import load_train, load_test, combined_text


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main(sample=None):
    train, test = load_train(nrows=sample), load_test(nrows=sample)

    section("SHAPES")
    print(f"train: {train.shape}   test: {test.shape}")
    print(f"test/train ratio: {len(test)/max(len(train),1):.2f}")

    section("COLUMNS / DTYPES / MISSING %")
    info = pd.DataFrame({
        "dtype": train.dtypes.astype(str),
        "missing_%": (train.isna().mean() * 100).round(2),
        "nunique": train.nunique(),
    })
    print(info.to_string())

    if C.TARGET_COL in train.columns:
        section(f"TARGET: {C.TARGET_COL}")
        t = train[C.TARGET_COL]
        if C.TASK == "regression":
            print(t.describe().to_string())
            print(f"skew: {t.skew():.2f}  |  zeros: {(t==0).mean()*100:.2f}%  |  negatives: {(t<0).mean()*100:.2f}%")
            print("-> if skew > 2, LOG_TARGET=True is probably right")
        else:
            vc = t.value_counts()
            print(vc.head(30).to_string())
            print(f"classes: {t.nunique()} | rarest class count: {vc.min()}")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            if C.TASK == "regression":
                ax[0].hist(t.dropna(), bins=60); ax[0].set_title(C.TARGET_COL)
                ax[1].hist(np.log1p(np.clip(t.dropna(), 0, None)), bins=60); ax[1].set_title("log1p")
            else:
                t.value_counts().head(30).plot.bar(ax=ax[0]); ax[0].set_title("class counts")
            fig.tight_layout(); fig.savefig(C.FIG_DIR / "target.png", dpi=110)
            print(f"saved {C.FIG_DIR/'target.png'}")
        except Exception as e:
            print(f"(plot skipped: {e})")

    section("TEXT LENGTHS")
    try:
        for name, df in [("train", train), ("test", test)]:
            txt = combined_text(df)
            lens = txt.str.len()
            print(f"{name}: chars mean={lens.mean():.0f} p50={lens.median():.0f} p95={lens.quantile(.95):.0f} max={lens.max()}")
            empty = (lens < 5).mean() * 100
            print(f"  near-empty text rows: {empty:.2f}%")
    except Exception as e:
        print(f"(skipped: {e})")

    section("DUPLICATES & LEAKAGE CHECKS")
    if C.ID_COL in train.columns and C.ID_COL in test.columns:
        overlap = set(train[C.ID_COL]) & set(test[C.ID_COL])
        print(f"id overlap train∩test: {len(overlap)}  <-- if >0, investigate (possible leak)")
    try:
        tr_txt, te_txt = combined_text(train), combined_text(test)
        print(f"duplicate texts within train: {tr_txt.duplicated().sum()}")
        text_overlap = len(set(tr_txt) & set(te_txt))
        print(f"exact text overlap train∩test: {text_overlap}  <-- if big, exact-match lookup is free score")
    except Exception as e:
        print(f"(skipped: {e})")

    section("DONE — read this whole output adversarially before modeling")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None, help="read only N rows (fast first look)")
    a = p.parse_args()
    main(a.sample)
