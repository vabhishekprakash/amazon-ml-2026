"""Create a fake Amazon-catalog-style dataset so every script can be tested with zero downloads.
Run: python scripts/make_synthetic_data.py  -> data/train.csv, test.csv, sample_submission.csv
Mimics 2025's shape: product text -> price (regression, SMAPE).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

rng = np.random.default_rng(C.SEED)
N_TR, N_TE = 6000, 2000
BRANDS = ["Acme", "Zenvo", "Kloud", "Prax", "Nimbus", "Orbit", "Vela", "Quanta"]
CATS = {"headphones": 60, "laptop stand": 25, "usb cable": 8, "monitor": 180, "backpack": 35,
        "keyboard": 45, "webcam": 55, "router": 70, "ssd": 90, "desk lamp": 20}
ADJ = ["wireless", "ergonomic", "premium", "compact", "rgb", "portable", "heavy-duty", "slim", "pro", "eco"]

def make(n, with_target=True):
    rows = []
    for i in range(n):
        cat = rng.choice(list(CATS))
        brand = rng.choice(BRANDS)
        adjs = rng.choice(ADJ, size=rng.integers(1, 4), replace=False)
        size = rng.integers(1, 6)
        title = f"{brand} {' '.join(adjs)} {cat} v{rng.integers(1,9)}"
        desc = f"{' '.join(adjs)} {cat} by {brand}. size class {size}. " + \
               " ".join(rng.choice(ADJ, size=6))
        price = CATS[cat] * (1 + 0.15 * len(adjs)) * (1 + 0.1 * size) * float(rng.lognormal(0, 0.25))
        rows.append({"sample_id": i, "title": title, "description": desc,
                     **({"price": round(price, 2)} if with_target else {})})
    return pd.DataFrame(rows)

train = make(N_TR, True)
test = make(N_TE, False)
test["sample_id"] += 100000
train.to_csv(C.DATA_DIR / "train.csv", index=False)
test.to_csv(C.DATA_DIR / "test.csv", index=False)
pd.DataFrame({"sample_id": test["sample_id"], "price": 1.0}).to_csv(C.DATA_DIR / "sample_submission.csv", index=False)
print(f"wrote synthetic train {train.shape}, test {test.shape} to {C.DATA_DIR}")
