"""Held-out score at test density: every Source 1 entity of a set of cities.

A random held-out sample holds under 1% of Source 1, so two sampled entities
almost never compete for the same target record, and target-argmax (each record
goes to the entity that scores it highest) is never exercised. On test every
entity competes. Taking every entity whose address lies in a random subset of
cities restores most of that competition, because the records two entities
fight over are nearly always in the same city.

The matcher is fitted on the usual random sample (pair_model.fit); entities
from that sample still compete here but are left out of the average.

Run:
    python src/dense_eval.py --entities 60000 --city-mod 40 --workers 9
"""
import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import er_io, pair_model as pm
from src.er_norm import hash64, normalize_name


def city_rows(city_mod: int) -> np.ndarray:
    """Source 1 training rows whose city falls in a 1/city_mod subset of cities.

    The city is the second-to-last comma-separated address component, which
    is the city for US addresses and the district for Indian ones.
    """
    rows = []
    with open(C.source_file("train", 1), encoding="utf-8") as f:
        next(f, None)
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            comps = [c for c in parts[2].split(",") if c.strip()] if len(parts) == 4 else []
            if len(comps) >= 2 and hash64(normalize_name(comps[-2])) % city_mod == 0:
                rows.append(i)
    return np.array(rows, dtype=np.int64)


def score_rows(rows, fitted, max_df, max_group, pool, workers, shard=40000):
    """(s1_row, enc_id, prob) for every pre-ranked candidate of the given rows."""
    s1 = er_io.parse_source("train", 1)
    ctry = er_io.country_keys(s1)
    tid = {s: er_io.parse_source("train", s)["ids"] for s in (2, 3)}
    index = pm.build_index("train", max_df)
    out_r, out_e, out_p = [], [], []
    for start in range(0, len(rows), shard):
        part = rows[start:start + shard]
        text = pm.read_s1_text("train", part)
        cands = pm.gather(part, s1, ctry, text, index, max_group)
        tgt = pm.fetch_target_text(cands, "train")
        X, keys = pm.build_features_parallel(cands, text, tgt, pool, workers, "train",
                                             fitted["top_k"])
        X = pm.add_group_features(X, keys)
        if keys:
            out_p.append(fitted["model"].predict(X))
            out_r.append(np.array([k[0] for k in keys], dtype=np.int64))
            out_e.append(np.array([k[1] * pm.ENCODE + int(tid[k[1]][k[2]]) for k in keys],
                                  dtype=np.int64))
    return np.concatenate(out_r), np.concatenate(out_e), np.concatenate(out_p)


def macro(truth, rows, pred):
    """Macro F0.5 over rows for {row: set(ids)} predictions."""
    tot = 0.0
    for r in rows:
        t, q = truth[r], pred.get(r, set())
        if not t and not q:
            tot += 1.0
        elif t and q:
            c = len(t & q)
            tot += 1.25 * c / (len(q) + 0.25 * len(t)) if c else 0.0
    return tot / len(rows)


def main(a) -> None:
    """Fit, score the dense city set, and sweep threshold with and without argmax."""
    t0 = time.time()
    pool = get_context("spawn").Pool(a.workers) if a.workers > 1 else None
    try:
        fitted = pm.fit(a.entities, a.max_df, a.max_group, False, pool, a.workers, a.top_k)
        rows = city_rows(a.city_mod)
        print(f"dense set: {len(rows):,} entities ({time.time() - t0:.0f}s)", flush=True)
        r, e, p = score_rows(rows, fitted, a.max_df, a.max_group, pool, a.workers)
    finally:
        if pool is not None:
            pool.close()
    from src.full_density_eval import argmax_mask, truth_all
    truth, _ = truth_all()
    _, sample, _ = pm.sample_entities(a.entities, C.SEED)
    fitted_rows = set(sample.tolist()) - fitted["holdout_set"]
    evaluate = [x for x in rows.tolist() if x not in fitted_rows]
    am = argmax_mask(e, p)
    results = {}
    for t in [float(x) for x in a.thresholds.split(",")]:
        for name, mask in (("threshold", p >= t), ("argmax_threshold", am & (p >= t))):
            pred = {}
            for rr, ee in zip(r[mask].tolist(), e[mask].tolist()):
                pred.setdefault(rr, set()).add(ee)
            results[f"{name} {t:.2f}"] = macro(truth, evaluate, pred)
    print(f"\ndense held-out: {len(evaluate):,} entities "
          f"({len(rows) - len(evaluate):,} fitted entities excluded)")
    for k, v in results.items():
        print(f"  {k:>22}: {v:.5f}")
    print(f"total {time.time() - t0:.0f}s")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"entities": len(evaluate), "city_mod": a.city_mod,
                                           "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=int, default=60000)
    ap.add_argument("--city-mod", type=int, default=40)
    ap.add_argument("--max-df", type=int, default=300)
    ap.add_argument("--max-group", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--thresholds", default="0.40,0.50,0.60,0.70")
    ap.add_argument("--out", default="results/dense_eval/dense.json")
    main(ap.parse_args())
