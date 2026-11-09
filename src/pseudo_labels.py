"""Pseudo-labelled training pairs for countries that have no ground truth.

The test split holds records from a country the training data never shows.
A model fitted on the other countries has already scored every test pair
(predict_matches.py --dump-scores). Its confident decisions on pairs of that
country, with the full feature set rebuilt here from the test split's own
files, become extra training rows, so the next model sees what that
country's records look like.

A pair is labelled from its score in the dump:
  positive  score >= --hi and the entity is the record's highest-scoring claimant
  negative  score <= --lo
  anything in between is left out.
Only Source 1 entities whose country has no Source 1 record in training are
sampled; no label of the test split exists or is used.

Run:
    python src/pseudo_labels.py --scores outputs/scores/test_v8 \
        --model outputs/models/sub08_addridf --entities 100000 \
        --out outputs/cache/pseudo_v8.npz
"""
import argparse
import glob
import hashlib
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import er_io, pair_model as pm

ROW_BITS = 21  # test Source 1 has 1.73M rows < 2**21


def pair_key(rows, enc):
    """One int64 per (Source 1 row, encoded target id)."""
    return np.asarray(enc, dtype=np.int64) * (1 << ROW_BITS) + np.asarray(rows, dtype=np.int64)


def unseen_rows(split: str):
    """Rows of the split's Source 1 whose country never occurs in training Source 1."""
    seen = set(er_io.parse_source("train", 1)["countries"])
    s1 = er_io.parse_source(split, 1)
    labels = np.array(s1["countries"])
    unseen = sorted(set(labels) - seen)
    rows = np.flatnonzero(np.isin(labels[s1["country"]], unseen))
    return rows, unseen


def read_dump(scores_dir: str, rows: np.ndarray):
    """(sorted pair keys, their scores) for the given rows, and each record's owner.

    The owner of a record is the entity scoring it highest among pairs scored
    0.5 or more anywhere in the split, as target-argmax would assign it.
    """
    files = sorted(glob.glob(f"{scores_dir}/shard_*.npz"))
    if not files:
        raise SystemExit(f"no shard_*.npz in {scores_dir}")
    want = np.sort(rows)
    own_k, own_p, hi_e, hi_r, hi_p = [], [], [], [], []
    for f in files:
        with np.load(f) as b:
            r, e, p = b["s1_row"], b["enc_id"], b["prob"]
        m = p >= 0.5
        hi_e.append(e[m]); hi_r.append(r[m]); hi_p.append(p[m])
        s = np.isin(r, want)
        own_k.append(pair_key(r[s], e[s])); own_p.append(p[s])
    k = np.concatenate(own_k); p = np.concatenate(own_p)
    o = np.argsort(k)
    e = np.concatenate(hi_e); r = np.concatenate(hi_r); q = np.concatenate(hi_p)
    order = np.lexsort((-q, e))
    e, r, q = e[order], r[order], q[order]
    first = np.concatenate(([True], e[1:] != e[:-1]))
    # A record whose two best claimants score exactly the same (isotonic
    # calibration makes ties common) has no clear owner.
    tied = np.zeros(len(e), dtype=bool)
    tied[:-1] = (e[1:] == e[:-1]) & (q[1:] == q[:-1])
    owner = np.where(tied, -1, r)
    return k[o], p[o], e[first], owner[first], len(files)


def harvest(scores_dir, model_dir, n_entities, hi, lo, max_df, max_group, top_k, shard,
            workers, seed):
    """Feature matrix and labels of the confident pairs of sampled unseen-country entities."""
    t0 = time.time()
    pool_rows, unseen = unseen_rows("test")
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(pool_rows, size=min(n_entities, len(pool_rows)), replace=False))
    print(f"unseen countries {unseen}: {len(pool_rows):,} entities, sampled {len(rows):,}", flush=True)
    keys_sorted, probs, own_enc, own_row, n_files = read_dump(scores_dir, rows)
    print(f"dump read ({n_files} shards, {len(keys_sorted):,} pairs of the sample) "
          f"({time.time() - t0:.0f}s)", flush=True)

    pr = Path(model_dir) / "preranker.txt"
    preranker = pr.read_text(encoding="utf-8") if pr.exists() else None
    s1 = er_io.parse_source("test", 1)
    ctry_s1 = er_io.country_keys(s1)
    tid = {s: er_io.parse_source("test", s)["ids"] for s in (2, 3)}
    index = pm.build_index("test", max_df)
    pool = get_context("spawn").Pool(workers) if workers > 1 else None
    Xs, labels, s1r, encs, ps = [], [], [], [], []
    found = total = 0
    try:
        for start in range(0, len(rows), shard):
            part = rows[start:start + shard]
            text = pm.read_s1_text("test", part)
            cands = pm.gather(part, s1, ctry_s1, text, index, max_group)
            tgt_text = pm.fetch_target_text(cands, "test")
            X, keys = pm.build_features_parallel(cands, text, tgt_text, pool, workers, "test",
                                                 top_k or None, preranker)
            del tgt_text
            if not len(keys):
                continue
            X = pm.add_group_features(X, keys)
            r = np.array([k[0] for k in keys], dtype=np.int64)
            e = np.array([k[1] * pm.ENCODE + int(tid[k[1]][k[2]]) for k in keys], dtype=np.int64)
            key = pair_key(r, e)
            p = np.full(len(key), np.nan)
            if len(keys_sorted):
                pos = np.clip(np.searchsorted(keys_sorted, key), 0, len(keys_sorted) - 1)
                hit = keys_sorted[pos] == key
                p[hit] = probs[pos[hit]]
                found += int(hit.sum())
            total += len(key)
            owner = np.full(len(e), -1)
            if len(own_enc):
                op = np.clip(np.searchsorted(own_enc, e), 0, len(own_enc) - 1)
                owner = np.where(own_enc[op] == e, own_row[op], -1)
            y = np.full(len(key), -1, dtype=np.int8)
            y[(p >= hi) & (owner == r)] = 1
            y[p <= lo] = 0
            keep = y >= 0
            Xs.append(X[keep]); labels.append(y[keep]); s1r.append(r[keep]); encs.append(e[keep])
            ps.append(p[keep])
            print(f"  {start + len(part):,}/{len(rows):,} entities | {int(keep.sum()):,} of "
                  f"{len(key):,} pairs labelled | {time.time() - t0:.0f}s", flush=True)
    finally:
        if pool is not None:
            pool.close()
    if not Xs:
        raise SystemExit("no pairs were rebuilt for the sampled entities")
    if total and found / total < 0.99:
        raise SystemExit(f"only {found / total:.1%} of rebuilt pairs are in the dump: "
                         "the dump was scored with different candidates or code")
    return (np.concatenate(Xs), np.concatenate(labels), np.concatenate(s1r),
            np.concatenate(encs), np.concatenate(ps), unseen, len(rows), found / max(total, 1))


def main(a) -> None:
    """Harvest and write the pseudo-labelled pairs with a summary."""
    X, y, r, e, p, unseen, n, cover = harvest(
        a.scores, a.model, a.entities, a.hi, a.lo, a.max_df, a.max_group, a.top_k, a.shard,
        a.workers, a.seed)
    pr = Path(a.model) / "preranker.txt"
    meta = {"scores": a.scores, "model": a.model, "hi": a.hi, "lo": a.lo, "entities": n,
            "top_k": a.top_k, "preranker_sha256": (hashlib.sha256(pr.read_bytes()).hexdigest()
                                                   if pr.exists() else None),
            "countries": [str(c) for c in unseen], "pairs": int(len(y)), "positives": int(y.sum()),
            "entities_with_positive": int(len(np.unique(r[y == 1]))), "dump_coverage": cover}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, X=X.astype(np.float32), label=y, s1_row=r.astype(np.int32), enc_id=e,
             prob=p.astype(np.float32), features=np.array(pm.FEATURES + pm.GROUP_FEATURES),
             meta=np.array(json.dumps(meta)))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="predict_matches.py --dump-scores directory")
    ap.add_argument("--model", required=True, help="model directory of that run (preranker.txt)")
    ap.add_argument("--entities", type=int, default=100000)
    ap.add_argument("--hi", type=float, default=0.9)
    ap.add_argument("--lo", type=float, default=0.05)
    ap.add_argument("--max-df", type=int, default=300)
    ap.add_argument("--max-group", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--shard", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--seed", type=int, default=C.SEED + 303)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
