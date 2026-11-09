r"""Extra cross-encoder training pairs from training entities no earlier step used.

sample builds one batch:
  1. draws --entities training Source 1 rows outside the held-out set (the
     holdout_set of a src/loss_breakdown.py --dump) and outside the entities
     of every file in --exclude (the matcher's --dump-pairs file, earlier
     batches' train_pairs.npz);
  2. builds their pre-ranked candidates and pair features with the matcher's
     own code (src/pair_model.py: same blocking keys, max_df 300, max_group
     200, top 40 by the learned pre-ranker in --model-dir) and scores them
     with the mean of the boosters in --model-dir;
  3. keeps every positive, every negative scoring at or above --floor, and one
     other random negative per positive (at least one per entity);
  4. drops every pair whose target record is a target of a held-out pair
     (heldout_pairs.npz of --export), so no held-out target is trained on.
Rows are processed CHUNK at a time, so memory stays flat.

merge writes the union of several batches, for a model trained on them in
one pass.

Both write --out as a pair folder for src/ce_rerank.py: train_pairs.npz
(s1_id int64, enc_id int64, label int8, sorted by s1_id then enc_id), copies of
the export's heldout_pairs.npz, test_pairs.npz and shards.json, and the
export's manifest.json with the training-pair source, counts and sha256
replaced. enc_id is source * 2_000_000_000 + the numeric target id, as in
src/pair_model.py.

Run:
    python src/ce_more_pairs.py sample --entities 400000 \
        --heldout-dump outputs/cache/cv4m5_addridf.pkl \
        --exclude outputs/cache/pairs_addridf.npz \
        --model-dir outputs/models/sub09_g127_pseudo --out outputs/ce/pairs_A
    python src/ce_more_pairs.py sample --entities 400000 \
        --heldout-dump outputs/cache/cv4m5_addridf.pkl \
        --exclude outputs/cache/pairs_addridf.npz,outputs/ce/pairs_A/train_pairs.npz \
        --model-dir outputs/models/sub09_g127_pseudo --seed-offset 606 --out outputs/ce/pairs_B
    python src/ce_more_pairs.py merge --parts outputs/ce/pairs_A,outputs/ce/pairs_B \
        --out outputs/ce/pairs_AB
"""
import argparse
import glob
import json
import shutil
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import er_io, pair_model as pm
from src.ce_pairs import load_heldout, sha256

ENC = pm.ENCODE
CHUNK = 20000      # sampled entities per feature pass
MAX_DF = 300       # blocking settings of the matcher's runs
MAX_GROUP = 200
TOP_K = 40
EASY_SEED = 7      # generator of the random easy negatives
PAIR_FILES = ("train_pairs.npz", "heldout_pairs.npz", "test_pairs.npz")
COPIED = ("heldout_pairs.npz", "test_pairs.npz", "shards.json")


def sample_rows(n: int, holdout, exclude, seed_offset: int):
    """(sorted sampled rows, pool size): n training Source 1 rows outside holdout and the excluded entities.

    A file with s1_row (the matcher's --dump-pairs output) gives rows
    directly; a file with only s1_id (an earlier batch's train_pairs.npz) is
    mapped to rows through the training Source 1 ids. The draw uses seed
    config.SEED + seed_offset.
    """
    s1 = er_io.parse_source("train", 1)
    used = []
    for path in exclude:
        with np.load(path) as f:
            used.append(np.unique(f["s1_row"]) if "s1_row" in f.files
                        else er_io.index_of(s1["ids"], np.unique(f["s1_id"])))
    used = np.unique(np.concatenate(used))
    pool_rows = np.setdiff1d(np.arange(len(s1["ids"])), np.union1d(holdout, used))
    rng = np.random.default_rng(C.SEED + seed_offset)
    return np.sort(rng.choice(pool_rows, size=n, replace=False)), len(pool_rows)


def build_batch(rows, model_dir, held_enc, floor: float, workers: int) -> dict:
    """Kept pairs of the sampled rows: {s1_id, enc_id, label}, sorted by s1_id then enc_id."""
    import lightgbm as lgb

    t0 = time.time()
    s1 = er_io.parse_source("train", 1)
    gt = er_io.parse_ground_truth()
    rows_gt = er_io.index_of(s1["ids"], gt["s1"])
    boosters = [lgb.Booster(model_file=f) for f in sorted(glob.glob(f"{model_dir}/booster_*.txt"))]
    if not boosters:
        raise SystemExit(f"no booster_*.txt in {model_dir}")
    preranker = (Path(model_dir) / "preranker.txt").read_text(encoding="utf-8")
    ctry = er_io.country_keys(s1)
    tid = {s: er_io.parse_source("train", s)["ids"] for s in (2, 3)}
    index = pm.build_index("train", MAX_DF)
    rng = np.random.default_rng(EASY_SEED)
    pool = get_context("spawn").Pool(workers) if workers > 1 else None
    keep_s1, keep_e, keep_y = [], [], []
    try:
        for a in range(0, len(rows), CHUNK):
            part = rows[a:a + CHUNK]
            truth = {}
            for k in np.flatnonzero(np.isin(rows_gt, part)):
                truth[int(rows_gt[k])] = {int(gt["match_src"][j]) * ENC + int(gt["match_id"][j])
                                          for j in range(gt["offsets"][k], gt["offsets"][k + 1])}
            text = pm.read_s1_text("train", part)
            cands = pm.gather(part, s1, ctry, text, index, MAX_GROUP)
            tt = pm.fetch_target_text(cands, "train")
            X, keys = pm.build_features_parallel(cands, text, tt, pool, workers, "train", TOP_K,
                                                 preranker)
            if not len(keys):
                continue
            X = pm.add_group_features(X, keys)
            p = np.mean([b.predict(X) for b in boosters], axis=0)
            r = np.array([q[0] for q in keys], dtype=np.int64)
            e = np.array([q[1] * ENC + int(tid[q[1]][q[2]]) for q in keys], dtype=np.int64)
            y = np.fromiter((x in truth.get(rr, ()) for rr, x in zip(r.tolist(), e.tolist())),
                            bool, len(e))
            ok = ~np.isin(e, held_enc)
            hard = ~y & (p >= floor)
            npos = np.bincount(np.searchsorted(part, r), weights=y, minlength=len(part))
            # One random easy negative per positive, at least one per entity:
            # per entity, easy negatives first in random order, then rank < cap.
            u = rng.random(len(e))
            easy = ~y & ~hard
            order = np.lexsort((u, ~easy, r))
            rr, ee = r[order], easy[order]
            first = np.r_[True, rr[1:] != rr[:-1]]
            rank = np.arange(len(rr)) - np.flatnonzero(first)[np.cumsum(first) - 1]
            cap = np.maximum(npos, 1)[np.searchsorted(part, rr)]
            pick = np.zeros(len(e), bool)
            pick[order[ee & (rank < cap)]] = True
            m = ok & (y | hard | pick)
            keep_s1.append(s1["ids"][r[m]])
            keep_e.append(e[m])
            keep_y.append(y[m])
            print(f"  {a + len(part):,}/{len(rows):,} | kept {int(m.sum()):,} of {len(e):,} "
                  f"(pos {int((y & m).sum()):,}, hard neg {int((hard & m).sum()):,}) | "
                  f"{time.time() - t0:.0f}s", flush=True)
    finally:
        if pool is not None:
            pool.close()
    s1_id = np.concatenate(keep_s1).astype(np.int64)
    enc = np.concatenate(keep_e)
    o = np.lexsort((enc, s1_id))
    return {"s1_id": s1_id[o], "enc_id": enc[o], "label": np.concatenate(keep_y)[o].astype(np.int8)}


def write_folder(export: Path, out: Path, arrays: dict, source: str, extra: dict) -> None:
    """Write train_pairs.npz, copy the export's other pair files and write the updated manifest."""
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "train_pairs.npz", **arrays)
    for name in COPIED:
        if (export / name).exists():
            shutil.copy2(export / name, out / name)
    man = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
    lab = arrays["label"]
    man.setdefault("sources", {})["fit_pairs"] = source
    man.setdefault("counts", {})["train_pairs"] = {
        "pairs": int(len(lab)), "positives": int(lab.sum()),
        "entities": int(len(np.unique(arrays["s1_id"])))}
    man["sha256"] = {n: sha256(out / n) for n in PAIR_FILES if (out / n).exists()}
    man["batch"] = extra
    (out / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"wrote {out}: {len(lab):,} training pairs ({int(lab.sum()):,} positive)", flush=True)


def check_out(out: Path) -> None:
    """Refuse to overwrite the training pairs of an existing folder."""
    if (out / "train_pairs.npz").exists():
        raise SystemExit(f"{out}/train_pairs.npz already exists")


def cmd_sample(a) -> None:
    """Sample the rows, build the batch and write the pair folder."""
    export, out = Path(a.export), Path(a.out)
    check_out(out)
    exclude = [x.strip() for x in a.exclude.split(",") if x.strip()]
    holdout = np.array(sorted(load_heldout(a.heldout_dump)["holdout_set"]), dtype=np.int64)
    with np.load(export / "heldout_pairs.npz") as f:
        held_enc = np.unique(f["enc_id"])
    rows, n_pool = sample_rows(a.entities, holdout, exclude, a.seed_offset)
    print(f"pool {n_pool:,}, sampled {len(rows):,}", flush=True)
    arrays = build_batch(rows, a.model_dir, held_enc, a.floor, a.workers)
    extra = {"entities": a.entities, "pool": n_pool, "floor": a.floor, "seed_offset": a.seed_offset,
             "heldout_dump": str(a.heldout_dump), "exclude": exclude, "model_dir": str(a.model_dir)}
    write_folder(export, out, arrays,
                 f"ce_more_pairs.py sample: {a.entities:,} training entities outside the held-out set "
                 f"and {', '.join(exclude)}", extra)


def cmd_merge(a) -> None:
    """Write a pair folder whose training pairs are the union of the given folders' train_pairs.npz."""
    out = Path(a.out)
    check_out(out)
    parts = [x.strip() for x in a.parts.split(",") if x.strip()]
    loaded = []
    for d in parts:
        with np.load(Path(d) / "train_pairs.npz") as f:
            loaded.append({k: f[k] for k in ("s1_id", "enc_id", "label")})
    s1_id = np.concatenate([x["s1_id"] for x in loaded])
    enc = np.concatenate([x["enc_id"] for x in loaded])
    lab = np.concatenate([x["label"] for x in loaded])
    o = np.lexsort((enc, s1_id))
    s1_id, enc, lab = s1_id[o], enc[o], lab[o]
    if len(o) > 1 and np.any((s1_id[1:] == s1_id[:-1]) & (enc[1:] == enc[:-1])):
        raise SystemExit("the batches share a (s1_id, enc_id) pair")
    write_folder(Path(a.export), out, {"s1_id": s1_id, "enc_id": enc, "label": lab},
                 f"ce_more_pairs.py merge of {', '.join(parts)}", {"merged": parts})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    export_default = str(C.OUT_DIR / "ce" / "export")
    s = sub.add_parser("sample", help="build one batch of training pairs from unused training entities")
    s.add_argument("--entities", type=int, required=True, help="training Source 1 entities to sample")
    s.add_argument("--heldout-dump", required=True,
                   help="src/loss_breakdown.py --dump pickle; its holdout_set is never sampled")
    s.add_argument("--exclude", required=True,
                   help="comma separated pair files whose entities are never sampled: the matcher's "
                        "--dump-pairs file and earlier batches' train_pairs.npz")
    s.add_argument("--model-dir", required=True,
                   help="model folder with booster_*.txt and preranker.txt (src/predict_matches.py)")
    s.add_argument("--export", default=export_default,
                   help="src/ce_pairs.py export folder (heldout_pairs.npz, manifest.json)")
    s.add_argument("--out", required=True, help="pair folder to write")
    s.add_argument("--floor", type=float, default=0.02,
                   help="negatives scoring at or above this are all kept")
    s.add_argument("--seed-offset", type=int, default=505, help="sampling seed is config.SEED + this")
    s.add_argument("--workers", type=int, default=8, help="processes for feature building")
    m = sub.add_parser("merge", help="write a pair folder with the union of several batches")
    m.add_argument("--parts", required=True, help="comma separated pair folders with train_pairs.npz")
    m.add_argument("--export", default=export_default,
                   help="src/ce_pairs.py export folder (heldout_pairs.npz, manifest.json)")
    m.add_argument("--out", required=True, help="pair folder to write")
    args = p.parse_args()
    cmd_sample(args) if args.cmd == "sample" else cmd_merge(args)
