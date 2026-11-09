"""Apply the pairwise matcher to a whole split and write both submission files.

Steps:
  1. Fit the matcher on a sample of training entities (pair_model.fit) and
     report its held-out score. The model and calibrator are saved.
  2. Build the blocking index for the target split once.
  3. Walk every Source 1 entity in shards: gather candidates, build features
     across a process pool, keep each entity's top candidates by name and
     address similarity (the pre-ranker), score those, and admit per entity.
     The pre-ranked lists are what the matcher sees, so they are what each
     shard writes to candidate_pairs.tsv; memory stays flat.
  4. One-to-one over the whole split: a target record kept by two entities
     stays only with the one that scores it highest.
  5. Write matching_results.tsv, one row per Source 1 entity in file order.

Feature building dominates the run time (about 65 microseconds per pair per
process), so --workers should be close to the core count.

Run:
    python src/predict_matches.py --split test --run-name sub02_pair_model --workers 8
    python src/predict_matches.py --split test --limit 20000 --run-name smoke   # quick check
"""
import argparse
import json
import pickle
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import er_io, pair_model as pm
from src.log_experiment import log_experiment


def decode(enc: int) -> str:
    """Encoded target id back to its entity_id string, e.g. S2-12345."""
    src, num = divmod(enc, pm.ENCODE)
    return f"S{src}-{num}"


def predict_split(split, fitted, max_df, max_group, shard, pool, workers, limit, cand_path,
                  dump_dir=None, countries=None):
    """Score every Source 1 entity of a split; returns {s1_row: [(prob, enc_id)]} admitted.

    Candidate lists are streamed to cand_path in Source 1 row order.

    dump_dir writes every scored pair to disk, one .npz per shard, so decision
    rules can be compared offline without re-scoring. Per-entity admission and
    per-target assignment are cheap to try once the probabilities exist; the
    scoring pass is what costs an hour.

    countries, if given, limits scoring to Source 1 entities of those countries.
    """
    t0 = time.time()
    s1 = er_io.parse_source(split, 1)
    n = len(s1["ids"]) if limit is None else min(limit, len(s1["ids"]))
    ctry_s1 = er_io.country_keys(s1)
    tid = {s: er_io.parse_source(split, s)["ids"] for s in (2, 3)}
    index = pm.build_index(split, max_df)
    print(f"index built for {split} ({time.time() - t0:.0f}s); {n:,} entities", flush=True)

    admitted = {}
    n_pairs = 0
    s1_ids = s1["ids"].tolist()
    wanted = (np.isin(np.array(s1["countries"])[s1["country"]], list(countries))
              if countries else None)
    with open(cand_path, "w", encoding="utf-8", newline="\n") as out:
        out.write(f"{C.GT_ID_COL}\t{C.CANDIDATE_COL}\n")
        for start in range(0, n, shard):
            rows = np.arange(start, min(start + shard, n))
            if wanted is not None:
                rows = rows[wanted[rows]]
                if not len(rows):
                    continue
            text = pm.read_s1_text(split, rows)
            cands = pm.gather(rows, s1, ctry_s1, text, index, max_group)
            tgt_text = pm.fetch_target_text(cands, split)
            X, keys = pm.build_features_parallel(cands, text, tgt_text, pool, workers, split,
                                                 fitted["top_k"], fitted.get("preranker"))
            del tgt_text
            X = pm.add_group_features(X, keys)
            n_pairs += len(keys)

            p = (np.mean([m.predict(X) for m in fitted["models"]], axis=0) if len(keys)
                 else np.zeros(0))
            if fitted["iso"] is not None and len(p):
                p = fitted["iso"].predict(p)
            if dump_dir is not None and len(keys):
                Path(dump_dir).mkdir(parents=True, exist_ok=True)
                np.savez(Path(dump_dir) / f"shard_{start:08d}.npz",
                         s1_row=np.array([k[0] for k in keys], dtype=np.int32),
                         enc_id=np.array([k[1] * pm.ENCODE + int(tid[k[1]][k[2]])
                                          for k in keys], dtype=np.int64),
                         prob=p.astype(np.float32))

            by_ent = {}
            for (r, src, tr), prob in zip(keys, p):
                by_ent.setdefault(r, []).append((float(prob), src * pm.ENCODE + int(tid[src][tr])))
            for r, lst in by_ent.items():
                probs = np.array([x[0] for x in lst])
                k_hat = (fitted["k_global"] if fitted["k_global"] is not None
                         else max(float(probs.sum()), 1.0))
                chosen = set(pm.admit(probs, [x[1] for x in lst], k_hat))
                admitted[r] = [(pr, i) for pr, i in lst if i in chosen]

            kept = {}
            for r, src, tr in keys:
                kept.setdefault(r, []).append(f"S{src}-{tid[src][tr]}")
            for r in rows.tolist():
                out.write(f"S1-{s1_ids[r]}\t{','.join(kept.get(r, ()))}\n")
            print(f"  {rows[-1] + 1:,}/{n:,} entities | {n_pairs:,} pairs "
                  f"| {time.time() - t0:.0f}s", flush=True)
    return admitted, s1_ids[:n], n_pairs


def one_to_one(admitted):
    """Keep each target only for the entity that scores it highest."""
    best = {}
    for r, lst in admitted.items():
        for prob, i in lst:
            if i not in best or prob > best[i][1]:
                best[i] = (r, prob)
    out = {}
    for i, (r, prob) in best.items():
        out.setdefault(r, []).append((prob, i))
    return {r: [i for _, i in sorted(v, reverse=True)] for r, v in out.items()}


def write_matches(path, s1_ids, assigned):
    """One row per Source 1 entity in file order, empty list when unmatched."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{C.GT_ID_COL}\t{C.GT_MATCH_COL}\n")
        for r, num in enumerate(s1_ids):
            f.write(f"S1-{num}\t{','.join(decode(i) for i in assigned.get(r, ()))}\n")


def main(a) -> None:
    """Fit, predict the split, write and summarise both files."""
    t0 = time.time()
    pool = get_context("spawn").Pool(a.workers) if a.workers > 1 else None
    try:
        if a.load_model and a.pseudo:
            raise SystemExit("--pseudo needs a fit; it cannot be combined with --load-model")
        if a.load_model:
            # Refitting costs about 30 minutes and produces the same booster, so
            # a repeat run over the same training sample should reuse it.
            import lightgbm as lgb
            md = Path(a.load_model)
            with open(md / "calibration.pkl", "rb") as f:
                cal = pickle.load(f)
            models = [lgb.Booster(model_file=str(f)) for f in sorted(md.glob("booster*.txt"))]
            pr = md / "preranker.txt"
            fitted = {"model": models[-1], "models": models,
                      "preranker": pr.read_text(encoding="utf-8") if pr.exists() else None,
                      "iso": cal["iso"], "k_global": cal["k_global"],
                      "top_k": a.top_k or 50, "entities": a.entities, "holdout": 0,
                      "macro_f05_one_to_one": float("nan"),
                      "macro_f05_greedy": float("nan")}
            print(f"loaded {len(models)} model(s) from {md} (fit skipped)", flush=True)
        else:
            fitted = pm.fit(a.entities, a.max_df, a.max_group, not a.no_calibrate, pool,
                            a.workers, a.top_k or None, a.extra_train,
                            json.loads(a.lgb) if a.lgb else None, n_models=a.models,
                            preranker_entities=a.preranker, pseudo=a.pseudo)
            print(f"held-out macro F0.5 at 0.75: {fitted['macro_f05_threshold'][0.75]:.5f} "
                  f"on {fitted['holdout']:,} entities ({len(fitted['models'])} model(s))",
                  flush=True)
            model_dir = C.MODEL_DIR / a.run_name
            model_dir.mkdir(parents=True, exist_ok=True)
            for j, m in enumerate(fitted["models"]):
                m.save_model(str(model_dir / f"booster_{j}.txt"))
            if fitted.get("preranker"):
                (model_dir / "preranker.txt").write_text(fitted["preranker"], encoding="utf-8")
            with open(model_dir / "calibration.pkl", "wb") as f:
                pickle.dump({"iso": fitted["iso"], "k_global": fitted["k_global"]}, f)

        out_dir = C.OUT_DIR / "submissions" / a.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        admitted, s1_ids, n_pairs = predict_split(
            a.split, fitted, a.max_df, a.max_group, a.shard, pool, a.workers, a.limit,
            out_dir / C.CANDIDATE_FILE, a.dump_scores, a.countries)
    finally:
        if pool is not None:
            pool.close()

    assigned = one_to_one(admitted)
    write_matches(out_dir / C.MATCHING_FILE, s1_ids, assigned)
    n_ids = sum(len(v) for v in assigned.values())
    n_nonempty = sum(1 for v in assigned.values() if v)
    summary = {
        "split": a.split, "run": a.run_name, "rows": len(s1_ids), "limit": a.limit,
        "entities_fit": fitted["entities"], "max_df": a.max_df, "max_group": a.max_group,
        "calibrated": not a.no_calibrate, "top_k": a.top_k, "pseudo": a.pseudo, "lgb": a.lgb,
        "heldout_macro_f05": fitted["macro_f05_one_to_one"],
        "heldout_entities": fitted["holdout"],
        "candidate_pairs": n_pairs, "cand_per_entity": n_pairs / max(len(s1_ids), 1),
        "predicted_ids": n_ids, "ids_per_entity": n_ids / max(len(s1_ids), 1),
        "entities_with_match": n_nonempty, "seconds": time.time() - t0}
    print(json.dumps(summary, indent=2))
    res = C.RESULTS_DIR / a.run_name
    res.mkdir(parents=True, exist_ok=True)
    (res / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if a.limit is None and a.split == "test":
        log_experiment(name=a.run_name, cv=fitted["macro_f05_one_to_one"],
                       pred_file=str(out_dir / C.MATCHING_FILE), who="auto",
                       notes=(f"pair model on address+name+number-word candidates "
                              f"(max_df {a.max_df}), top {a.top_k} by trigram overlap; "
                              f"fitted on {fitted['entities']:,} train entities, held-out score "
                              f"on {fitted['holdout']:,}; {summary['cand_per_entity']:.0f} "
                              f"candidates and {summary['ids_per_entity']:.2f} ids per test entity"
                              + (f"; pseudo-labelled training pairs from {a.pseudo}" if a.pseudo else "")))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--run-name", default="sub02_pair_model")
    p.add_argument("--entities", type=int, default=25000, help="training entities to fit on")
    p.add_argument("--max-df", type=int, default=300)
    p.add_argument("--max-group", type=int, default=200)
    p.add_argument("--shard", type=int, default=50000, help="Source 1 entities per shard")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="only the first N entities (smoke run)")
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--models", type=int, default=5, help="average this many bagged models")
    p.add_argument("--preranker", type=int, default=0,
                   help="train a learned pre-ranker on this many other entities")
    p.add_argument("--extra-train", type=int, default=0,
                   help="entities from outside the sample added to training only")
    p.add_argument("--dump-scores", default=None,
                   help="write every scored pair here, one npz per shard")
    p.add_argument("--load-model", default=None,
                   help="reuse a saved model directory instead of refitting (~30 min saved)")
    p.add_argument("--pseudo", default=None,
                   help="pseudo_labels.py output added to training only")
    p.add_argument("--lgb", default=None, help='LightGBM overrides as JSON, e.g. {"num_leaves": 127}')
    p.add_argument("--countries", nargs="*", default=None,
                   help="score only Source 1 entities of these countries")
    p.add_argument("--top-k", type=int, default=40,
                   help="candidates kept per entity by the pre-ranker (0 keeps all)")
    main(p.parse_args())
