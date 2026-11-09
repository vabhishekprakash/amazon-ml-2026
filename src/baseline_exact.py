"""Exact-name baseline for entity resolution.

Rule: a Source 1 record matches a Source 2/3 record when both share a country
and an identical normalized name, and either that name is unique in the target
source for that country, or an address number (house number or postcode) also
agrees. Every other entity gets an empty list.

Nothing in the rule is fitted, so scoring it on the full training set is an
honest estimate rather than an optimistic one. The candidate set is identical to
the match set: the rule does no post-filtering.

Run:
    python src/baseline_exact.py --split train --score
    python src/baseline_exact.py --split test --run-name sub01_exact_name
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import er_io
from src.er_norm import house_numbers, postcodes
from src.log_experiment import log_experiment
from src.metrics import f05_entity

# Entity numbers stay below 1e9, so this keeps S2 and S3 ids in disjoint ranges.
ENCODE_OFFSET = 2_000_000_000


def candidate_groups(target: dict):
    """Index the target source by (country, normalized name).

    Returns the sorted unique keys, the group start offsets, and the row indices
    ordered by key, so every row sharing a key is one contiguous slice.
    """
    keys = er_io.combine_keys(target["name_key"], er_io.country_keys(target))
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    uniq, starts = np.unique(sorted_keys, return_index=True)
    return uniq, np.append(starts, len(sorted_keys)), order


def expand_groups(rows, pos, bounds, order, block=200_000):
    """Yield (s1_row, target_row) pair blocks for the named groups.

    Expansion is vectorized and chunked: the full ambiguous set is tens of
    millions of pairs, which is too large to materialize at once.
    """
    for start in range(0, len(rows), block):
        chunk = rows[start:start + block]
        lo = bounds[pos[chunk]]
        counts = bounds[pos[chunk] + 1] - lo
        if not counts.sum():
            continue
        s1_idx = np.repeat(chunk, counts)
        # Offset of each pair within its own group.
        starts = np.repeat(np.cumsum(counts) - counts, counts)
        within = np.arange(counts.sum(), dtype=np.int64) - starts
        yield s1_idx, order[np.repeat(lo, counts) + within]


def match_one_source(split: str, s1: dict, target: dict, source: int, verbose=True):
    """Apply the rule against one target source.

    Returns (s1_row, target_row) arrays for accepted matches.
    """
    uniq, bounds, order = candidate_groups(target)
    s1_keys = er_io.combine_keys(s1["name_key"], er_io.country_keys(s1))

    pos = np.clip(np.searchsorted(uniq, s1_keys), 0, len(uniq) - 1)
    found = uniq[pos] == s1_keys
    # A name that normalizes to nothing would otherwise match every blank name.
    found &= s1["name_key"] != 0
    group_size = np.where(found, bounds[pos + 1] - bounds[pos], 0)

    unique_hit = found & (group_size == 1)
    multi_hit = found & (group_size > 1)
    if verbose:
        print(f"  S{source}: {int(found.sum()):,} entities hit a name group "
              f"({int(unique_hit.sum()):,} unique, {int(multi_hit.sum()):,} ambiguous)")

    # A name unique in the target source for that country is accepted outright.
    keep_s1 = [np.flatnonzero(unique_hit)]
    keep_tgt = [order[bounds[pos[unique_hit]]]]

    # An ambiguous name needs an address number to agree. Both sides are
    # compared through the precomputed slot arrays, so no text is re-read.
    amb = np.flatnonzero(multi_hit)
    checked = kept = 0
    for s1_idx, tgt_idx in expand_groups(amb, pos, bounds, order):
        agree = (er_io.slots_intersect(s1["house"][s1_idx], target["house"][tgt_idx]) |
                 er_io.slots_intersect(s1["pcode"][s1_idx], target["pcode"][tgt_idx]))
        checked += len(s1_idx)
        kept += int(agree.sum())
        if agree.any():
            keep_s1.append(s1_idx[agree])
            keep_tgt.append(tgt_idx[agree])
    if verbose and checked:
        print(f"      {checked:,} ambiguous pairs checked on address numbers, kept {kept:,} "
              f"({kept / checked * 100:.2f}%)")
    return np.concatenate(keep_s1), np.concatenate(keep_tgt)


def predict(split: str, verbose=True) -> dict:
    """Run the rule for both target sources.

    Returns {s1_row_index: [encoded match ids]} where an encoded id is
    source * OFFSET + numeric id, so S2 and S3 never collide.
    """
    s1 = er_io.parse_source(split, 1)
    preds = {}
    for source in (2, 3):
        target = er_io.parse_source(split, source)
        rows, tgt = match_one_source(split, s1, target, source, verbose)
        ids = target["ids"][tgt]
        for row, num in zip(rows.tolist(), ids.tolist()):
            preds.setdefault(row, []).append((source, num))
    if verbose:
        n_pred = sum(1 for v in preds.values() if v)
        total = sum(len(v) for v in preds.values())
        print(f"  entities with at least one match: {n_pred:,} of {len(s1['ids']):,} "
              f"({n_pred / len(s1['ids']) * 100:.2f}%) | predicted ids: {total:,}")
    return preds, s1


def score_on_train(preds: dict, s1: dict) -> dict:
    """Macro F0.5 of the rule over every training Source 1 entity."""
    gt = er_io.parse_ground_truth()
    row_of = er_io.index_of(s1["ids"], gt["s1"])
    sizes = np.diff(gt["offsets"])
    total, n = 0.0, len(gt["s1"])
    empty_right = empty_wrong = 0
    scored_nonempty = []
    # Encode (source, id) as one int so set operations stay cheap over 2.2M entities.
    enc = (gt["match_src"].astype(np.int64) * ENCODE_OFFSET + gt["match_id"]).tolist()
    offsets = gt["offsets"].tolist()
    rows = row_of.tolist()
    for i in range(n):
        truth = set(enc[offsets[i]:offsets[i + 1]])
        got = preds.get(rows[i]) if rows[i] >= 0 else None
        pred = {src * ENCODE_OFFSET + num for src, num in got} if got else set()
        s = f05_entity(truth, pred)
        total += s
        if not truth:
            empty_right += not pred
            empty_wrong += bool(pred)
        else:
            scored_nonempty.append(s)
    macro = total / n
    nonempty = float(np.mean(scored_nonempty)) if scored_nonempty else 0.0
    print(f"\n  macro F0.5 over all {n:,} entities: {macro:.5f}")
    print(f"  singletons: {empty_right:,} correct (1.0), {empty_wrong:,} wrong (0.0)")
    print(f"  mean F0.5 on the {len(scored_nonempty):,} entities that do have matches: {nonempty:.5f}")
    print(f"  mean true matches per entity: {sizes.mean():.3f}")
    return {"macro_f05": macro, "entities": int(n), "nonempty_mean_f05": nonempty,
            "singletons_correct": int(empty_right), "singletons_wrong": int(empty_wrong)}


def write_tsv(path: Path, s1: dict, preds: dict, column: str) -> int:
    """Write one row per Source 1 entity, empty list when there is no match."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = s1["ids"].tolist()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{column}\n")
        for row, num in enumerate(ids):
            got = preds.get(row)
            joined = ",".join(f"S{src}-{mid}" for src, mid in got) if got else ""
            f.write(f"S1-{num}\t{joined}\n")
    return len(ids)


def main(split: str, run_name: str, do_score: bool) -> None:
    """Predict, optionally score on train, and write the two submission TSVs."""
    print(f"exact-name baseline on {split}")
    preds, s1 = predict(split)

    metrics = {"split": split, "run": run_name}
    if do_score and split == "train":
        metrics.update(score_on_train(preds, s1))

    out_dir = C.OUT_DIR / "submissions" / run_name
    n = write_tsv(out_dir / C.MATCHING_FILE, s1, preds, C.GT_MATCH_COL)
    write_tsv(out_dir / C.CANDIDATE_FILE, s1, preds, C.CANDIDATE_COL)
    print(f"\n  wrote {n:,} rows to {out_dir}")
    metrics["rows"] = n

    res = C.RESULTS_DIR / run_name
    res.mkdir(parents=True, exist_ok=True)
    (res / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  wrote {res / 'metrics.json'}")

    if do_score and split == "train":
        log_experiment(name=run_name, cv=metrics["macro_f05"],
                       pred_file=str(out_dir / C.MATCHING_FILE), who="auto",
                       notes=("exact normalized name within country, accepted when unique in "
                              "target source or an address number agrees; unfitted rule scored "
                              "on full train"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--run-name", default="sub01_exact_name")
    p.add_argument("--score", action="store_true", help="score on train (train split only)")
    a = p.parse_args()
    main(a.split, a.run_name, a.score)
