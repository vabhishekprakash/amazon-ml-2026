"""Entity-level ceiling of each candidate-generation scheme.

Pair-level recall does not translate into the competition score: scoring is
F0.5 per Source 1 entity, macro-averaged, and an entity with no candidates
contributes a hard 0.0 while an entity with no true matches contributes 1.0 for
free. This measures the score a *perfect* matcher would get from each scheme,
which is the real upper bound on that scheme.

For one entity, the best achievable prediction is exactly the true matches that
survived blocking, giving c = |truth AND candidates|, n = c, k = |truth| and
F = 1.25c / (c + 0.25k). An entity with k = 0 scores 1.0 by predicting nothing.

Run:
    python src/blocking_ceiling.py --sample 20000
    python src/blocking_ceiling.py --sample 20000 --out results/blocking
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import er_io

ENCODE = 2_000_000_000


def group_index(keys: np.ndarray):
    """Sort rows by key and return (unique keys, group bounds, row order)."""
    order = np.argsort(keys, kind="stable")
    uniq, starts = np.unique(keys[order], return_index=True)
    return uniq, np.append(starts, len(order)), order


def candidates_by_key(s1_keys, target_keys, target_ids, source, rows, max_group):
    """Candidate encoded ids per sampled row, for one key definition.

    Groups larger than max_group are skipped: a key shared by thousands of
    records identifies nothing and would dominate the candidate budget.
    """
    uniq, bounds, order = group_index(target_keys)
    out = [[] for _ in rows]
    pos = np.clip(np.searchsorted(uniq, s1_keys[rows]), 0, len(uniq) - 1)
    hit = uniq[pos] == s1_keys[rows]
    for i, (p, ok) in enumerate(zip(pos.tolist(), hit.tolist())):
        if not ok:
            continue
        lo, hi = int(bounds[p]), int(bounds[p + 1])
        if hi - lo > max_group:
            continue
        out[i] = (target_ids[order[lo:hi]] + source * ENCODE).tolist()
    return out


def house_candidates(s1, target, source, rows, max_group):
    """Candidates sharing a country and at least one address number."""
    ctry_t = er_io.country_keys(target)
    ctry_s = er_io.country_keys(s1)
    # One row per (record, house slot) on the target side.
    slot_rows, slot_vals = [], []
    for slot in range(target["house"].shape[1]):
        vals = target["house"][:, slot]
        keep = np.flatnonzero(vals != er_io.EMPTY)
        slot_rows.append(keep)
        slot_vals.append(er_io.combine_keys(vals[keep].astype(np.int64), ctry_t[keep]))
    flat_rows = np.concatenate(slot_rows)
    flat_keys = np.concatenate(slot_vals)
    uniq, bounds, order = group_index(flat_keys)

    out = [[] for _ in rows]
    for slot in range(s1["house"].shape[1]):
        vals = s1["house"][rows, slot]
        live = vals != er_io.EMPTY
        keys = er_io.combine_keys(vals.astype(np.int64), ctry_s[rows])
        pos = np.clip(np.searchsorted(uniq, keys), 0, len(uniq) - 1)
        hit = live & (uniq[pos] == keys)
        for i, (p, ok) in enumerate(zip(pos.tolist(), hit.tolist())):
            if not ok:
                continue
            lo, hi = int(bounds[p]), int(bounds[p + 1])
            if hi - lo > max_group:
                continue
            out[i].extend((target["ids"][flat_rows[order[lo:hi]]] + source * ENCODE).tolist())
    return out


def ceiling(truth_sets, cand_sets):
    """Macro F0.5 a perfect matcher would score given these candidates."""
    total = 0.0
    covered = hit_pairs = true_pairs = 0
    for truth, cands in zip(truth_sets, cand_sets):
        k = len(truth)
        if k == 0:
            total += 1.0  # predict nothing, score 1.0
            continue
        c = len(truth & cands)
        true_pairs += k
        hit_pairs += c
        covered += c == k
        if c:
            total += 1.25 * c / (c + 0.25 * k)
    n = len(truth_sets)
    return {"entity_macro_ceiling": total / n,
            "pair_recall": hit_pairs / max(true_pairs, 1),
            "entities_fully_covered": covered / max(sum(1 for t in truth_sets if t), 1)}


def main(sample: int, max_group: int, out_dir) -> None:
    """Score every scheme on one fixed entity sample and print a ranking."""
    rng = np.random.default_rng(C.SEED)
    s1 = er_io.parse_source("train", 1)
    gt = er_io.parse_ground_truth()
    row_of = er_io.index_of(s1["ids"], gt["s1"])

    rows = np.sort(rng.choice(len(s1["ids"]), size=min(sample, len(s1["ids"])), replace=False))
    gt_of_row = np.full(len(s1["ids"]), -1, dtype=np.int64)
    gt_of_row[row_of] = np.arange(len(gt["s1"]))
    enc = (gt["match_src"].astype(np.int64) * ENCODE + gt["match_id"])
    truth_sets = []
    for row in rows.tolist():
        g = gt_of_row[row]
        truth_sets.append(set(enc[gt["offsets"][g]:gt["offsets"][g + 1]].tolist())
                          if g >= 0 else set())
    k_all = np.array([len(t) for t in truth_sets])
    print(f"sample: {len(rows):,} entities | zero-match {int((k_all == 0).sum()):,} "
          f"({(k_all == 0).mean() * 100:.2f}%) | mean matches {k_all.mean():.3f}")
    print(f"max group size allowed per key: {max_group:,}\n")

    ctry_s1 = er_io.country_keys(s1)
    schemes = {}
    for source in (2, 3):
        target = er_io.parse_source("train", source)
        ctry_t = er_io.country_keys(target)
        for label, col in (("exact_name", "name_key"), ("token_sort", "sort_key")):
            schemes.setdefault(label, [set() for _ in rows])
            got = candidates_by_key(er_io.combine_keys(s1[col], ctry_s1),
                                    er_io.combine_keys(target[col], ctry_t),
                                    target["ids"], source, rows, max_group)
            for i, ids in enumerate(got):
                schemes[label][i] |= set(ids)
        schemes.setdefault("house_number", [set() for _ in rows])
        for i, ids in enumerate(house_candidates(s1, target, source, rows, max_group)):
            schemes["house_number"][i] |= set(ids)

    schemes["name_or_sort"] = [a | b for a, b in zip(schemes["exact_name"], schemes["token_sort"])]
    schemes["all_three"] = [a | b for a, b in zip(schemes["name_or_sort"], schemes["house_number"])]

    results = {}
    print(f"{'scheme':>16} {'ceiling':>9} {'pair_recall':>12} {'full_cover':>11} {'cand/entity':>12}")
    for label, cands in schemes.items():
        stats = ceiling(truth_sets, cands)
        stats["cand_per_entity"] = float(np.mean([len(c) for c in cands]))
        results[label] = stats
        print(f"{label:>16} {stats['entity_macro_ceiling']:>9.4f} "
              f"{stats['pair_recall'] * 100:>11.2f}% {stats['entities_fully_covered'] * 100:>10.2f}% "
              f"{stats['cand_per_entity']:>12.2f}")
    print("\nceiling = macro F0.5 a perfect matcher would score from these candidates")
    print("full_cover = share of matched entities whose every true match survived blocking")

    if out_dir:
        path = Path(out_dir) / "ceiling.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"sample": len(rows), "max_group": max_group,
                                    "schemes": results}, indent=2), encoding="utf-8")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=20000)
    p.add_argument("--max-group", type=int, default=100)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    main(a.sample, a.max_group, a.out)
