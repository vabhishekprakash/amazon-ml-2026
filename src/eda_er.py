"""Data report for the entity resolution dataset.

Run:
    python src/eda_er.py                          # full data, writes results/eda/
    python src/eda_er.py --sample 200000          # first N rows per file, quick pass
    python src/eda_er.py --out results/eda

Reports per-source counts and country coverage, the match-count distribution per
Source 1 entity, the Source 2 / Source 3 split of true matches, how often true
matches share a normalized name, how ambiguous those names are inside a country,
and blocking recall for name-key candidate generation (per entity and per pair).

Records are held as int64 arrays (see src/er_io) so the full 24M rows fit in
memory. All paths come from config.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import er_io
from src.er_norm import house_numbers, normalize_name, postcodes


def section(title: str) -> None:
    print("\n" + "=" * 74 + f"\n{title}\n" + "=" * 74)


def report_sources(frames: dict, split: str, out: dict) -> None:
    """Row counts, country coverage and unusable names for each source."""
    section(f"{split.upper()}: SOURCES")
    print(f"{'source':>7} {'records':>12} {'blank_name':>11} {'uniq_name_keys':>15}  countries")
    for src, d in frames.items():
        labels = d["countries"]
        counts = Counter(labels[c] for c in d["country"])
        uniq = len(np.unique(d["name_key"]))
        print(f"{src:>7} {len(d['ids']):>12,} {d['n_blank_name']:>11,} {uniq:>15,}  "
              + ", ".join(f"{k}={v:,}" for k, v in sorted(counts.items())))
        out[f"{split}_source{src}"] = {"records": int(len(d["ids"])),
                                       "blank_name": int(d["n_blank_name"]),
                                       "unique_name_keys": int(uniq),
                                       "countries": {k: int(v) for k, v in counts.items()}}


def report_ground_truth(gt: dict, out: dict) -> None:
    """Match-count distribution per Source 1 entity and the S2/S3 split."""
    section("TRAIN: GROUND TRUTH")
    sizes = np.diff(gt["offsets"])
    n = len(sizes)
    zero = int((sizes == 0).sum())
    print(f"source 1 entities:      {n:,}")
    print(f"zero-match (singleton): {zero:,} ({zero / n * 100:.2f}%)")
    print(f"exactly one match:      {int((sizes == 1).sum()):,} ({(sizes == 1).mean() * 100:.2f}%)")
    print(f"two or more:            {int((sizes >= 2).sum()):,} ({(sizes >= 2).mean() * 100:.2f}%)")
    print(f"matches per entity: mean={sizes.mean():.3f} p50={np.median(sizes):.0f} "
          f"p95={np.percentile(sizes, 95):.0f} max={sizes.max()}")
    hist = Counter(sizes.tolist())
    print("histogram:", {k: hist[k] for k in sorted(hist)})
    s2 = int((gt["match_src"] == 2).sum())
    s3 = int((gt["match_src"] == 3).sum())
    print(f"matched ids: {s2 + s3:,}  (S2 {s2:,} = {s2 / (s2 + s3) * 100:.2f}%, "
          f"S3 {s3:,} = {s3 / (s2 + s3) * 100:.2f}%)")

    print("\nWhat a partial answer is worth (F0.5 for k true matches, all correct, none extra):")
    for k in range(1, 8):
        print(f"  k={k}: predict 1 -> {1.25 * (1 / k) / (0.25 + 1 / k):.3f}   "
              f"predict all {k} -> 1.000   predict nothing -> 0.000")
    out["ground_truth"] = {"entities": int(n), "zero_match_frac": zero / n,
                           "mean_matches": float(sizes.mean()), "max_matches": int(sizes.max()),
                           "s2_share": s2 / (s2 + s3)}


def build_pairs(gt: dict, frames: dict, out: dict) -> dict:
    """Resolve ground-truth matches to row indices in each source.

    Returns arrays of (s1_row, target_row, target_source) for every true pair
    that can be located, so later sections can compare their fields.
    """
    section("TRAIN: RESOLVING TRUE PAIRS TO ROWS")
    s1_row = er_io.index_of(frames[1]["ids"], gt["s1"])
    missing = int((s1_row < 0).sum())
    print(f"ground-truth S1 ids not found in train_source1: {missing:,}")

    sizes = np.diff(gt["offsets"])
    owner = np.repeat(np.arange(len(gt["s1"])), sizes)  # which S1 entity each match belongs to
    pairs = {}
    for src in (2, 3):
        pick = gt["match_src"] == src
        tgt_row = er_io.index_of(frames[src]["ids"], gt["match_id"][pick])
        lost = int((tgt_row < 0).sum())
        print(f"S{src} matched ids not found in train_source{src}: {lost:,}")
        pairs[src] = {"s1_row": s1_row[owner[pick]], "tgt_row": tgt_row}
    out["pair_resolution"] = {"s1_missing": missing}
    return pairs


def report_name_agreement(pairs: dict, frames: dict, out: dict) -> None:
    """How often a true pair shares a normalized name or a token-sorted name."""
    section("TRAIN: DO TRUE MATCHES SHARE A NAME?")
    rows = []
    for src, p in pairs.items():
        ok = (p["s1_row"] >= 0) & (p["tgt_row"] >= 0)
        a, b = p["s1_row"][ok], p["tgt_row"][ok]
        same_name = frames[1]["name_key"][a] == frames[src]["name_key"][b]
        same_sort = frames[1]["sort_key"][a] == frames[src]["sort_key"][b]
        same_ctry = er_io.country_keys(frames[1])[a] == er_io.country_keys(frames[src])[b]
        rows.append({"source": src, "pairs": len(a),
                     "same_norm_name": float(same_name.mean()),
                     "same_sorted_name": float(same_sort.mean()),
                     "same_country": float(same_ctry.mean())})
    for r in rows:
        print(f"  S1-S{r['source']}: {r['pairs']:,} pairs | identical normalized name "
              f"{r['same_norm_name'] * 100:.2f}% | identical token-sorted name "
              f"{r['same_sorted_name'] * 100:.2f}% | same country {r['same_country'] * 100:.2f}%")
    print("\n  Exact-name recall is the ceiling for the exact-match baseline.")
    print("  Token-sorted recall shows what fixing word order alone would buy.")
    print("  Blocking within country is only safe if same-country is ~100%.")
    out["name_agreement"] = rows


def report_ambiguity(frames: dict, out: dict) -> None:
    """How often a normalized name is reused inside one country and source."""
    section("TRAIN: NAME AMBIGUITY INSIDE A COUNTRY")
    stats = {}
    for src, d in frames.items():
        # Fold country into the key so counting is per (country, name).
        sizes = er_io.group_sizes(er_io.combine_keys(d["name_key"], er_io.country_keys(d)))
        share = float((sizes > 1).mean())
        print(f"  source {src}: {share * 100:.2f}% of records share their "
              f"(country, name) with another record | max group {sizes.max():,}")
        stats[f"source{src}"] = {"non_unique_share": share, "max_group": int(sizes.max())}
    print("\n  The baseline only accepts an exact-name hit when the name is unique in")
    print("  the target source for that country, or an address number agrees.")
    out["ambiguity"] = stats


def report_blocking(pairs: dict, frames: dict, out: dict) -> None:
    """Recall and candidate volume for name-key blocking inside a country."""
    section("TRAIN: BLOCKING RECALL (exact name key, within country)")
    rows = []
    for key_name in ("name_key", "sort_key"):
        for src in (2, 3):
            p = pairs[src]
            ok = (p["s1_row"] >= 0) & (p["tgt_row"] >= 0)
            a, b = p["s1_row"][ok], p["tgt_row"][ok]
            s1_keys = er_io.combine_keys(frames[1][key_name], er_io.country_keys(frames[1]))
            tgt_keys = er_io.combine_keys(frames[src][key_name], er_io.country_keys(frames[src]))
            hit = s1_keys[a] == tgt_keys[b]
            # Candidate volume: group size of each S1 key in the target source.
            uniq, counts = np.unique(tgt_keys, return_counts=True)
            pos = np.searchsorted(uniq, s1_keys)
            pos = np.clip(pos, 0, len(uniq) - 1)
            vol = np.where(uniq[pos] == s1_keys, counts[pos], 0)
            rows.append({"key": key_name, "target": f"S{src}",
                         "pair_recall": float(hit.mean()),
                         "cand_per_entity_mean": float(vol.mean()),
                         "entities_with_0_cand": float((vol == 0).mean())})
    print(f"{'key':>10} {'target':>7} {'pair_recall':>12} {'cand/entity':>12} {'no_cand':>9}")
    for r in rows:
        print(f"{r['key']:>10} {r['target']:>7} {r['pair_recall'] * 100:>11.2f}% "
              f"{r['cand_per_entity_mean']:>12.2f} {r['entities_with_0_cand'] * 100:>8.2f}%")
    print("\n  pair_recall here is the share of true pairs whose two records share the key.")
    print("  A pair missing from candidates can never be recovered downstream.")
    out["blocking"] = rows


def report_address_signal(pairs: dict, frames: dict, sample_n: int, out: dict) -> None:
    """How often address numbers agree on true pairs, sampled to bound memory."""
    section(f"TRAIN: ADDRESS SIGNAL (sample of {sample_n:,} true pairs per source)")
    rng = np.random.default_rng(C.SEED)
    stats = {}
    for src in (2, 3):
        p = pairs[src]
        ok = (p["s1_row"] >= 0) & (p["tgt_row"] >= 0)
        a, b = p["s1_row"][ok], p["tgt_row"][ok]
        take = rng.choice(len(a), size=min(sample_n, len(a)), replace=False)
        left = er_io.fetch_fields("train", 1, a[take])
        right = er_io.fetch_fields("train", src, b[take])
        has_pc = house_hit = pc_hit = both = n = 0
        for la, rb in zip(a[take], b[take]):
            if la not in left or rb not in right:
                continue
            n += 1
            ha, hb = house_numbers(left[la][1]), house_numbers(right[rb][1])
            pa, pb = postcodes(left[la][1]), postcodes(right[rb][1])
            has_pc += bool(pa or pb)
            house_hit += bool(ha & hb)
            pc_hit += bool(pa & pb)
            both += bool((ha & hb) or (pa & pb))
        if n:
            print(f"  S1-S{src} ({n:,} sampled pairs): a postcode is present on either side "
                  f"{has_pc / n * 100:.2f}% | house number agrees {house_hit / n * 100:.2f}% | "
                  f"postcode agrees {pc_hit / n * 100:.2f}% | either agrees {both / n * 100:.2f}%")
            stats[f"S{src}"] = {"sampled": n, "postcode_present": has_pc / n,
                                "house_agree": house_hit / n, "postcode_agree": pc_hit / n,
                                "either_agree": both / n}
    out["address_signal"] = stats


def report_test(out: dict) -> None:
    """Test-side counts, the required row count, and countries unseen in train."""
    section("TEST")
    test = {s: er_io.parse_source("test", s) for s in C.SOURCES}
    report_sources(test, "test", out)
    train_countries = set()
    for s in C.SOURCES:
        train_countries |= set(er_io.parse_source("train", s)["countries"])
    test_countries = set()
    for s in C.SOURCES:
        test_countries |= set(test[s]["countries"])
    print(f"\ncountries in train: {sorted(train_countries)}")
    print(f"countries only in test: {sorted(test_countries - train_countries)}")
    print(f"submission rows required: {len(test[1]['ids']):,}")
    out["test"] = {"submission_rows": int(len(test[1]["ids"])),
                   "unseen_countries": sorted(test_countries - train_countries)}


def main(sample, out_dir, addr_sample) -> None:
    """Load every split and print the full report, optionally writing json."""
    limit = sample
    if limit:
        print(f"SAMPLE MODE: first {limit:,} rows per file. "
              "Recall and ambiguity numbers are NOT comparable to a full run.")
    summary = {"sample": limit}
    train = {s: er_io.parse_source("train", s, limit=limit) for s in C.SOURCES}
    report_sources(train, "train", summary)
    gt = er_io.parse_ground_truth(limit=limit)
    report_ground_truth(gt, summary)
    pairs = build_pairs(gt, train, summary)
    report_name_agreement(pairs, train, summary)
    report_ambiguity(train, summary)
    report_blocking(pairs, train, summary)
    report_address_signal(pairs, train, addr_sample, summary)
    if not limit:
        report_test(summary)

    if out_dir:
        path = Path(out_dir) / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None, help="read only the first N rows per file")
    p.add_argument("--addr-sample", type=int, default=20000, help="true pairs sampled for address stats")
    p.add_argument("--out", default=None, help="directory for summary.json, e.g. results/eda")
    a = p.parse_args()
    main(a.sample, a.out, a.addr_sample)
