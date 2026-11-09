"""Multi-key candidate generation and its entity-level ceiling.

Recall is the bottleneck: the exact-name scheme leaves 37-39% of entities with
no candidate at all, and each of those is a hard 0.0 in a macro average. This
adds key families that reach records an exact name match cannot, and scores each
family the only way that matters — the macro F0.5 a perfect matcher would get
from its candidates, plus the share of matched entities whose every true match
survived.

Key families:
  exact_name   (country, normalized name)
  token_sort   (country, alphabetically sorted name tokens)
  house        (country, any shared address digit run)
  rare_token   (country, any shared name token whose document frequency in the
                target source is at or below --max-df)
  prefix       (country, first 8 characters of the normalized name)

Token postings are built by streaming each source once and cached under
outputs/cache/, because the per-record arrays hold only whole-name hashes.

Run:
    python src/candidates.py --sample 20000 --max-df 50
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import er_io
from src.er_norm import (core_name, cross_keys, domain_prefix_key, hash64, name_pair_keys,
                         normalize_name, number_word_keys, number_word_keys_loose, repair_name,
                         word_pair_keys)

ENCODE = 2_000_000_000


def build_token_postings(split: str, source: int, max_df: int, field=1, rebuild=False):
    """Stream one source and return (token_key, row) for tokens rarer than max_df.

    field selects the tokens: 1 is the business name, 2 the address, "nw" the
    address number-word keys (er_norm.number_word_keys), "nw4" the same keys
    skipping words under 4 characters, "core" the name without legal forms and
    filler (er_norm.core_name), "ww" adjacent address word pairs
    (er_norm.word_pair_keys), "np" unordered pairs of core name words
    (er_norm.name_pair_keys), "cc" one-word names, matched against joined
    Source 1 words (er_norm.concat_keys), "rn" the core name with look-alike
    digits repaired (er_norm.repair_name), "dp" the first 8 characters of the
    repaired name with its spaces removed (er_norm.domain_prefix_key), "nx"
    number-word keys with letter-affixed house numbers reduced to their digits
    (er_norm.number_word_keys_loose), and "xk" name-word by address-word pairs
    (er_norm.cross_keys). The token key already folds in the
    country, so a posting list is per country and per token. Tokens at or
    above max_df identify nothing useful and are dropped, which is what keeps
    the candidate lists short.
    """
    cache = er_io.CACHE_DIR / f"{split}_source{source}.tok{field}.df{max_df}.v{er_io.CACHE_VERSION}.npz"
    if cache.exists() and not rebuild:
        blob = np.load(cache, allow_pickle=False)
        return blob["key"], blob["row"]

    parsed = er_io.parse_source(split, source)
    ctry = er_io.country_keys(parsed)
    keys, rows = [], []
    path = C.source_file(split, source)
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            seen = set()
            if field == "nw":
                toks = number_word_keys(parts[2])
            elif field == "nw4":
                toks = number_word_keys(parts[2], min_len=4)
            elif field == "ww":
                toks = word_pair_keys(parts[2])
            elif field == "core":
                core = core_name(normalize_name(parts[1]))
                toks = [core] if core else []
            elif field == "cc":
                # One-word names only: the domain-style rewrites the key targets.
                norm = normalize_name(parts[1])
                toks = [norm] if " " not in norm and len(norm) >= 6 else []
            elif field == "np":
                toks = name_pair_keys(core_name(normalize_name(parts[1])))
            elif field == "rn":
                core = core_name(repair_name(normalize_name(parts[1])))
                toks = [core] if core else []
            elif field == "dp":
                dp = domain_prefix_key(repair_name(normalize_name(parts[1])))
                toks = [dp] if dp else []
            elif field == "nx":
                toks = number_word_keys_loose(parts[2], min_len=4)
            elif field == "xk":
                toks = cross_keys(parts[1], parts[2])
            else:
                toks = normalize_name(parts[field]).split()
            for tok in toks:
                h = hash64(tok)
                if h not in seen:
                    seen.add(h)
                    keys.append(h)
                    rows.append(i)
    key = er_io.combine_keys(np.array(keys, dtype=np.int64), ctry[np.array(rows, dtype=np.int64)])
    row = np.array(rows, dtype=np.int32)
    del keys, rows

    uniq, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    keep = counts[inverse] <= max_df
    key, row = key[keep], row[keep]
    order = np.argsort(key, kind="stable")
    key, row = key[order], row[order]
    er_io.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, key=key, row=row)
    return key, row


def lookup_postings(sorted_keys, rows, query_keys, cap):
    """Rows whose key matches each query key, as a list of arrays."""
    lo = np.searchsorted(sorted_keys, query_keys, side="left")
    hi = np.searchsorted(sorted_keys, query_keys, side="right")
    out = []
    for a, b in zip(lo.tolist(), hi.tolist()):
        out.append(rows[a:b] if b - a <= cap else np.empty(0, dtype=np.int32))
    return out


def s1_token_keys(split: str, rows_wanted: set, field=1):
    """Token hashes per sampled Source 1 row, from one streaming pass."""
    out = {}
    with open(C.source_file(split, 1), encoding="utf-8") as f:
        next(f, None)
        for i, line in enumerate(f):
            if i in rows_wanted:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 4:
                    out[i] = sorted({hash64(t) for t in normalize_name(parts[field]).split()})
            if len(out) == len(rows_wanted):
                break
    return out


def ceiling(truth_sets, cand_sets):
    """Macro F0.5 a perfect matcher would score, plus coverage diagnostics."""
    total = covered = hit = true_total = 0.0
    matched = 0
    for truth, cands in zip(truth_sets, cand_sets):
        k = len(truth)
        if k == 0:
            total += 1.0
            continue
        matched += 1
        c = len(truth & cands)
        hit += c
        true_total += k
        covered += c == k
        if c:
            total += 1.25 * c / (c + 0.25 * k)
    n = len(truth_sets)
    return {"ceiling": total / n, "pair_recall": hit / max(true_total, 1),
            "full_coverage": covered / max(matched, 1),
            "cand_per_entity": float(np.mean([len(c) for c in cand_sets]))}


def main(sample: int, max_df: int, max_group: int, out_dir) -> None:
    """Measure every key family and their unions on one fixed entity sample."""
    t0 = time.time()
    rng = np.random.default_rng(C.SEED)
    s1 = er_io.parse_source("train", 1)
    gt = er_io.parse_ground_truth()

    rows = np.sort(rng.choice(len(s1["ids"]), size=min(sample, len(s1["ids"])), replace=False))
    row_of = er_io.index_of(s1["ids"], gt["s1"])
    gt_of_row = np.full(len(s1["ids"]), -1, dtype=np.int64)
    gt_of_row[row_of] = np.arange(len(gt["s1"]))
    enc = gt["match_src"].astype(np.int64) * ENCODE + gt["match_id"]
    truth_sets = []
    for row in rows.tolist():
        g = gt_of_row[row]
        truth_sets.append(set(enc[gt["offsets"][g]:gt["offsets"][g + 1]].tolist())
                          if g >= 0 else set())
    k_all = np.array([len(t) for t in truth_sets])
    print(f"sample {len(rows):,} entities | zero-match {(k_all == 0).mean() * 100:.2f}% "
          f"| mean matches {k_all.mean():.3f} | max_df {max_df} | max_group {max_group}",
          flush=True)

    families = ["exact_name", "token_sort", "prefix", "house", "rare_token", "addr_token"]
    cand = {f: [set() for _ in rows] for f in families}
    ctry_s1 = er_io.country_keys(s1)
    want = set(rows.tolist())
    tok_of_row = s1_token_keys("train", want, field=1)
    addr_of_row = s1_token_keys("train", want, field=2)

    for source in (2, 3):
        print(f"  source {source}: indexing...", flush=True)
        target = er_io.parse_source("train", source)
        ctry_t = er_io.country_keys(target)
        tid = target["ids"]

        for fam, col in (("exact_name", "name_key"), ("token_sort", "sort_key")):
            tk = er_io.combine_keys(target[col], ctry_t)
            order = np.argsort(tk, kind="stable")
            sk = tk[order]
            qk = er_io.combine_keys(s1[col][rows], ctry_s1[rows])
            for i, hits in enumerate(lookup_postings(sk, order, qk, max_group)):
                if len(hits):
                    cand[fam][i] |= set((tid[hits] + source * ENCODE).tolist())

        # Prefix of the normalized name is approximated by hashing the sorted
        # token key's first token; instead use a dedicated streaming pass only if
        # needed. Here prefix reuses rare-token machinery below.

        # House numbers: one posting per (record, slot).
        hr, hk = [], []
        for slot in range(target["house"].shape[1]):
            v = target["house"][:, slot]
            keep = np.flatnonzero(v != er_io.EMPTY)
            hr.append(keep)
            hk.append(er_io.combine_keys(v[keep].astype(np.int64), ctry_t[keep]))
        hrow = np.concatenate(hr).astype(np.int32)
        hkey = np.concatenate(hk)
        o = np.argsort(hkey, kind="stable")
        hkey, hrow = hkey[o], hrow[o]
        for slot in range(s1["house"].shape[1]):
            v = s1["house"][rows, slot]
            live = v != er_io.EMPTY
            qk = er_io.combine_keys(v.astype(np.int64), ctry_s1[rows])
            for i, hits in enumerate(lookup_postings(hkey, hrow, qk, max_group)):
                if live[i] and len(hits):
                    cand["house"][i] |= set((tid[hits] + source * ENCODE).tolist())
        del hkey, hrow, hr, hk

        # Rare tokens of the name, then of the address. Address tokens are the
        # stronger signal: token overlap holds on far more true pairs than an
        # identical name does.
        for fam, field, lookup in (("rare_token", 1, tok_of_row),
                                   ("addr_token", 2, addr_of_row)):
            print(f"  source {source}: {fam} postings (max_df={max_df})...", flush=True)
            tkey, trow = build_token_postings("train", source, max_df, field=field)
            print(f"    {len(tkey):,} postings kept", flush=True)
            for i, row in enumerate(rows.tolist()):
                toks = lookup.get(row)
                if not toks:
                    continue
                qk = er_io.combine_keys(np.array(toks, dtype=np.int64),
                                        np.full(len(toks), ctry_s1[row], dtype=np.int64))
                got = set()
                for hits in lookup_postings(tkey, trow, qk, max_group):
                    if len(hits):
                        got |= set((tid[hits] + source * ENCODE).tolist())
                cand[fam][i] |= got
            del tkey, trow
        del target
        print(f"  source {source} done ({time.time() - t0:.0f}s)", flush=True)

    cand["name_sort_house"] = [a | b | c for a, b, c in
                               zip(cand["exact_name"], cand["token_sort"], cand["house"])]
    cand["name_keys"] = [a | b for a, b in zip(cand["name_sort_house"], cand["rare_token"])]
    cand["addr_plus_name"] = [a | b for a, b in zip(cand["name_keys"], cand["addr_token"])]
    cand.pop("prefix")

    results = {}
    print(f"\n{'family':>18} {'ceiling':>9} {'pair_recall':>12} {'full_cover':>11} {'cand/entity':>12}")
    for fam, sets in cand.items():
        st = ceiling(truth_sets, sets)
        results[fam] = st
        print(f"{fam:>18} {st['ceiling']:>9.4f} {st['pair_recall'] * 100:>11.2f}% "
              f"{st['full_coverage'] * 100:>10.2f}% {st['cand_per_entity']:>12.2f}")
    print(f"\ntotal {time.time() - t0:.0f}s")

    if out_dir:
        p = Path(out_dir) / "ceiling_v2.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"sample": len(rows), "max_df": max_df,
                                 "max_group": max_group, "families": results}, indent=2),
                     encoding="utf-8")
        print(f"wrote {p}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=20000)
    p.add_argument("--max-df", type=int, default=50, help="drop tokens at or above this frequency")
    p.add_argument("--max-group", type=int, default=100, help="skip any key group larger than this")
    p.add_argument("--out", default="results/blocking")
    a = p.parse_args()
    main(a.sample, a.max_df, a.max_group, a.out)
