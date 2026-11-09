"""What the new blocking key families reach that the current ones never do.

Draws the held-out sample src/loss_breakdown.py works on, finds the true pairs
no current blocking key reaches, and asks of each new family, alone and
together:

  recovered   share of those never-blocked pairs the family now reaches
  added       raw candidates it adds per entity, against the current lists
  survives    share of the recovered pairs still there after the top-k
              pre-ranker, which is the last blocking stage

A family that recovers pairs the pre-ranker then drops has bought nothing, so
the third number decides as much as the first.

Run:
    python src/blocking_keys_recall.py --entities 60000 --out results/blocking_keys/recall.json
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
from src import pair_model as pm

BASE_FIELDS = (1, 2, "nw4", "core", "ww", "np", "cc")
NEW_FIELDS = ("rn", "dp", "nx", "xk")


def tighten(postings, max_df: int):
    """The same postings kept only where a key occurs at most max_df times.

    A key list cached at a looser df already holds every key of a tighter one
    with its full count, so a tighter family is a filter over the cache rather
    than another pass over the source.
    """
    key, row = postings
    uniq, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    keep = counts[inverse] <= max_df
    key, row = key[keep], row[keep]
    order = np.argsort(key, kind="stable")
    return key[order], row[order]


def ids_of(cands, tid) -> dict:
    """{s1_row: set of encoded target id} from {s1_row: {source: set of target row}}."""
    return {r: {src * pm.ENCODE + int(tid[src][tr]) for src, trs in per.items() for tr in trs}
            for r, per in cands.items()}


def merge(left, right) -> dict:
    """Per-entity, per-source union of two candidate maps."""
    out = {}
    for r in left:
        out[r] = {src: set(left[r].get(src, ())) | set(right[r].get(src, ()))
                  for src in (2, 3)}
    return out


def survives_prerank(sub, text, tid, top_k, want) -> tuple:
    """How many of `want`'s pairs survive the top-k pre-ranker, and the kept total."""
    if not sub:
        return 0, 0
    tt = pm.fetch_target_text(sub, "train")
    keys, a_, n_ = pm.prerank_scores(sub, text, tt)
    keep = pm.prefilter_index(a_, n_, keys, top_k)
    kept = {}
    for i in keep.tolist():
        r, src, tr = keys[i]
        kept.setdefault(r, set()).add(src * pm.ENCODE + int(tid[src][tr]))
    return sum(len(want[r] & kept.get(r, set())) for r in want), sum(len(v) for v in kept.values())


def main(a) -> None:
    """Measure every new family against the current candidate lists."""
    t0 = time.time()
    s1, rows, truth = pm.sample_entities(a.entities, C.SEED)
    ctry = er_io.country_keys(s1)
    text = pm.read_s1_text("train", rows)
    tid = {src: er_io.parse_source("train", src)["ids"] for src in (2, 3)}
    labels, country = s1["countries"], s1["country"]
    n_true = sum(len(v) for v in truth.values())
    print(f"{len(rows):,} entities, {n_true:,} true pairs ({time.time() - t0:.0f}s)", flush=True)

    fields = tuple(f for f in NEW_FIELDS if f not in set(a.drop.split(","))) if a.drop else NEW_FIELDS
    index = pm.build_index("train", a.max_df, fields=BASE_FIELDS + fields)
    for src in (2, 3):
        for fam, df in (("xk", a.xk_df), ("dp", a.dp_df)):
            if fam in fields and df < a.max_df:
                index[src][f"tok{fam}"] = tighten(index[src][f"tok{fam}"], df)
    print(f"index built, families {fields} (xk df {a.xk_df}, dp df {a.dp_df}) "
          f"({time.time() - t0:.0f}s)", flush=True)

    base = pm.gather(rows, s1, ctry, text, index, a.max_group, fields=BASE_FIELDS)
    base_ids = ids_of(base, tid)
    base_raw = sum(len(v) for v in base_ids.values())
    never = {r: truth[r] - base_ids[r] for r in truth}
    n_never = sum(len(v) for v in never.values())
    print(f"baseline: {base_raw:,} raw candidates, {n_never:,} never-blocked true pairs "
          f"({n_never / max(n_true, 1):.2%} of true) ({time.time() - t0:.0f}s)", flush=True)

    by_ctry = {}
    for r, miss in never.items():
        if miss:
            c = labels[country[r]]
            by_ctry[c] = by_ctry.get(c, 0) + len(miss)

    out = {"entities": len(rows), "true_pairs": n_true, "never_blocked": n_never,
           "never_blocked_by_country": by_ctry, "baseline_raw": base_raw,
           "baseline_raw_per_entity": base_raw / len(rows), "families": {}}

    combined = None
    for field in fields + ("ALL",):
        flds = fields if field == "ALL" else (field,)
        cands = pm.gather(rows, s1, ctry, text, index, a.max_group, fields=flds, base_keys=False)
        fam = ids_of(cands, tid)
        if field == "ALL":
            combined = cands
        recovered = {r: never[r] & fam[r] for r in never}
        n_rec = sum(len(v) for v in recovered.values())
        added = sum(len(fam[r] - base_ids[r]) for r in fam)
        rec_ctry = {}
        for r, got in recovered.items():
            if got:
                c = labels[country[r]]
                rec_ctry[c] = rec_ctry.get(c, 0) + len(got)
        out["families"][str(field)] = {
            "recovered": n_rec,
            "recovered_share": n_rec / max(n_never, 1),
            "recovered_by_country": rec_ctry,
            "added_raw": added,
            "added_per_entity": added / len(rows),
            "added_share_of_baseline": added / max(base_raw, 1),
        }
        print(f"{str(field):>4}: recovers {n_rec:,} ({n_rec / max(n_never, 1):6.2%}) | "
              f"+{added:,} raw (+{added / max(base_raw, 1):5.2%}, "
              f"{added / len(rows):.2f}/entity) ({time.time() - t0:.0f}s)", flush=True)

    # The pre-ranker only matters for entities that actually gained a pair, but
    # those entities' full candidate lists compete for the k slots.
    all_ids = ids_of(combined, tid)
    won = {r: never[r] & all_ids[r] for r in never}
    won = {r: v for r, v in won.items() if v}
    sub = {r: {src: set(base[r].get(src, ())) | set(combined[r].get(src, ()))
               for src in (2, 3)} for r in won}
    kept, kept_total = survives_prerank(sub, text, tid, a.top_k, won)
    n_won = sum(len(v) for v in won.values())
    out["prerank"] = {"top_k": a.top_k, "entities_touched": len(won),
                      "recovered": n_won, "survives_top_k": kept,
                      "survives_share": kept / max(n_won, 1), "kept_pairs": kept_total}
    print(f"\ntop-{a.top_k} pre-ranker: {kept:,} of {n_won:,} recovered pairs survive "
          f"({kept / max(n_won, 1):.2%}) on {len(won):,} entities ({time.time() - t0:.0f}s)")

    out["seconds"] = round(time.time() - t0, 1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--entities", type=int, default=60000)
    p.add_argument("--max-df", type=int, default=300)
    p.add_argument("--max-group", type=int, default=200)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--xk-df", type=int, default=300, help="tighter document-frequency cap for xk")
    p.add_argument("--dp-df", type=int, default=300, help="tighter document-frequency cap for dp")
    p.add_argument("--drop", default="", help="comma-separated new families to leave out")
    p.add_argument("--out", default="results/blocking_keys/recall.json")
    main(p.parse_args())
