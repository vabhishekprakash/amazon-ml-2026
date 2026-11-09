"""Turn dumped pair scores into a submission, under a chosen decision rule.

Scoring a split costs hours; deciding what to do with the scores costs seconds.
predict_matches.py --dump-scores writes every (entity, target, probability)
triple to disk, and this applies a rule to them and writes the two TSVs. That
means a decision-rule change never requires re-running the feature pass.

Rules:
  greedy           per-entity greedy admission under F0.5 (the current default)
  argmax_greedy    a target may only be claimed by the entity that scores it
                   highest, applied BEFORE admission, then greedy per entity
  threshold        keep every pair at or above --threshold
  argmax_threshold each target goes to its best entity, kept above --threshold
  topk             the best --topk candidates per entity

The argmax rules apply one-to-one before admission, as a hard exclusion, rather
than as a cleanup afterwards where it can only break ties. The training data
licenses it: no Source 2/3 record belongs to two Source 1 entities.

--unseen-threshold replaces --threshold for entities of countries that have no
Source 1 record in training. Only the threshold rules use it.

--blank-name-rule COUNTRY adds, for entities of that country, every candidate
whose record has a blank address and whose normalized name equals the entity's
while no other Source 1 entity of the country carries that name. Each record
still goes only to the entity that scores it highest, and only pairs already in
the written candidate set are eligible.

--candidates final-stage writes candidate_pairs.tsv as each country's final
inference set: the pairs the cross-encoder blend scored (the stage2 flag of a
ce_blend.py apply dump), plus every pair LightGBM scored for countries without
training labels when the dump kept them on LightGBM. The default, scored, lists
every pair in the dump.

Run:
    python src/decide.py --scores outputs/scores/test_v9_final --split test \\
        --rule argmax_threshold --threshold 0.80 --unseen-threshold 0.80 \\
        --candidates final-stage --run-name final
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import er_io
from src.decision_rules import entity_slices, load_scores
from src.er_norm import normalize_name

ENCODE = 2_000_000_000


def decode(enc: int) -> str:
    """Encoded target id back to its entity_id string."""
    src, num = divmod(int(enc), ENCODE)
    return f"S{src}-{num}"


def apply_argmax(s1_row, enc_id, prob):
    """Keep only each target's single best entity. Returns a boolean mask."""
    order = np.lexsort((-prob, enc_id))
    first = np.zeros(len(enc_id), dtype=bool)
    e = enc_id[order]
    first[order[np.concatenate(([True], e[1:] != e[:-1]))]] = True
    return first


def greedy_admit(p, ids, k_hat):
    """Admit while p exceeds the running score divided by 1.25."""
    chosen, c, n = [], 0.0, 0
    for j in range(len(p)):
        if n and p[j] <= c / (n + 0.25 * k_hat):
            break
        chosen.append(int(ids[j]))
        c += float(p[j])
        n += 1
    return chosen


def unseen_rows(split: str) -> np.ndarray:
    """Boolean per Source 1 row of the split: its country has no Source 1 record in training."""
    seen = set(er_io.parse_source("train", 1)["countries"])
    s1 = er_io.parse_source(split, 1)
    return ~np.isin(np.array(s1["countries"])[s1["country"]], sorted(seen))


def build(rule, s1_row, enc_id, prob, k_hat, threshold, topk, row_threshold=None):
    """Apply one rule; returns {s1_row: [enc_id, ...]}.

    row_threshold, if given, is a per Source 1 row threshold for the threshold rules.
    """
    if rule in ("argmax_greedy", "argmax_threshold"):
        keep = apply_argmax(s1_row, enc_id, prob)
        s1_row, enc_id, prob = s1_row[keep], enc_id[keep], prob[keep]
        order = np.lexsort((-prob, s1_row))
        s1_row, enc_id, prob = s1_row[order], enc_id[order], prob[order]

    ents, starts, stops = entity_slices(s1_row)
    out = {}
    for row, a, b in zip(ents.tolist(), starts, stops):
        p, ids = prob[a:b], enc_id[a:b]
        if rule in ("greedy", "argmax_greedy"):
            picked = greedy_admit(p, ids, k_hat)
        elif rule in ("threshold", "argmax_threshold"):
            t = threshold if row_threshold is None else row_threshold[row]
            picked = ids[p >= t].tolist()
        elif rule == "topk":
            picked = ids[:topk].tolist()
        else:
            raise SystemExit(f"unknown rule {rule}")
        if picked:
            out[row] = picked
    return out


def write_pair(out_dir: Path, split: str, assigned, scored):
    """Write matching_results.tsv and candidate_pairs.tsv in file order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = er_io.parse_source(split, 1)["ids"].tolist()
    with open(out_dir / C.MATCHING_FILE, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{C.GT_ID_COL}\t{C.GT_MATCH_COL}\n")
        for row, num in enumerate(ids):
            f.write(f"S1-{num}\t{','.join(decode(i) for i in assigned.get(row, ()))}\n")
    with open(out_dir / C.CANDIDATE_FILE, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{C.GT_ID_COL}\t{C.CANDIDATE_COL}\n")
        for row, num in enumerate(ids):
            f.write(f"S1-{num}\t{','.join(decode(i) for i in scored.get(row, ()))}\n")
    return len(ids)


def final_stage_candidates(score_dir: Path, split: str):
    """{s1_row: [enc_id, ...]} of the pairs the final model of each entity's country scored.

    The cross-encoder blend only sees pairs that passed the LightGBM gate (the
    stage2 flag ce_blend.py apply writes). When the dump kept countries absent
    from training on LightGBM (apply --unseen-keep), LightGBM is their final
    model, so every pair it scored is a candidate there; otherwise they follow
    the same gate as everyone else.
    """
    meta = json.loads((score_dir / "blend_apply.json").read_text(encoding="utf-8"))
    unseen = unseen_rows(split) if meta.get("unseen_keep") else None
    out = {}
    for f in sorted(score_dir.glob("shard_*.npz")):
        b = np.load(f)
        if "stage2" not in b.files:
            raise SystemExit(f"{f.name}: no stage2 flag; write the dump with ce_blend.py apply first")
        keep = b["stage2"] if unseen is None else b["stage2"] | unseen[b["s1_row"]]
        for row, enc in zip(b["s1_row"][keep].tolist(), b["enc_id"][keep].tolist()):
            out.setdefault(row, []).append(enc)
    return out


def blank_name_additions(split, s1_row, enc_id, prob, countries, scored):
    """{s1_row: [enc_id, ...]} of pairs the blank-address name rule adds.

    A pair qualifies when its entity belongs to one of countries, the record has
    a blank address, both normalized names are equal, no other Source 1 entity of
    the country has that name, the entity is the record's highest-scoring
    claimant, and the pair is in scored (the candidate set being written).
    """
    names, ctry = [], []
    with open(C.source_file(split, 1), encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            names.append(normalize_name(parts[1]) if len(parts) == 4 else "")
            ctry.append(parts[-1])
    count = {}
    for key in zip(ctry, names):
        count[key] = count.get(key, 0) + 1
    unique_row = np.array([bool(n) and c in countries and count[(c, n)] == 1 for n, c in zip(names, ctry)])
    keep = apply_argmax(s1_row, enc_id, prob) & unique_row[s1_row]
    rows, encs = s1_row[keep], enc_id[keep]
    blank_name = {}
    for src in (2, 3):
        wanted = set((encs[encs // ENCODE == src] % ENCODE).tolist())
        with open(C.source_file(split, src), encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 4 or parts[2].strip() not in ("", "<NULL>"):
                    continue
                num = int(parts[0].split("-", 1)[1])
                if num in wanted:
                    blank_name[src * ENCODE + num] = normalize_name(parts[1])
    out = {}
    for row, enc in zip(rows.tolist(), encs.tolist()):
        if blank_name.get(enc) == names[row] and enc in scored.get(row, ()):
            out.setdefault(row, []).append(enc)
    return out


def main(a) -> None:
    """Load scores, apply the rule, write both files and a summary."""
    s1_row, enc_id, prob = load_scores(Path(a.scores))
    k_hat = a.k_hat
    row_t = None
    if a.unseen_threshold is not None:
        unseen = unseen_rows(a.split)
        row_t = np.where(unseen, a.unseen_threshold, a.threshold)
        print(f"threshold {a.unseen_threshold} for {int(unseen.sum()):,} entities of countries absent from training")
    assigned = build(a.rule, s1_row, enc_id, prob, k_hat, a.threshold, a.topk, row_t)

    # candidate_pairs must be what the model scored, so it comes from the dump.
    if a.candidates == "final-stage":
        scored = final_stage_candidates(Path(a.scores), a.split)
    else:
        scored = {}
        ents, starts, stops = entity_slices(s1_row)
        for row, x, y in zip(ents.tolist(), starts, stops):
            scored[row] = enc_id[x:y].tolist()

    added = 0
    if a.blank_name_rule:
        scored_sets = {row: set(v) for row, v in scored.items()}
        for row, encs in blank_name_additions(a.split, s1_row, enc_id, prob, set(a.blank_name_rule),
                                              scored_sets).items():
            have = assigned.setdefault(row, [])
            new_ids = [e for e in encs if e not in have]
            have.extend(new_ids)
            added += len(new_ids)
        print(f"blank-address name rule ({', '.join(a.blank_name_rule)}): {added:,} ids added")

    out_dir = C.OUT_DIR / "submissions" / a.run_name
    rows = write_pair(out_dir, a.split, assigned, scored)
    n_ids = sum(len(v) for v in assigned.values())
    n_cand = sum(len(v) for v in scored.values())
    summary = {"rule": a.rule, "split": a.split, "rows": rows,
               "threshold": a.threshold, "unseen_threshold": a.unseen_threshold,
               "candidates": a.candidates, "candidates_per_entity": n_cand / max(rows, 1),
               "topk": a.topk, "k_hat": k_hat,
               "blank_name_rule": a.blank_name_rule, "blank_name_added": added,
               "predicted_ids": n_ids, "ids_per_entity": n_ids / max(rows, 1),
               "entities_with_match": len(assigned),
               "empty_share": 1 - len(assigned) / max(rows, 1)}
    print(json.dumps(summary, indent=2))
    res = C.RESULTS_DIR / a.run_name
    res.mkdir(parents=True, exist_ok=True)
    (res / "decide.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--rule", default="argmax_greedy",
                   choices=["greedy", "argmax_greedy", "threshold", "argmax_threshold", "topk"])
    p.add_argument("--run-name", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--unseen-threshold", type=float, default=None,
                   help="threshold for countries with no Source 1 record in training")
    p.add_argument("--candidates", default="scored", choices=["scored", "final-stage"],
                   help="candidate_pairs.tsv from every scored pair, or from each country's final-stage input")
    p.add_argument("--blank-name-rule", nargs="*", default=[], metavar="COUNTRY",
                   help="countries whose entities take blank-address records with their exact, unique name")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--k-hat", type=float, default=3.479,
                   help="expected matches per entity, used by the admission bar")
    main(p.parse_args())
