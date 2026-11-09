"""Compare decision rules offline, on scored pairs dumped by predict_matches.

Scoring a full split costs an hour; choosing what to do with the scores costs
seconds. This loads the dumped (entity, target, probability) triples once and
evaluates every candidate rule against the training labels, including two
oracles that bound what any rule could achieve:

  ceiling      pick exactly the true matches present in the candidates
  best_prefix  keep the best-scoring prefix of the model's own ranking, with
               the cut chosen per entity by an oracle

The difference between best_prefix and a real rule is the cost of *stopping*;
the difference between ceiling and best_prefix is the cost of *ranking*. Those
two numbers say whether to work on the classifier or the decision layer.

Run:
    python src/decision_rules.py --scores outputs/scores/train
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


def load_scores(score_dir: Path):
    """Concatenate every shard into (s1_row, enc_id, prob), sorted by entity."""
    files = sorted(Path(score_dir).glob("shard_*.npz"))
    if not files:
        raise SystemExit(f"no shards in {score_dir}")
    rows, ids, probs = [], [], []
    for f in files:
        b = np.load(f)
        rows.append(b["s1_row"])
        ids.append(b["enc_id"])
        probs.append(b["prob"])
    s1_row = np.concatenate(rows)
    enc_id = np.concatenate(ids)
    prob = np.concatenate(probs)
    order = np.lexsort((-prob, s1_row))          # by entity, best first
    print(f"loaded {len(files)} shards, {len(s1_row):,} scored pairs")
    return s1_row[order], enc_id[order], prob[order]


def entity_slices(s1_row):
    """Start and stop index of each entity's contiguous block."""
    change = np.flatnonzero(np.diff(s1_row)) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(s1_row)]))
    return s1_row[starts], starts, stops


def truth_map(rows_needed):
    """True encoded match set per Source 1 row, for the rows we scored."""
    s1 = er_io.parse_source("train", 1)
    gt = er_io.parse_ground_truth()
    row_of = er_io.index_of(s1["ids"], gt["s1"])
    enc = gt["match_src"].astype(np.int64) * ENCODE + gt["match_id"]
    want = set(rows_needed.tolist())
    out = {}
    for g, row in enumerate(row_of.tolist()):
        if row in want:
            lo, hi = gt["offsets"][g], gt["offsets"][g + 1]
            out[row] = set(enc[lo:hi].tolist())
    return out, len(s1["ids"])


def f05(c: int, n: int, k: int) -> float:
    """F0.5 = 1.25c / (n + 0.25k); the empty-empty case scores 1.0."""
    if k == 0 and n == 0:
        return 1.0
    if c == 0:
        return 0.0
    return 1.25 * c / (n + 0.25 * k)


def evaluate(pred_counts, truth, n_entities_total):
    """Macro F0.5 over every Source 1 entity, including those never scored.

    Entities with no scored candidate predict nothing: 1.0 if they truly have
    no match, 0.0 otherwise.
    """
    total = 0.0
    seen = set()
    for row, chosen in pred_counts.items():
        t = truth.get(row, set())
        c = len(t & chosen)
        total += f05(c, len(chosen), len(t))
        seen.add(row)
    for row, t in truth.items():
        if row not in seen:
            total += 1.0 if not t else 0.0
    return total / len(truth)


def main(score_dir, out_path, thresholds) -> None:
    """Evaluate every rule and print them ranked."""
    s1_row, enc_id, prob = load_scores(Path(score_dir))
    ents, starts, stops = entity_slices(s1_row)
    truth, n_total = truth_map(ents)
    print(f"entities scored: {len(ents):,} | with labels: {len(truth):,}")

    results = {}

    def record(name, preds):
        results[name] = evaluate(preds, truth, n_total)
        print(f"  {name:>28} {results[name]:.5f}", flush=True)

    # Oracle 1: perfect selection from the candidates we generated.
    ceiling = {}
    for row, a, b in zip(ents.tolist(), starts, stops):
        ceiling[row] = truth.get(row, set()) & set(enc_id[a:b].tolist())
    record("ceiling (perfect pick)", ceiling)

    # Oracle 2: best prefix of the model's own ranking, cut per entity.
    best_prefix = {}
    for row, a, b in zip(ents.tolist(), starts, stops):
        t = truth.get(row, set())
        ids = enc_id[a:b]
        best_score, best_n = (1.0 if not t else 0.0), 0
        c = 0
        for n in range(1, len(ids) + 1):
            if ids[n - 1] in t:
                c += 1
            s = f05(c, n, len(t))
            if s > best_score:
                best_score, best_n = s, n
        best_prefix[row] = set(ids[:best_n].tolist())
    record("best_prefix (oracle stop)", best_prefix)

    # Current rule: greedy admission under the metric, global k.
    k_global = float(np.mean([len(v) for v in truth.values()]))
    greedy = {}
    for row, a, b in zip(ents.tolist(), starts, stops):
        p, ids = prob[a:b], enc_id[a:b]
        chosen, c, n = [], 0.0, 0
        for j in range(len(p)):
            if n and p[j] <= c / (n + 0.25 * k_global):
                break
            chosen.append(ids[j])
            c += float(p[j])
            n += 1
        greedy[row] = set(chosen)
    record("greedy admission (current)", greedy)

    # Fixed top-k per entity.
    for k in (1, 2, 3, 4, 5):
        record(f"top-{k} per entity",
               {row: set(enc_id[a:min(a + k, b)].tolist())
                for row, a, b in zip(ents.tolist(), starts, stops)})

    # Per-entity probability threshold.
    for t in thresholds:
        record(f"prob >= {t:.2f}",
               {row: set(enc_id[a:b][prob[a:b] >= t].tolist())
                for row, a, b in zip(ents.tolist(), starts, stops)})

    # The reframe: each target goes to its single best entity, then threshold.
    order = np.lexsort((-prob, enc_id))
    e_sorted, r_sorted, p_sorted = enc_id[order], s1_row[order], prob[order]
    first = np.concatenate(([True], e_sorted[1:] != e_sorted[:-1]))
    win_e, win_r, win_p = e_sorted[first], r_sorted[first], p_sorted[first]
    print(f"  (one-to-one: {len(win_e):,} distinct targets out of {len(enc_id):,} pairs)")
    for t in thresholds:
        keep = win_p >= t
        preds = {}
        for row, i in zip(win_r[keep].tolist(), win_e[keep].tolist()):
            preds.setdefault(row, set()).add(i)
        record(f"target-argmax, prob >= {t:.2f}", preds)

    # The two oracles are upper bounds, not candidate rules, so exclude them
    # when naming the best usable rule.
    oracles = {"ceiling (perfect pick)", "best_prefix (oracle stop)"}
    real = {k: v for k, v in results.items() if k not in oracles}
    best = max(real, key=real.get)
    ceil_v = results["ceiling (perfect pick)"]
    prefix_v = results["best_prefix (oracle stop)"]
    print(f"\nbest usable rule: {best} at {real[best]:.5f}")
    print(f"ranking cost  = ceiling - best_prefix   = {ceil_v - prefix_v:.5f}"
          "   (the model's ordering is imperfect)")
    print(f"stopping cost = best_prefix - best rule = {prefix_v - real[best]:.5f}"
          "   (we cut the ranked list in the wrong place)")
    print("Spend hours on whichever of those two is larger.")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scores", default="outputs/scores/train")
    p.add_argument("--out", default="results/decision_rules.json")
    p.add_argument("--thresholds", default="0.10,0.20,0.30,0.40,0.50,0.60,0.70")
    a = p.parse_args()
    main(a.scores, a.out, [float(x) for x in a.thresholds.split(",")])
