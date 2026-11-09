"""Score decision rules on a whole scored training split, at test density.

A held-out sample of a few thousand entities almost never has two entities
competing for the same target, so it cannot measure target-argmax (each record
goes to the one entity that scores it highest) or the threshold that works
under it. Scoring the entire training split does: every entity competes, as on
test. The entities the matcher was fitted on are excluded from the average;
every other Source 1 entity counts, including those with no candidate at all.

Run (after predict_matches.py --split train --dump-scores DIR):
    python src/full_density_eval.py --scores outputs/scores/train_f28 --exclude-sample 60000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import er_io
from src.decision_rules import load_scores

ENCODE = 2_000_000_000


def truth_all():
    """True encoded match set for every Source 1 training row."""
    s1 = er_io.parse_source("train", 1)
    gt = er_io.parse_ground_truth()
    row_of = er_io.index_of(s1["ids"], gt["s1"])
    enc = (gt["match_src"].astype(np.int64) * ENCODE + gt["match_id"]).tolist()
    off = gt["offsets"].tolist()
    return {r: set(enc[off[g]:off[g + 1]]) for g, r in enumerate(row_of.tolist())}, s1


def argmax_mask(enc_id, prob):
    """True for each target's single highest-scoring pair."""
    order = np.lexsort((-prob, enc_id))
    e = enc_id[order]
    keep = np.zeros(len(e), dtype=bool)
    keep[order[np.r_[True, e[1:] != e[:-1]]]] = True
    return keep


def macro(truth, rows, pred_rows, pred_ids, country, labels):
    """Macro F0.5 over `rows`, overall and per country label."""
    pred = {}
    for r, i in zip(pred_rows.tolist(), pred_ids.tolist()):
        pred.setdefault(r, set()).add(i)
    tot, per = 0.0, {}
    for r in rows:
        t, q = truth[r], pred.get(r, set())
        if not t and not q:
            s = 1.0
        elif not t or not q:
            s = 0.0
        else:
            c = len(t & q)
            s = 1.25 * c / (len(q) + 0.25 * len(t)) if c else 0.0
        tot += s
        lab = labels[country[r]]
        a = per.setdefault(lab, [0.0, 0])
        a[0] += s
        a[1] += 1
    return tot / len(rows), {k: v[0] / v[1] for k, v in per.items()}


def main(a) -> None:
    """Load scores, sweep rules, print and save the table."""
    s1_row, enc_id, prob = load_scores(Path(a.scores))
    truth, s1 = truth_all()
    excluded = set()
    if a.exclude_sample:
        from src.pair_model import sample_entities
        _, rows, _ = sample_entities(a.exclude_sample, C.SEED, a.exclude_extra)
        excluded = set(rows.tolist())
    rows = [r for r in truth if r not in excluded]
    print(f"evaluating {len(rows):,} entities ({len(excluded):,} fitted entities excluded)")
    am = argmax_mask(enc_id, prob)
    out = {}
    for rule, mask in (("threshold", np.ones(len(prob), dtype=bool)), ("argmax_threshold", am)):
        for t in [float(x) for x in a.thresholds.split(",")]:
            keep = mask & (prob >= t)
            score, per = macro(truth, rows, s1_row[keep], enc_id[keep], s1["country"],
                               s1["countries"])
            out[f"{rule} {t:.2f}"] = {"macro_f05": score, **per}
            print(f"  {rule:>17} {t:.2f}: {score:.5f}  "
                  + "  ".join(f"{k} {v:.5f}" for k, v in sorted(per.items())), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--exclude-sample", type=int, default=60000,
                   help="drop the entities of the fitting sample (same seed as pair_model.fit)")
    p.add_argument("--exclude-extra", type=int, default=0,
                   help="also drop the extra training entities drawn by sample_entities")
    p.add_argument("--thresholds", default="0.60,0.70,0.75,0.80")
    p.add_argument("--out", default="results/full_density/rules.json")
    main(p.parse_args())
