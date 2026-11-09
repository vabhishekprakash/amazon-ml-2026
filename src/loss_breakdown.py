"""Where the held-out F0.5 loses its points, by cause and by country.

Fits the matcher (pair_model.fit), applies the threshold rule to the held-out
entities, and then asks, for each kind of error, how much the macro score would
rise if only that error were fixed:

  not_blocked   true matches that no blocking key reached
  prefiltered   true matches that were blocked but cut by the top-k pre-ranker
  scored_low    true matches that were scored but fell under the threshold
  wrong_ids     predicted ids that are not true matches, on entities with matches
  singletons    entities with no true match that were given one anyway

Each gain is measured alone against the actual predictions, so the largest
number is the thing to fix first.

Run:
    python src/loss_breakdown.py --entities 60000 --threshold 0.70 --workers 9
"""
import argparse
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import pair_model as pm


def f05(t: set, q: set) -> float:
    """Per-entity F0.5; the empty-empty case scores 1.0."""
    if not t and not q:
        return 1.0
    if not t or not q:
        return 0.0
    c = len(t & q)
    return 1.25 * c / (len(q) + 0.25 * len(t)) if c else 0.0


def breakdown(res, threshold):
    """Gain per cause and country, in points of the overall macro score."""
    truth, hold, by_ent, raw = res["truth"], res["holdout_set"], res["val_by_ent"], res["raw_cands"]
    labels = res["countries"]
    n = len(hold)
    causes = ["not_blocked", "prefiltered", "scored_low", "wrong_ids", "singletons"]
    gain = {c: {} for c in causes}
    actual = {}
    for r in hold:
        t = truth[r]
        scored = {i for _, i in by_ent.get(r, ())}
        pred = {i for p, i in by_ent.get(r, ()) if p >= threshold}
        ctry = labels[res["country"][r]]
        base = f05(t, pred)
        actual[ctry] = actual.get(ctry, 0.0) + base
        fixes = {
            "not_blocked": pred | (t - raw[r]),
            "prefiltered": pred | ((t & raw[r]) - scored),
            "scored_low": pred | ((t & scored) - pred),
            "wrong_ids": (pred & t) if t else pred,
            "singletons": set() if not t else pred,
        }
        for c in causes:
            gain[c][ctry] = gain[c].get(ctry, 0.0) + f05(t, fixes[c]) - base
    total = sum(actual.values()) / n
    return total, {c: {k: v / n for k, v in d.items()} for c, d in gain.items()}, \
        {k: v / n for k, v in actual.items()}


def main(a) -> None:
    """Fit, score the held-out entities and print the loss table."""
    pool = get_context("spawn").Pool(a.workers) if a.workers > 1 else None
    try:
        res = pm.fit(a.entities, a.max_df, a.max_group, False, pool, a.workers, a.top_k,
                     a.extra_train, json.loads(a.lgb) if a.lgb else None, a.rounds, a.cv,
                     a.models, tuple(a.drop.split(",")) if a.drop else (), a.preranker,
                     pairs_out=a.dump_pairs, pseudo=a.pseudo)
    finally:
        if pool is not None:
            pool.close()
    total, gain, actual = breakdown(res, a.threshold)
    if res.get("fold_scores"):
        fs = np.array(res["fold_scores"])
        print(f"cross-fit folds at 0.75: {', '.join(f'{x:.5f}' for x in fs)} | "
              f"mean {fs.mean():.5f} std {fs.std(ddof=1):.5f}")
    if a.dump:
        import pickle
        keep = ("val_by_ent", "truth", "holdout_set", "raw_cands", "country", "countries")
        with open(a.dump, "wb") as f:
            pickle.dump({k: res[k] for k in keep}, f)
    countries = sorted(actual)
    counts = {}
    for r in res["holdout_set"]:
        c = res["countries"][res["country"][r]]
        counts[c] = counts.get(c, 0) + 1
    print(f"\nheld-out macro F0.5 at threshold {a.threshold}: {total:.5f} "
          f"on {len(res['holdout_set']):,} entities "
          f"({', '.join(f'{c} {counts[c]:,}' for c in countries)})")
    print(f"{'gain if fixed':>16} " + " ".join(f"{c:>9}" for c in countries) + "     total")
    for cause, d in sorted(gain.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"{cause:>16} " + " ".join(f"{d.get(c, 0):>9.5f}" for c in countries)
              + f" {sum(d.values()):>9.5f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"threshold": a.threshold, "entities": a.entities, "heldout": len(res["holdout_set"]),
             "fold_scores": res.get("fold_scores"), "pseudo": a.pseudo,
             "macro_f05": total, "gain_if_fixed": gain,
             "country_share_of_score": actual}, indent=2), encoding="utf-8")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--entities", type=int, default=60000)
    p.add_argument("--max-df", type=int, default=300)
    p.add_argument("--max-group", type=int, default=200)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--threshold", type=float, default=0.75)
    p.add_argument("--workers", type=int, default=9)
    p.add_argument("--out", default="results/loss_breakdown/breakdown.json")
    p.add_argument("--dump", default=None, help="pickle the held-out predictions here")
    p.add_argument("--dump-pairs", default=None,
                   help="save every scored pair with its label here (.npz)")
    p.add_argument("--pseudo", default=None,
                   help="pseudo_labels.py output added to training only")
    p.add_argument("--lgb", default=None, help='LightGBM overrides as JSON, e.g. {"num_leaves": 127}')
    p.add_argument("--rounds", type=int, default=300)
    p.add_argument("--models", type=int, default=1, help="average this many bagged models")
    p.add_argument("--preranker", type=int, default=0,
                   help="train a learned pre-ranker on this many other entities")
    p.add_argument("--drop", default="", help="comma-separated feature names to leave out")
    p.add_argument("--cv", type=int, default=0,
                   help="cross-fit over the whole sample with this many entity folds")
    p.add_argument("--extra-train", type=int, default=0,
                   help="entities from outside the sample added to training only")
    main(p.parse_args())
