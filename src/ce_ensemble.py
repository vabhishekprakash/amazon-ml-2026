r"""Logistic blend of the LightGBM pair model with one or more cross-encoders.

fit    reads the held-out LightGBM probabilities of a src/loss_breakdown.py
       --dump and, for each cross-encoder folder given with --ce, its held-out
       probabilities (ce_heldout.npy of src/ce_rerank.py, one per row of the
       export's heldout_pairs.npz). A dump pair is blended when it is one of
       the exported pairs. Its inputs are the logit of the LightGBM
       probability, then for each cross-encoder the logit of its probability
       and the product of the two logits (logits clipped at 1e-4 as in
       src/ce_blend.py). A logistic regression over these columns is
       cross-fitted over the entity folds of src/ce_blend.py, so every blended
       held-out pair gets an out-of-fold probability; every other pair keeps
       its LightGBM probability. Macro F0.5 of target-argmax plus a threshold
       is printed for each of THRESHOLDS, weighted to the test Source 1
       country mix, and per country. The model JSON holds the coefficients
       refitted on every blended pair, the cross-encoder folders in input
       order, the best threshold and its held-out scores. A cross-encoder is
       used only if its ce_log.json records the digest of the export's
       held-out pairs (src/ce_blend.py load_ce).
apply  writes a copy of a test score dump in which each exported test pair
       (test_pairs.npz of the export) carries the blended probability of the
       model's cross-encoders (their ce_test.npy), through src/ce_blend.py
       cmd_apply with all its shard, pair and scale checks. Test pairs of
       countries absent from the held-out data keep their LightGBM
       probability. Rescored pairs carry the stage2 flag cmd_apply writes,
       which src/decide.py --candidates final-stage lists as the final
       stage's candidates; decide.py runs on the new folder unchanged.

Above logit(1 - 1e-4) the blended logit is compressed as in src/ce_blend.py
blend, so confident pairs keep distinct float32 probabilities.

Run:
    python src/ce_ensemble.py fit --heldout-dump outputs/cache/cv4m5_g127_pseudo.pkl \
        --ce outputs/ce/mpnet_A --out results/ce_blend/blend_mpnet_A.json
    python src/ce_ensemble.py fit --heldout-dump outputs/cache/cv4m5_g127_pseudo.pkl \
        --ce outputs/ce/mpnet_A outputs/ce/mpnet_B outputs/ce/mpnet_C --out results/ce_blend/blend_ABC.json
    python src/ce_ensemble.py apply --model results/ce_blend/blend_mpnet_A.json \
        --scores outputs/scores/test_v9 --out outputs/scores/test_v9_mpnet_A
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.ce_blend import (CLIP, FOLDS, SQUASH, cmd_apply, country_weights, entity_scores, fit_lr, fold_of,
                          load_ce, load_heldout, logit, pair_key)
from src.ce_pairs import pairs_digest
from src.decide import apply_argmax

THRESHOLDS = (0.70, 0.75, 0.80, 0.85)


def design(lg, ces):
    """Blend inputs: the LightGBM logit, then each cross-encoder's logit and its product with the LightGBM logit."""
    cols = [lg]
    for c in ces:
        cols += [c, lg * c]
    return np.column_stack(cols)


def prob(model, X):
    """Blended probability (float32) of design rows under saved coefficients, logits above logit(1 - CLIP) compressed by SQUASH."""
    z = float(model["intercept"]) + X @ np.asarray(model["coef"], dtype=np.float64)
    top = np.log((1 - CLIP) / CLIP)
    z = np.where(z > top, top + (z - top) / SQUASH, z)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def cmd_fit(a) -> None:
    """Cross-fit the blend on the held-out pairs, print its scores and save the model JSON."""
    h = load_heldout(a.heldout_dump)
    with np.load(Path(a.pairs) / "heldout_pairs.npz") as hp:
        dg = pairs_digest(hp["s1_row"], hp["enc_id"])
        ke = pair_key(hp["s1_row"], hp["enc_id"])
    kh = pair_key(h["s1_row"], h["enc_id"])
    pos = np.minimum(np.searchsorted(ke, kh), len(ke) - 1)
    has = ke[pos] == kh
    ces = [logit(load_ce(Path(d), dg, len(ke))[0][pos][has]) for d in a.ce]
    X = design(logit(h["prob"][has]), ces)
    y = h["label"][has]
    fold = fold_of(h["s1_row"][has])
    oof = np.empty(len(y))
    for k in range(FOLDS):
        te = fold == k
        oof[te] = fit_lr(X[~te], y[~te]).predict_proba(X[te])[:, 1]
    p = h["prob"].copy()
    p[has] = oof
    w, table, _ = country_weights(h["country"])
    c = h["country"].astype(str)
    keep = apply_argmax(h["s1_row"], h["enc_id"], p)
    res = {}
    for t in THRESHOLDS:
        f = entity_scores(h["ent_idx"], h["label"], (p >= t) & keep, h["k_true"])
        res[t] = (float(np.average(f, weights=w)), {cc: float(f[c == cc].mean()) for cc in sorted(set(c))})
    for t, (s, pc) in res.items():
        print(f"  {t:.2f}: {s:.5f} {pc}")
    best = max(res, key=lambda t: res[t][0])
    m = fit_lr(X, y)
    model = {"intercept": float(m.intercept_[0]), "coef": m.coef_[0].tolist(),
             "ce_dirs": [str(d) for d in a.ce], "threshold": float(best),
             "per_country": {cc: {} for cc in table}, "verdict": "multi",
             "heldout": str(a.heldout_dump), "heldout_weighted": res[best][0],
             "per_country_at_best": res[best][1]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(model, indent=1))
    print("best", best, res[best][0], "coef", np.round(m.coef_[0], 3), "->", a.out)


def cmd_apply_ensemble(a) -> None:
    """Write the blended test score dump through src/ce_blend.py cmd_apply with an ensemble rescore hook."""
    model = json.loads(Path(a.model).read_text())
    tests = [np.load(Path(d) / "ce_test.npy").astype(np.float32) for d in model["ce_dirs"]]
    if len({len(t) for t in tests}) != 1:
        raise SystemExit("ce_test.npy lengths differ between the model's cross-encoders")

    def rescore(m, s1_row, enc_id, p, ce):
        """Blended probability of the exported test pairs from their LightGBM and cross-encoder probabilities."""
        return prob(m, design(logit(p), [logit(t) for t in tests]))

    ns = argparse.Namespace(model=a.model, scores=a.scores, pairs=a.pairs, ce=model["ce_dirs"][0],
                            unseen_keep=True, out=a.out)
    cmd_apply(ns, rescore=rescore)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    export_default = str(C.OUT_DIR / "ce" / "export")
    f = sub.add_parser("fit", help="fit the blend on held-out pairs and save the model JSON")
    f.add_argument("--heldout-dump", required=True, help="src/loss_breakdown.py --dump pickle")
    f.add_argument("--pairs", default=export_default,
                   help="src/ce_pairs.py export folder holding heldout_pairs.npz")
    f.add_argument("--ce", nargs="+", required=True,
                   help="cross-encoder output folders (ce_heldout.npy, ce_log.json), in blend order")
    f.add_argument("--out", required=True, help="model JSON to write")
    ap = sub.add_parser("apply", help="write a test score dump with blended probabilities")
    ap.add_argument("--model", required=True, help="JSON written by fit")
    ap.add_argument("--scores", required=True, help="test score dump folder (src/predict_matches.py)")
    ap.add_argument("--pairs", default=export_default,
                    help="src/ce_pairs.py export folder holding test_pairs.npz and shards.json")
    ap.add_argument("--out", required=True, help="new score dump folder")
    args = p.parse_args()
    cmd_fit(args) if args.cmd == "fit" else cmd_apply_ensemble(args)
