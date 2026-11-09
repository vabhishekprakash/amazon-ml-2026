"""Second-stage pair model over LightGBM and cross-encoder probabilities.

fit    trains a LightGBM binary model on the held-out pairs exported by
       ce_pairs.py (LightGBM prob at or above the export floor). Its features
       need only a score dump and cross-encoder scores, so test has them too:
       both logits and their product; for each of the two probs its rank
       inside the entity (1 = best, ties share the best rank), gap to the
       entity max, entity max, count of the entity's pairs at or above 0.5
       and 0.9, and entity mean; the entity's exported pair count and its
       count of LightGBM probs at or above BASE_THRESHOLD; the target's
       source. Entity context is taken over the entity's exported pairs.
       Five entity folds (ce_blend.fold_of) give every exported pair an
       out-of-fold prob; pairs below the floor keep their LightGBM prob.
       Scoring is ce_blend fit's: macro F0.5 over every held-out entity,
       weighted to the test country mix, at THRESHOLDS. LightGBM alone, the
       logistic blend (cross-fitted on the same folds) and stage-2 are
       reported side by side, per fold and per country. Per country both
       second stages are also refitted without the country and scored on it;
       with --loco-ce both are fitted, on a cross-encoder's scores, to the
       countries that cross-encoder was trained on and scored on the others.
       Each of SETTINGS is cross-fitted and the one with the best weighted
       score is kept. The verdict is adopt only if stage-2 at its threshold
       beats the blend at its threshold by at least MIN_GAIN (weighted),
       every fold improves on the blend, and no country scores below the
       blend in the plain, refit-without-country or --loco-ce view. The kept
       setting is refitted on every exported pair and saved as LightGBM text
       (--out with a .txt suffix) next to the JSON report, which records the
       feature list, threshold, floor and held-out countries.
apply  ce_blend.py apply with the stage-2 prob in place of the blend: the same
       shard alignment checks, --unseen-keep and admission-flip report. The
       model is fitted on raw held-out probs and has no calibrator, so a test
       export marked calibrated, or a shard with step-valued probs, is
       refused. The export floor must equal the held-out one, because the
       entity counts depend on it, and the booster file must match the sha256
       fit recorded.

Run:
    python src/stage2.py fit --heldout-dump outputs/cache/cv4m5_addridf.pkl \
        --pairs outputs/ce/export --ce outputs/ce/run/ce --loco-ce outputs/ce/run_us/ce \
        --out results/stage2/stage2_addridf.json
    python src/stage2.py apply --model results/stage2/stage2_addridf.json \
        --scores outputs/scores/test_v8_raw --pairs outputs/ce/export_raw \
        --ce outputs/ce/run/ce --unseen-keep --out outputs/scores/test_v8_s2
    python src/decide.py --scores outputs/scores/test_v8_s2 --split test \
        --rule argmax_threshold --threshold <printed threshold> --run-name <run>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import ce_blend
from src.ce_blend import (BASE_THRESHOLD, CLIP, ENCODE, FOLDS, MIN_GAIN, SQUASH, THRESHOLDS, align,
                          country_weights, design, entity_scores, fit_lr, flip_rates, fold_of, load_ce,
                          load_heldout, oof_blend)
from src.ce_pairs import MIN_DISTINCT, distinct_frac, pairs_digest, sha256

CONTEXT = ("rank", "gap", "max", "n50", "n90", "mean")
FEATURES = (["logit_lgbm", "logit_ce", "logit_lgbm*logit_ce"]
            + [f"{c}_lgbm" for c in CONTEXT] + [f"{c}_ce" for c in CONTEXT]
            + ["n_pairs", "n_admit_lgbm", "source"])
BASE_PARAMS = {"objective": "binary", "verbose": -1, "seed": 0, "deterministic": True,
               "force_row_wise": True}
SETTINGS = {
    "leaves31": {"num_leaves": 31, "min_data_in_leaf": 100, "learning_rate": 0.05, "rounds": 300},
    "leaves15_l2": {"num_leaves": 15, "min_data_in_leaf": 300, "learning_rate": 0.05,
                    "lambda_l2": 10.0, "rounds": 300},
}


def context(g, p, n_groups, out):
    """Write the CONTEXT columns of p within groups g (0..n_groups-1, each non-empty) into out."""
    order = np.lexsort((-p, g))
    gs, ps = g[order], p[order]
    n = len(p)
    new = np.ones(n, dtype=bool)
    new[1:] = (gs[1:] != gs[:-1]) | (ps[1:] != ps[:-1])
    run = np.maximum.accumulate(np.where(new, np.arange(n), 0))
    first = np.searchsorted(gs, np.arange(n_groups))
    out[order, 0] = run - first[gs] + 1
    top = ps[first][g]
    del gs, ps, new, run, order
    cnt = np.bincount(g, minlength=n_groups)
    out[:, 1] = top - p
    out[:, 2] = top
    out[:, 3] = np.bincount(g, weights=p >= 0.5, minlength=n_groups)[g]
    out[:, 4] = np.bincount(g, weights=p >= 0.9, minlength=n_groups)[g]
    out[:, 5] = (np.bincount(g, weights=p, minlength=n_groups) / cnt)[g]


def features(s1_row, enc_id, p_lgbm, p_ce):
    """Stage-2 feature matrix (float32, columns FEATURES) for one export's pairs."""
    _, g = np.unique(s1_row, return_inverse=True)
    g = g.ravel()
    n_groups = int(g.max()) + 1 if len(g) else 0
    pl = np.asarray(p_lgbm, dtype=np.float64)
    pc = np.asarray(p_ce, dtype=np.float64)
    X = np.empty((len(pl), len(FEATURES)), dtype=np.float32)
    X[:, :3] = design(pl, pc)
    k = len(CONTEXT)
    view = np.empty((len(pl), k), dtype=np.float32)
    context(g, pl, n_groups, view)
    X[:, 3:3 + k] = view
    context(g, pc, n_groups, view)
    X[:, 3 + k:3 + 2 * k] = view
    X[:, -3] = np.bincount(g, minlength=n_groups)[g]
    X[:, -2] = np.bincount(g, weights=pl >= BASE_THRESHOLD, minlength=n_groups)[g]
    X[:, -1] = np.asarray(enc_id, dtype=np.int64) // ENCODE
    return X


def train(X, y, setting, threads):
    """LightGBM binary booster for one of SETTINGS."""
    import lightgbm as lgb
    params = {**BASE_PARAMS, "num_threads": threads, **{k: v for k, v in setting.items() if k != "rounds"}}
    data = lgb.Dataset(X, label=y.astype(np.float32), feature_name=FEATURES)
    return lgb.train(params, data, num_boost_round=setting["rounds"])


def predict(booster, X):
    """Stage-2 probability as float32, with raw scores above logit(1 - CLIP) compressed by SQUASH as in ce_blend.blend."""
    z = booster.predict(X, raw_score=True)
    top = np.log((1 - CLIP) / CLIP)
    z = np.where(z > top, top + (z - top) / SQUASH, z)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def cross_fit(X, y, fold, setting, threads):
    """Out-of-fold stage-2 probs over the entity folds."""
    oof = np.empty(len(y), dtype=np.float32)
    for k in range(FOLDS):
        te = fold == k
        if te.any():
            oof[te] = predict(train(X[~te], y[~te], setting, threads), X[te])
    return oof


def cmd_fit(a):
    """Cross-fit stage-2 on the held-out pairs, compare it with LightGBM and the blend, save model and report."""
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    h = load_heldout(a.heldout_dump)
    if distinct_frac(h["prob"]) < MIN_DISTINCT:
        raise SystemExit("held-out probs are step-valued (isotonic); stage-2 expects raw booster output")
    pos, floor = align(h, Path(a.pairs))
    with np.load(Path(a.pairs) / "heldout_pairs.npz") as hp:
        digest = pairs_digest(hp["s1_row"], hp["enc_id"])
    ce, trained = load_ce(Path(a.ce), digest, len(pos))
    rows, enc, pl = h["s1_row"][pos], h["enc_id"][pos], h["prob"][pos]
    X = features(rows, enc, pl, ce)
    blended, Xb, y, oof_b = oof_blend(h, pos, ce)
    fold = fold_of(rows)
    print(f"held-out entities {len(h['ents']):,} | scored pairs {len(h['prob']):,} | exported {len(pos):,} "
          f"(floor {floor}) | positives {int(y.sum()):,} ({time.time() - t0:.0f}s)", flush=True)

    ent_idx, label, k_true = h["ent_idx"], h["label"], h["k_true"]
    w, weights, unseen = country_weights(h["country"])
    ctry = h["country"].astype(str)
    efold = fold_of(h["ents"])
    pair_ctry = ctry[ent_idx[pos]]

    def full(p_exp, mask=slice(None)):
        """Held-out probs with the exported pairs selected by mask set to p_exp."""
        p = h["prob"].copy()
        p[pos[mask]] = p_exp
        return p

    def scores(p, t):
        """Per-entity F0.5 when every pair at or above t is predicted."""
        return entity_scores(ent_idx, label, p >= t, k_true)

    def wmean(f, m=slice(None)):
        """Mean of per-entity scores under the test-mix weights."""
        return float(np.average(f[m], weights=w[m]))

    def grid(p, weighted=True, m=slice(None)):
        """Score at every threshold: weighted mean, or plain mean over entities m."""
        return {f"{t:.2f}": wmean(scores(p, t)) if weighted else float(scores(p, t)[m].mean())
                for t in THRESHOLDS}

    lgbm, blend_g = grid(h["prob"]), grid(blended)
    oofs, s2_grids = {}, {}
    for name, setting in SETTINGS.items():
        oofs[name] = cross_fit(X, y, fold, setting, a.threads)
        s2_grids[name] = grid(full(oofs[name]))
        print(f"setting {name}: best {max(s2_grids[name].values()):.5f} ({time.time() - t0:.0f}s)", flush=True)
    chosen = max(s2_grids, key=lambda n: max(s2_grids[n].values()))
    setting, oof_s = SETTINGS[chosen], oofs[chosen]
    staged, s2 = full(oof_s), s2_grids[chosen]
    best_b, best_s = max(blend_g, key=blend_g.get), max(s2, key=s2.get)
    tb, ts = float(best_b), float(best_s)
    base_key = f"{BASE_THRESHOLD:.2f}"
    unweighted = {"lgbm": grid(h["prob"], False), "blend": grid(blended, False), "stage2": grid(staged, False)}

    f_base, f_blend, f_s2 = scores(h["prob"], BASE_THRESHOLD), scores(blended, tb), scores(staged, ts)
    per_fold = [{"fold": k, "entities": int((efold == k).sum()), "lgbm_at_base": wmean(f_base, efold == k),
                 "blend_at_best": wmean(f_blend, efold == k), "stage2_at_best": wmean(f_s2, efold == k)}
                for k in range(FOLDS) if (efold == k).any()]

    # Both second stages refitted without a country, scored on it (same cross-encoder).
    refit = {}
    for c in sorted(weights):
        tr, te = pair_ctry != c, pair_ctry == c
        if len(np.unique(y[tr])) < 2 or not te.any():
            continue
        p_b = full(fit_lr(Xb[tr], y[tr]).predict_proba(Xb[te])[:, 1].astype(np.float32), te)
        p_s = full(predict(train(X[tr], y[tr], setting, a.threads), X[te]), te)
        refit[c] = {"blend": float(scores(p_b, tb)[ctry == c].mean()),
                    "stage2": float(scores(p_s, ts)[ctry == c].mean())}

    # Cross-encoder trained on some countries only: both second stages fitted
    # on those countries' pairs, scored on the others.
    unseen_check = None
    if a.loco_ce:
        ce_l, trained_l = load_ce(Path(a.loco_ce), digest, len(pos))
        seen = trained_l.get("train_countries") or []
        left = sorted(set(weights) - set(seen))
        if not seen or not left:
            raise SystemExit(f"{a.loco_ce}: trained on countries {seen or 'all'}; it leaves out "
                             f"none of the held-out countries {sorted(weights)}")
        tr = np.isin(pair_ctry, seen)
        X_l, Xb_l = features(rows, enc, pl, ce_l), design(pl, ce_l)
        te = ~tr
        p_b = full(fit_lr(Xb_l[tr], y[tr]).predict_proba(Xb_l[te])[:, 1].astype(np.float32), te)
        p_s = full(predict(train(X_l[tr], y[tr], setting, a.threads), X_l[te]), te)
        unseen_check = {"dir": str(a.loco_ce), "train_countries": seen, "trained_on": trained_l,
                        "countries": {c: {"lgbm": grid(h["prob"], False, ctry == c),
                                          "blend": grid(p_b, False, ctry == c),
                                          "stage2": grid(p_s, False, ctry == c),
                                          "blend_at_best": float(scores(p_b, tb)[ctry == c].mean()),
                                          "stage2_at_best": float(scores(p_s, ts)[ctry == c].mean())}
                                      for c in left}}

    labels_c = sorted(weights)
    code_ent = np.searchsorted(labels_c, ctry)
    flips = flip_rates(pl, oof_s, ts, code_ent[ent_idx[pos]], np.bincount(code_ent, minlength=len(labels_c)))
    per_country = {}
    for k, c in enumerate(labels_c):
        m = ctry == c
        loco = (unseen_check or {}).get("countries", {}).get(c, {})
        per_country[c] = {"entities": int(m.sum()), **weights[c],
                          "lgbm_at_base": float(f_base[m].mean()), "blend_at_best": float(f_blend[m].mean()),
                          "stage2_at_best": float(f_s2[m].mean()),
                          "blend_refit_without": refit.get(c, {}).get("blend"),
                          "stage2_refit_without": refit.get(c, {}).get("stage2"),
                          "blend_ce_loco": loco.get("blend_at_best"), "stage2_ce_loco": loco.get("stage2_at_best"),
                          "flips": {f: float(v[k]) for f, v in flips.items()}}

    gain = s2[best_s] - blend_g[best_b]
    every_fold = all(f["stage2_at_best"] > f["blend_at_best"] for f in per_fold)
    every_country = all(d[f"stage2_{v}"] is None or d[f"stage2_{v}"] >= d[f"blend_{v}"]
                        for d in per_country.values() for v in ("at_best", "refit_without", "ce_loco"))
    verdict = "adopt" if gain >= MIN_GAIN and every_fold and every_country else "reject"

    booster = train(X, y, setting, a.threads)
    gain_imp = booster.feature_importance("gain")
    split_imp = booster.feature_importance("split")
    importance = {f: {"gain_share": float(g / gain_imp.sum()), "splits": int(s)}
                  for f, g, s in sorted(zip(FEATURES, gain_imp, split_imp), key=lambda x: -x[1])}
    auc = {"lgbm": float(roc_auc_score(y, pl)), "ce": float(roc_auc_score(y, ce)),
           "blend_oof": float(roc_auc_score(y, oof_b)), "stage2_oof": float(roc_auc_score(y, oof_s))}

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    txt = out.with_suffix(".txt")
    booster.save_model(str(txt))
    report = {"booster_file": txt.name, "booster_sha256": sha256(txt), "features": FEATURES,
              "setting": chosen, "params": {**BASE_PARAMS, **setting}, "clip": CLIP, "squash": SQUASH,
              "threshold": ts, "blend_threshold": tb, "floor": floor, "calibrator": None,
              "verdict": verdict, "gain_vs_blend": gain, "every_fold_improves_vs_blend": every_fold,
              "no_country_below_blend": every_country, "min_gain": MIN_GAIN, "base_threshold": BASE_THRESHOLD,
              "weighting": "held-out entities reweighted to the test Source 1 country mix",
              "test_share_of_unseen_countries": unseen, "heldout_entities": int(len(h["ents"])),
              "scored_pairs": int(len(h["prob"])), "exported_pairs": int(len(pos)),
              "lgbm": lgbm, "blend": blend_g, "stage2": s2, "stage2_settings": s2_grids,
              "unweighted": unweighted, "per_fold": per_fold, "per_country": per_country,
              "unseen_country_check": unseen_check, "importance": importance, "auc_exported_pairs": auc,
              "cross_encoder_trained_on": trained,
              "sources": {"heldout_dump": str(a.heldout_dump), "pairs": str(a.pairs), "ce": str(a.ce)}}
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"setting {chosen} | {'thr':>5} {'lgbm':>9} {'blend':>9} {'stage2':>9}   "
          "(test-mix weighted; unweighted lgbm / blend / stage2)")
    for t in THRESHOLDS:
        k = f"{t:.2f}"
        print(f"{k:>22} {lgbm[k]:>9.5f} {blend_g[k]:>9.5f} {s2[k]:>9.5f}   "
              f"{unweighted['lgbm'][k]:.5f} / {unweighted['blend'][k]:.5f} / {unweighted['stage2'][k]:.5f}")
    for f in per_fold:
        print(f"fold {f['fold']} ({f['entities']:,}): lgbm@{base_key} {f['lgbm_at_base']:.5f} | "
              f"blend@{best_b} {f['blend_at_best']:.5f} | stage2@{best_s} {f['stage2_at_best']:.5f}")

    def num(v):
        """Score or n/a."""
        return "n/a" if v is None else f"{v:.5f}"
    for c, d in per_country.items():
        print(f"{c} ({d['entities']:,}, weight {d['weight']:.3f}): lgbm@{base_key} {d['lgbm_at_base']:.5f} | "
              f"blend {d['blend_at_best']:.5f} stage2 {d['stage2_at_best']:.5f} | refit without {c}: "
              f"blend {num(d['blend_refit_without'])} stage2 {num(d['stage2_refit_without'])} | "
              f"CE without {c}: blend {num(d['blend_ce_loco'])} stage2 {num(d['stage2_ce_loco'])} | "
              f"flips up {d['flips']['up_per_1k']:.2f} down {d['flips']['down_per_1k']:.2f} per 1k")
    if unseen_check:
        for c, d in unseen_check["countries"].items():
            print(f"unseen-country check, cross-encoder and second stages fitted on {unseen_check['train_countries']}, "
                  f"scored on {c} (unweighted):")
            for t in THRESHOLDS:
                k = f"{t:.2f}"
                print(f"{k:>10} lgbm {d['lgbm'][k]:.5f} | blend {d['blend'][k]:.5f} | stage2 {d['stage2'][k]:.5f}")
    else:
        print("no --loco-ce: nothing here measures either second stage with a cross-encoder that never saw a country")
    print(f"AUC on exported pairs: {auc}")
    print("gain importance: " + ", ".join(f"{f} {d['gain_share']:.3f}" for f, d in importance.items()))
    print(f"verdict: {verdict} (stage2@{best_s} {s2[best_s]:.5f} vs blend@{best_b} {blend_g[best_b]:.5f}, "
          f"gain {gain:+.5f}, every fold improves: {every_fold}, no country below: {every_country})")
    if best_s in (f"{THRESHOLDS[0]:.2f}", f"{THRESHOLDS[-1]:.2f}"):
        print(f"stage-2 threshold {best_s} is at the edge of the grid")
    print(f"threshold: {best_s}")
    print(f"wrote {out} and {txt} ({time.time() - t0:.0f}s)")
    return report


def cmd_apply(a):
    """Check the saved model against the export, then run ce_blend apply with stage-2 probs."""
    import lightgbm as lgb
    model = json.loads(Path(a.model).read_text())
    txt = Path(a.model).with_name(model["booster_file"])
    if sha256(txt) != model["booster_sha256"]:
        raise SystemExit(f"{txt} is not the booster {a.model} was written with")
    if model["features"] != FEATURES:
        raise SystemExit(f"{a.model} was fitted on features {model['features']}, this script builds {FEATURES}")
    floor = json.loads((Path(a.pairs) / "manifest.json").read_text()).get("floor")
    if floor != model["floor"]:
        raise SystemExit(f"test export floor {floor} differs from the held-out floor {model['floor']}; "
                         "the entity counts would not mean the same")
    booster = lgb.Booster(model_file=str(txt))
    return ce_blend.cmd_apply(a, rescore=lambda m, r, e, p, c: predict(booster, features(r, e, p, c)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--heldout-dump", required=True, help="loss_breakdown.py --dump pickle")
    f.add_argument("--pairs", required=True, help="ce_pairs.py export folder")
    f.add_argument("--ce", required=True, help="folder with ce_heldout.npy and ce_log.json")
    f.add_argument("--loco-ce", default=None,
                   help="the same for a cross-encoder trained on some countries only; both second stages "
                        "are fitted on those countries and scored on the others")
    f.add_argument("--threads", type=int, default=2, help="LightGBM threads")
    f.add_argument("--out", required=True, help="report JSON; the booster goes next to it as .txt")
    ap = sub.add_parser("apply")
    ap.add_argument("--model", required=True, help="JSON written by fit")
    ap.add_argument("--scores", required=True, help="raw test score dump folder")
    ap.add_argument("--pairs", required=True, help="folder with test_pairs.npz, shards.json and manifest.json")
    ap.add_argument("--ce", required=True, help="folder with ce_test.npy")
    ap.add_argument("--unseen-keep", action="store_true",
                    help="pairs of test countries absent from held-out keep their LightGBM prob")
    ap.add_argument("--out", required=True, help="new score dump folder")
    args = p.parse_args()
    cmd_fit(args) if args.cmd == "fit" else cmd_apply(args)
