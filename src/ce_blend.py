"""Blend LightGBM pair probabilities with cross-encoder probabilities.

fit    learns p = sigmoid(b + w1*l + w2*c + w3*l*c), where l and c are the
       logits of the LightGBM and cross-encoder probabilities, on the held-out
       pairs exported by ce_pairs.py. Five entity folds give every exported
       pair an out-of-fold blended prob; pairs below the export floor keep
       their LightGBM prob. Macro F0.5 is taken over every held-out entity,
       including those with no pair above the floor and those with no
       candidate at all. Held-out entities are weighted so each country
       counts as much as it does among the test Source 1 records of the
       countries held-out covers (test country column, no labels); the
       threshold, the gain and the per-fold check use that weighted mean.
       The cross-encoder output is used only if its ce_log.json records the
       digest of these held-out pairs as the ones its training pairs were
       checked against (ce_rerank.py trained_on), so no held-out entity or
       target record was trained on.
       Two leave-one-country-out checks, at the chosen threshold:
       blend_coef_loco refits only the four blend coefficients without a
       country and scores it; the cross-encoder under it was trained on every
       country, so this shows whether the coefficients carry over, not how
       the cross-encoder does on a country it never saw. --loco-ce gives
       that: output of a cross-encoder trained only on some countries
       (ce_pairs.py export --train-countries), blended with coefficients
       fitted on those countries' pairs and scored on the others. The
       verdict is adopt only if the weighted gain over LightGBM at 0.75 is at
       least MIN_GAIN, every fold improves, and no country scores below
       LightGBM at 0.75 in the plain blend, the coefficient refit or, when
       given, the --loco-ce blend. Without --loco-ce nothing measures the
       cross-encoder on an unseen country; the report says so. Writes the
       report, the coefficients refitted on all exported pairs, the chosen
       threshold and the verdict to one JSON file.
apply  copies a test score dump, replacing prob with the blended prob for the
       exported pairs and leaving every other value untouched, so decide.py
       runs on the new directory unchanged. It also reports, per test
       country, the pairs the blend at the chosen threshold admits that
       LightGBM at 0.75 does not (up) and the reverse (down), per 1,000
       Source 1 records, next to the held-out rates from fit: a label-free
       check that the cross-encoder moves an unseen country about as much as
       the countries it was measured on.

The held-out dump holds raw booster output, so the test dump must be raw too
(predict_matches.py --no-calibrate, same features as the held-out run); fit
and apply refuse step-valued (isotonic) probs. fit --calibrator instead maps
every held-out LightGBM prob through a test run's isotonic map (after the
export is checked against the raw dump, so the exported pairs stay the ones at
or above the floor on the raw scale) and records the map's path and sha256;
apply then takes only a test export marked calibrated with that same map, and
a raw model only an uncalibrated one. apply --unseen-keep leaves the LightGBM
prob of test pairs whose Source 1 country is not among the model's held-out
countries. Above 1 - CLIP the blended
logit is compressed by SQUASH before the float32 sigmoid, so confident pairs
keep distinct probs and decide.py's target argmax still ranks them by score
instead of tying them at 1.0; every threshold below 1 - CLIP admits the same
pairs.

Run:
    python src/ce_blend.py fit --heldout-dump outputs/cache/cv4m5_addridf.pkl \
        --pairs outputs/ce/export --ce outputs/ce/run --out results/ce_blend/blend.json
    # with a cross-encoder trained on US pairs only, scored on India:
    python src/ce_blend.py fit --heldout-dump outputs/cache/cv4m5_addridf.pkl \
        --pairs outputs/ce/export --ce outputs/ce/run --loco-ce outputs/ce/run_us \
        --out results/ce_blend/blend.json
    python src/ce_blend.py apply --model results/ce_blend/blend.json \
        --scores outputs/scores/test_v8_raw --pairs outputs/ce/export --ce outputs/ce/run \
        --out outputs/scores/test_v8_ce
    # on a calibrated test run (ce_pairs.py export --allow-calibrated):
    python src/ce_blend.py fit ... --calibrator outputs/models/sub08_addridf/calibration.pkl \
        --out results/ce_blend/blend_cal.json
    python src/ce_blend.py apply --model results/ce_blend/blend_cal.json --scores outputs/scores/test_v8 \
        --pairs outputs/ce/export --ce outputs/ce/run --unseen-keep --out outputs/scores/test_v8_ce
    python src/decide.py --scores outputs/scores/test_v8_ce --split test \
        --rule argmax_threshold --threshold <printed threshold> --run-name <run>
"""
import argparse
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import er_io
from src.ce_pairs import MIN_DISTINCT, distinct_frac, load_heldout as read_dump, pairs_digest, sha256

ENCODE = 2_000_000_000
CLIP = 1e-4
FOLDS = 5
THRESHOLDS = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
BASE_THRESHOLD = 0.75
MIN_GAIN = 0.001
TOL = 1e-6
SQUASH = 4.0


def logit(p):
    """Log-odds of probabilities clipped to [CLIP, 1 - CLIP]."""
    p = np.clip(np.asarray(p, dtype=np.float64), CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def design(p_lgbm, p_ce):
    """Blend inputs: both logits and their product."""
    lg, ce = logit(p_lgbm), logit(p_ce)
    return np.column_stack([lg, ce, lg * ce])


def blend(model, p_lgbm, p_ce):
    """Blended probability under saved coefficients, as float32.

    Logits above logit(1 - CLIP) are compressed by SQUASH: the order is kept,
    probs below 1 - CLIP are exact, and float32 stays below 1.0 up to a logit
    of about 42 instead of 17.
    """
    z = float(model["intercept"]) + design(p_lgbm, p_ce) @ np.asarray(model["coef"], dtype=np.float64)
    top = np.log((1 - CLIP) / CLIP)
    z = np.where(z > top, top + (z - top) / SQUASH, z)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def fold_of(rows):
    """Entity fold in 0..FOLDS-1 from a multiplicative hash of the S1 row."""
    h = (np.asarray(rows, dtype=np.uint64) * np.uint64(2654435761)) % np.uint64(1 << 32)
    return (h % np.uint64(FOLDS)).astype(np.int8)


def pair_key(s1_row, enc_id):
    """One sortable int64 per (S1 row, target) pair; enc_id is below 4 * ENCODE."""
    return np.asarray(s1_row, dtype=np.int64) * np.int64(4 * ENCODE) + np.asarray(enc_id, dtype=np.int64)


def load_heldout(path):
    """Flatten a loss_breakdown dump into held-out entities and every scored pair.

    The dump is read without raw_cands, which is not needed here.
    Pairs are sorted by S1 row then enc_id. Returns ents (sorted held-out rows),
    k_true (true match count per entity), country (label per entity), and per
    pair s1_row, enc_id, prob (float32) and label.
    """
    res = read_dump(path)
    ents = np.array(sorted(res["holdout_set"]), dtype=np.int64)
    truth = res["truth"]
    rows, ids, probs = [], [], []
    for r, lst in res["val_by_ent"].items():
        for p, i in lst:
            rows.append(r)
            ids.append(i)
            probs.append(p)
    s1_row = np.array(rows, dtype=np.int64)
    enc_id = np.array(ids, dtype=np.int64)
    prob = np.array(probs, dtype=np.float32)
    order = np.lexsort((enc_id, s1_row))
    s1_row, enc_id, prob = s1_row[order], enc_id[order], prob[order]
    if np.any(np.diff(pair_key(s1_row, enc_id)) == 0):
        raise SystemExit("duplicate (s1_row, enc_id) pairs in the held-out dump")
    label = np.fromiter((i in truth[r] for r, i in zip(s1_row.tolist(), enc_id.tolist())),
                        dtype=bool, count=len(s1_row))
    pos = np.searchsorted(ents, s1_row)
    if len(s1_row) and (pos.max() >= len(ents) or np.any(ents[np.minimum(pos, len(ents) - 1)] != s1_row)):
        raise SystemExit("held-out dump has pairs for entities outside holdout_set")
    k_true = np.array([len(truth[r]) for r in ents.tolist()], dtype=np.float64)
    labels = np.array(res["countries"], dtype=object)
    country = labels[np.asarray(res["country"])[ents]]
    return {"ents": ents, "k_true": k_true, "country": country, "ent_idx": pos,
            "s1_row": s1_row, "enc_id": enc_id, "prob": prob, "label": label}


def entity_scores(ent_idx, label, pred, k_true):
    """Per-entity F0.5 from per-pair predictions; empty-empty scores 1.0.

    Entities with no predicted pair (including those with no pair at all)
    score 1.0 when they have no true match and 0.0 otherwise.
    """
    n = np.bincount(ent_idx, weights=pred, minlength=len(k_true))
    c = np.bincount(ent_idx, weights=pred & label, minlength=len(k_true))
    f = np.zeros(len(k_true))
    hit = c > 0
    f[hit] = 1.25 * c[hit] / (n[hit] + 0.25 * k_true[hit])
    f[(n == 0) & (k_true == 0)] = 1.0
    return f


def align(h, pairs_dir: Path):
    """Positions of the exported held-out pairs inside h, checked against the dump.

    Every exported pair must exist in the dump with the same prob and label,
    and the export must hold exactly the pairs at or above the manifest floor.
    """
    hp = np.load(pairs_dir / "heldout_pairs.npz")
    key_all = pair_key(h["s1_row"], h["enc_id"])
    key_exp = pair_key(hp["s1_row"], hp["enc_id"])
    if np.any(np.diff(key_exp) <= 0):
        raise SystemExit("heldout_pairs.npz is not sorted by s1_row then enc_id")
    pos = np.searchsorted(key_all, key_exp)
    if len(pos) and (pos.max() >= len(key_all) or np.any(key_all[pos] != key_exp)):
        raise SystemExit("heldout_pairs.npz holds pairs missing from the held-out dump")
    if np.abs(h["prob"][pos] - hp["prob"]).max(initial=0.0) > TOL:
        raise SystemExit("heldout_pairs.npz prob differs from the held-out dump")
    if np.any(h["label"][pos] != hp["label"].astype(bool)):
        raise SystemExit("heldout_pairs.npz label differs from the held-out truth")
    floor = json.loads((pairs_dir / "manifest.json").read_text()).get("floor")
    if floor is not None:
        rest = np.ones(len(key_all), dtype=bool)
        rest[pos] = False
        if np.any(h["prob"][rest] >= floor + TOL) or np.any(h["prob"][pos] < floor - TOL):
            raise SystemExit(f"heldout_pairs.npz is not exactly the pairs with prob >= {floor}")
    return pos, floor


def load_ce(ce_dir: Path, digest: str, n: int):
    """Cross-encoder held-out probs (float32) and the ce_train.json record of the model behind them.

    Refused unless ce_log.json records trained_on with the digest of the
    held-out pairs being blended, and ce_heldout.npy holds n values.
    """
    log = ce_dir / "ce_log.json"
    if not log.exists():
        raise SystemExit(f"no ce_log.json in {ce_dir}: can not tell what its model was trained on")
    trained = json.loads(log.read_text(encoding="utf-8")).get("trained_on") or {}
    if trained.get("heldout_digest") != digest:
        raise SystemExit(f"{ce_dir}: the cross-encoder was not checked against these held-out pairs "
                         f"(trained_on digest {trained.get('heldout_digest')}, pairs {digest})")
    ce = np.load(ce_dir / "ce_heldout.npy").astype(np.float32)
    if len(ce) != n:
        raise SystemExit(f"{ce_dir}/ce_heldout.npy has {len(ce):,} values for {n:,} exported pairs")
    return ce, trained


def flip_rates(p_lgbm, p_blend, thr, code, n_ent):
    """Per country code: exported pairs the blend at thr admits that LightGBM at BASE_THRESHOLD rejects (up), and the reverse (down).

    Each as a count per 1,000 entities of the country, and as a share of the
    exported pairs LightGBM rejects (up) or admits (down) there; a country
    with more candidates per entity has more pairs that can flip, which the
    share leaves out. Returns {name: array over country codes}.
    """
    lo, hi = p_lgbm < BASE_THRESHOLD, p_blend >= thr

    def count(flag):
        """Flagged pairs per country code."""
        return np.bincount(code, weights=flag, minlength=len(n_ent))
    up, down = count(lo & hi), count(~lo & ~hi)
    return {"up_per_1k": 1000 * up / np.maximum(n_ent, 1), "down_per_1k": 1000 * down / np.maximum(n_ent, 1),
            "up_share": up / np.maximum(count(lo), 1), "down_share": down / np.maximum(count(~lo), 1)}


def country_weights(country):
    """Per-entity weights that give each held-out country its test share.

    A country's test share is its count of test Source 1 records over the
    count of those of every country present in held-out; each entity gets
    that share over its country's held-out share, so the weights average 1.
    Returns the weights and a per-country table, plus the test share (of all
    test Source 1 records) of countries held-out lacks.
    """
    t = er_io.parse_source("test", 1)
    n_test = dict(zip(t["countries"], np.bincount(t["country"], minlength=len(t["countries"])).tolist()))
    labels, inv, n_held = np.unique(country.astype(str), return_inverse=True, return_counts=True)
    covered = sum(n_test.get(c, 0) for c in labels)
    if covered == 0:
        raise SystemExit(f"no test Source 1 record is in a held-out country {labels.tolist()}")
    share = np.array([n_test.get(c, 0) / covered for c in labels])
    w_c = share / (n_held / len(country))
    table = {c: {"heldout_share": float(n / len(country)), "test_share": float(sh), "weight": float(w)}
             for c, n, sh, w in zip(labels.tolist(), n_held, share, w_c)}
    total = sum(n_test.values())
    unseen = {c: n / total for c, n in n_test.items() if c not in table}
    return w_c[inv], table, unseen


def fit_lr(X, y):
    """Logistic regression on the blend inputs."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, max_iter=2000).fit(X, y)


def oof_blend(h, pos, ce):
    """Out-of-fold blended prob for every held-out pair; pairs not exported keep LightGBM."""
    X = design(h["prob"][pos], ce)
    y = h["label"][pos]
    fold = fold_of(h["s1_row"][pos])
    oof = np.empty(len(y))
    for k in range(FOLDS):
        te = fold == k
        if te.any():
            oof[te] = fit_lr(X[~te], y[~te]).predict_proba(X[te])[:, 1]
    out = h["prob"].copy()
    out[pos] = oof.astype(np.float32)
    return out, X, y, oof


def cmd_fit(a):
    """Cross-fit the blend on the held-out pairs, report it and save the model JSON."""
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    h = load_heldout(a.heldout_dump)
    if distinct_frac(h["prob"]) < MIN_DISTINCT:
        raise SystemExit("held-out probs are step-valued (isotonic); the blend expects raw booster output")
    pos, floor = align(h, Path(a.pairs))
    calibrator = None
    if a.calibrator:
        with open(a.calibrator, "rb") as f:
            iso = pickle.load(f)["iso"]
        if iso is None:
            raise SystemExit(f"{a.calibrator} holds no isotonic map")
        h["prob"] = iso.predict(h["prob"]).astype(np.float32)
        calibrator = {"path": str(Path(a.calibrator).resolve()), "sha256": sha256(Path(a.calibrator))}
        print(f"LightGBM probs mapped through {a.calibrator}", flush=True)
    with np.load(Path(a.pairs) / "heldout_pairs.npz") as hp:
        digest = pairs_digest(hp["s1_row"], hp["enc_id"])
    ce, trained = load_ce(Path(a.ce), digest, len(pos))
    blended, X, y, oof = oof_blend(h, pos, ce)
    print(f"held-out entities {len(h['ents']):,} | scored pairs {len(h['prob']):,} | "
          f"exported {len(pos):,} (floor {floor}) | positives {int(y.sum()):,} "
          f"({time.time() - t0:.0f}s)", flush=True)

    ent_idx, label, k_true = h["ent_idx"], h["label"], h["k_true"]
    w, weights, unseen = country_weights(h["country"])
    ctry = h["country"].astype(str)
    efold = fold_of(h["ents"])

    def scores(p, t):
        """Per-entity F0.5 when every pair at or above t is predicted."""
        return entity_scores(ent_idx, label, p >= t, k_true)

    def wmean(f, m=slice(None)):
        """Mean of per-entity scores under the test-mix weights."""
        return float(np.average(f[m], weights=w[m]))

    lgbm = {f"{t:.2f}": wmean(scores(h["prob"], t)) for t in THRESHOLDS}
    mixed = {f"{t:.2f}": wmean(scores(blended, t)) for t in THRESHOLDS}
    lgbm_plain = {f"{t:.2f}": float(scores(h["prob"], t).mean()) for t in THRESHOLDS}
    mixed_plain = {f"{t:.2f}": float(scores(blended, t).mean()) for t in THRESHOLDS}
    best = max(mixed, key=mixed.get)
    own = max(lgbm, key=lgbm.get)
    base_key = f"{BASE_THRESHOLD:.2f}"
    f_base = scores(h["prob"], BASE_THRESHOLD)
    f_lgbm_best = scores(h["prob"], float(best))
    f_lgbm_own = scores(h["prob"], float(own))
    f_blend = scores(blended, float(best))

    per_fold = [{"fold": k, "entities": int((efold == k).sum()),
                 "lgbm_at_base": wmean(f_base, efold == k),
                 "lgbm_at_best": wmean(f_lgbm_best, efold == k),
                 "lgbm_at_own_best": wmean(f_lgbm_own, efold == k),
                 "blend_at_best": wmean(f_blend, efold == k)}
                for k in range(FOLDS) if (efold == k).any()]

    # Coefficient refit without a country, scored on it (same cross-encoder).
    pair_ctry = ctry[ent_idx[pos]]
    coef_loco = {}
    for c in sorted(weights):
        tr = pair_ctry != c
        if len(np.unique(y[tr])) < 2:
            continue
        p_loco = h["prob"].copy()
        te = ~tr
        if te.any():
            p_loco[pos[te]] = fit_lr(X[tr], y[tr]).predict_proba(X[te])[:, 1].astype(np.float32)
        coef_loco[c] = float(scores(p_loco, float(best))[ctry == c].mean())

    # Cross-encoder trained without some countries: blend fitted on the
    # countries it saw, scored on the ones it did not.
    ce_loco, loco_info = {}, None
    if a.loco_ce:
        ce_l, trained_l = load_ce(Path(a.loco_ce), digest, len(pos))
        seen = trained_l.get("train_countries") or []
        left = sorted(set(weights) - set(seen))
        if not seen or not left:
            raise SystemExit(f"{a.loco_ce}: trained on countries {seen or 'all'}; it leaves out "
                             f"none of the held-out countries {sorted(weights)}")
        X_l = design(h["prob"][pos], ce_l)
        tr = np.isin(pair_ctry, seen)
        if len(np.unique(y[tr])) < 2:
            raise SystemExit(f"{a.loco_ce}: pairs of {seen} hold one class only")
        m_l = fit_lr(X_l[tr], y[tr])
        for c in left:
            te = pair_ctry == c
            p_l = h["prob"].copy()
            if te.any():
                p_l[pos[te]] = m_l.predict_proba(X_l[te])[:, 1].astype(np.float32)
            ce_loco[c] = float(scores(p_l, float(best))[ctry == c].mean())
        loco_info = {"dir": str(a.loco_ce), "train_countries": seen, "scored_countries": left,
                     "intercept": float(m_l.intercept_[0]), "coef": m_l.coef_[0].tolist(),
                     "trained_on": trained_l}

    # Exported pairs whose admission changes: LightGBM at the base threshold
    # against the blend at the chosen one.
    labels_c = sorted(weights)
    code_ent = np.searchsorted(labels_c, ctry)
    flips = flip_rates(h["prob"][pos], blended[pos], float(best), code_ent[ent_idx[pos]],
                       np.bincount(code_ent, minlength=len(labels_c)))

    per_country = {c: {"entities": int((ctry == c).sum()), **weights[c],
                       "lgbm_at_base": float(f_base[ctry == c].mean()),
                       "lgbm_at_own_best": float(f_lgbm_own[ctry == c].mean()),
                       "blend_at_best": float(f_blend[ctry == c].mean()),
                       "blend_coef_loco_at_best": coef_loco.get(c),
                       "ce_loco_at_best": ce_loco.get(c),
                       "flips": {f: float(v[k]) for f, v in flips.items()}}
                   for k, c in enumerate(labels_c)}

    def holds(ref):
        """No country's blend (plain, coefficient refit, or --loco-ce) is below LightGBM under ref."""
        return all(all(d[f] is None or d[f] >= d[ref]
                       for f in ("blend_at_best", "blend_coef_loco_at_best", "ce_loco_at_best"))
                   for d in per_country.values())

    gain = mixed[best] - lgbm[base_key]
    every_fold = all(f["blend_at_best"] > f["lgbm_at_base"] for f in per_fold)
    every_country = holds("lgbm_at_base")
    verdict = "adopt" if gain >= MIN_GAIN and every_fold and every_country else "reject"
    # The same rule against LightGBM at its own best grid threshold, so a gain
    # that only comes from moving the threshold is not credited to the blend.
    gain_own = mixed[best] - lgbm[own]
    every_fold_own = all(f["blend_at_best"] > f["lgbm_at_own_best"] for f in per_fold)
    every_country_own = holds("lgbm_at_own_best")
    verdict_own = ("adopt" if gain_own >= MIN_GAIN and every_fold_own and every_country_own
                   else "reject")

    final = fit_lr(X, y)
    auc = {}
    if 0 < y.sum() < len(y):
        auc = {"lgbm": float(roc_auc_score(y, h["prob"][pos])), "ce": float(roc_auc_score(y, ce)),
               "blend_oof": float(roc_auc_score(y, oof))}
    report = {"intercept": float(final.intercept_[0]), "coef": final.coef_[0].tolist(),
              "features": ["logit_lgbm", "logit_ce", "logit_lgbm*logit_ce"], "clip": CLIP,
              "threshold": float(best), "verdict": verdict, "gain_vs_base": gain,
              "every_fold_improves": every_fold, "no_country_below_base": every_country,
              "base_threshold": BASE_THRESHOLD,
              "lgbm_best_threshold": float(own), "verdict_vs_lgbm_best": verdict_own,
              "gain_vs_lgbm_best": gain_own, "every_fold_improves_vs_lgbm_best": every_fold_own,
              "no_country_below_lgbm_best": every_country_own,
              "weighting": "held-out entities reweighted to the test Source 1 country mix",
              "test_share_of_unseen_countries": unseen,
              "floor": floor, "heldout_entities": int(len(h["ents"])),
              "scored_pairs": int(len(h["prob"])), "exported_pairs": int(len(pos)),
              "entities_without_exported_pair": int(len(h["ents"]) - len(np.unique(h["ent_idx"][pos]))),
              "lgbm": lgbm, "blend": mixed, "lgbm_unweighted": lgbm_plain, "blend_unweighted": mixed_plain,
              "per_fold": per_fold, "per_country": per_country,
              "ce_loco": loco_info, "cross_encoder_trained_on": trained,
              "auc_exported_pairs": auc, "calibrator": calibrator, "sources": {"heldout_dump": str(a.heldout_dump),
                                                     "pairs": str(a.pairs), "ce": str(a.ce)}}

    print(f"{'threshold':>10} {'lgbm':>9} {'blend':>9}   (test-mix weighted; unweighted lgbm / blend)")
    for t in lgbm:
        print(f"{t:>10} {lgbm[t]:>9.5f} {mixed[t]:>9.5f}   {lgbm_plain[t]:.5f} / {mixed_plain[t]:.5f}")
    for f in per_fold:
        print(f"fold {f['fold']} ({f['entities']:,}): lgbm@{base_key} {f['lgbm_at_base']:.5f} | "
              f"lgbm@{best} {f['lgbm_at_best']:.5f} | blend@{best} {f['blend_at_best']:.5f}")
    for c, d in per_country.items():
        lo = "n/a" if d["blend_coef_loco_at_best"] is None else f"{d['blend_coef_loco_at_best']:.5f}"
        cl = "" if d["ce_loco_at_best"] is None else f" | CE without {c} @{best} {d['ce_loco_at_best']:.5f}"
        print(f"{c} ({d['entities']:,}, weight {d['weight']:.3f}): lgbm@{base_key} {d['lgbm_at_base']:.5f} | "
              f"blend@{best} {d['blend_at_best']:.5f} | coefficients without {c} @{best} {lo}{cl} | "
              f"flips up {d['flips']['up_per_1k']:.2f} down {d['flips']['down_per_1k']:.2f} per 1k entities, "
              f"up {d['flips']['up_share']:.4f} of rejected / down {d['flips']['down_share']:.4f} of admitted pairs")
    if loco_info is None:
        print(f"no --loco-ce: the cross-encoder was trained on {trained.get('train_countries') or 'every country'}; "
              "nothing here measures it on a country it never saw")
    print(f"test countries absent from held-out (share of test S1): {unseen}")
    print(f"AUC on exported pairs: {auc}")
    print(f"coefficients: intercept {report['intercept']:.4f} coef {report['coef']}")
    print(f"verdict: {verdict} (blend@{best} {mixed[best]:.5f} vs lgbm@{base_key} "
          f"{lgbm[base_key]:.5f}, gain {gain:+.5f}, every fold improves: {every_fold}, "
          f"no country below: {every_country})")
    print(f"verdict vs lgbm at its best threshold {own}: {verdict_own} "
          f"(gain {gain_own:+.5f}, every fold improves: {every_fold_own}, "
          f"no country below: {every_country_own})")
    print(f"threshold: {best}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {a.out}")
    return report


def admission_drift(model, s1_row, p_lgbm, p_blend):
    """flip_rates per test country, next to the held-out rates fit stored for it (None if held-out lacks it)."""
    t = er_io.parse_source("test", 1)
    n_ent = np.bincount(t["country"], minlength=len(t["countries"]))
    rates = flip_rates(p_lgbm, p_blend, float(model["threshold"]), t["country"][s1_row], n_ent)
    held = model.get("per_country", {})
    return {c: {"entities": int(n_ent[k]), "test": {f: float(v[k]) for f, v in rates.items()},
                "heldout": held.get(c, {}).get("flips")}
            for k, c in enumerate(t["countries"])}


def cmd_apply(a, rescore=None):
    """Write a copy of a test score dump with blended probs for the exported pairs.

    Shards are written to a .partial sibling that is renamed to --out only after
    every shard passed its checks, so a failed run never leaves a partial dump
    that decide.py would read as complete. rescore(model, s1_row, enc_id, prob,
    ce), when given, returns the new probs of the exported pairs in place of
    the logistic blend; every check and the write path stay the same.
    """
    model = json.loads(Path(a.model).read_text())
    scores, pairs, out = Path(a.scores), Path(a.pairs), Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} already exists")
    names = json.loads((pairs / "shards.json").read_text())
    on_disk = [f.name for f in sorted(scores.glob("shard_*.npz"))]
    if names != on_disk:
        raise SystemExit(f"shards.json ({len(names)} files) does not match {scores} ({len(on_disk)} files)")
    man = json.loads((pairs / "manifest.json").read_text())
    floor, run = man.get("floor"), man.get("test_run") or {}
    cal = model.get("calibrator")
    if bool(cal) != bool(run.get("calibrated")):
        raise SystemExit(f"scale mismatch: model calibrator {cal and cal['path']}, "
                         f"test export calibrated {run.get('calibrated')}")
    if cal and cal["sha256"] != run.get("calibration_sha256"):
        raise SystemExit(f"model calibrator sha256 {cal['sha256']} is not the test run's "
                         f"{run.get('calibration_sha256')}")
    tp = np.load(pairs / "test_pairs.npz")
    shard, pos = tp["shard"].astype(np.int64), tp["pos"].astype(np.int64)
    t_row, t_enc, t_prob = tp["s1_row"], tp["enc_id"], tp["prob"]
    ce = np.load(Path(a.ce) / "ce_test.npy")
    if len(ce) != len(shard):
        raise SystemExit(f"ce_test.npy has {len(ce):,} values for {len(shard):,} exported pairs")
    if len(shard) and (shard.min() < 0 or shard.max() >= len(names)):
        raise SystemExit("test_pairs.npz shard index out of range")
    new = (rescore(model, t_row, t_enc, t_prob, ce) if rescore
           else blend(model, t_prob, ce.astype(np.float32)))
    t = er_io.parse_source("test", 1)
    code = t["country"][t_row]
    held = [k for k, c in enumerate(t["countries"]) if c in model.get("per_country", {})]
    mix = np.isin(code, held) if a.unseen_keep else np.ones(len(code), dtype=bool)
    new = np.where(mix, new, t_prob)
    n_mix = np.bincount(code, weights=mix, minlength=len(t["countries"])).astype(int)
    n_all = np.bincount(code, minlength=len(t["countries"]))
    split = {c: {"blended": int(n_mix[k]), "kept": int(n_all[k] - n_mix[k])}
             for k, c in enumerate(t["countries"])}
    for c, d in split.items():
        print(f"  {c}: blended {d['blended']:,} | kept LightGBM {d['kept']:,} exported pairs")
    drift = admission_drift(model, t_row, t_prob, new)
    order = np.argsort(shard, kind="stable")
    bounds = np.searchsorted(shard[order], np.arange(len(names) + 1))

    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    total, changed, at_one = 0, 0, 0
    for i, name in enumerate(names):
        b = np.load(scores / name)
        s1_row, enc_id, prob = b["s1_row"], b["enc_id"], b["prob"]
        if not cal and distinct_frac(prob) < MIN_DISTINCT:
            raise SystemExit(f"{name}: step-valued (isotonic) probs; the blend was fitted on raw ones")
        sel = order[bounds[i]:bounds[i + 1]]
        p = pos[sel]
        if len(p) and (p.min() < 0 or p.max() >= len(prob) or len(np.unique(p)) != len(p)):
            raise SystemExit(f"{name}: exported positions out of range or repeated")
        if (np.any(s1_row[p] != t_row[sel]) or np.any(enc_id[p] != t_enc[sel])
                or np.any(prob[p] != t_prob[sel])):
            raise SystemExit(f"{name}: exported (shard, pos) does not map to the same pair")
        if floor is not None and not ((prob >= floor + TOL).sum() <= len(p) <= (prob >= floor - TOL).sum()):
            raise SystemExit(f"{name}: {len(p):,} exported pairs, but the shard has a different "
                             f"number at or above the floor {floor}")
        new_prob = prob.copy()
        new_prob[p] = new[sel]
        # Pairs rescored by the cross-encoder blend: the final stage's inference set,
        # which decide.py --candidates final-stage writes as candidate_pairs.tsv.
        stage2 = np.zeros(len(prob), dtype=bool)
        stage2[p[mix[sel]]] = True
        np.savez(tmp / name, s1_row=s1_row, enc_id=enc_id, prob=new_prob, stage2=stage2)
        total += len(prob)
        changed += int(mix[sel].sum())
        at_one += int((new[sel][mix[sel]] >= 1.0).sum())
    summary = {"model": str(a.model), "scores": str(scores), "shards": len(names),
               "pairs": total, "blended_pairs": changed, "kept_pairs": int(len(shard) - changed),
               "blended_at_1": at_one, "unseen_keep": bool(a.unseen_keep), "per_country": split,
               "calibrator": cal,
               "threshold": model["threshold"],
               "verdict": model.get("verdict"), "admission_flips": drift}
    (tmp / "blend_apply.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tmp.rename(out)
    print(json.dumps({k: v for k, v in summary.items() if k != "admission_flips"}, indent=2))
    print(f"admission flips, blend@{model['threshold']:.2f} vs lgbm@{BASE_THRESHOLD}: per 1k entities "
          "(up/down) and share of LightGBM-rejected/admitted exported pairs (up/down)")

    def show(r):
        """One side of a drift line."""
        return (f"{r['up_per_1k']:.2f}/{r['down_per_1k']:.2f} per 1k, "
                f"{r['up_share']:.4f}/{r['down_share']:.4f} of pairs")
    for c, d in drift.items():
        ref = "not in held-out" if d["heldout"] is None else f"held-out {show(d['heldout'])}"
        print(f"  {c} ({d['entities']:,}): test {show(d['test'])} | {ref}")
    print(f"wrote {out}")
    print(f"threshold: {model['threshold']:.2f}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--heldout-dump", required=True, help="loss_breakdown.py --dump pickle")
    f.add_argument("--pairs", required=True, help="ce_pairs.py export folder")
    f.add_argument("--ce", required=True, help="folder with ce_heldout.npy and ce_log.json")
    f.add_argument("--loco-ce", default=None,
                   help="the same for a cross-encoder trained on some countries only (ce_pairs.py "
                        "--train-countries); scored on the held-out countries it left out")
    f.add_argument("--calibrator", default=None,
                   help="calibration.pkl of the test run; held-out LightGBM probs go through its isotonic map")
    f.add_argument("--out", required=True, help="model and report JSON")
    ap = sub.add_parser("apply")
    ap.add_argument("--model", required=True, help="JSON written by fit")
    ap.add_argument("--scores", required=True, help="test score dump folder")
    ap.add_argument("--pairs", required=True, help="folder with test_pairs.npz and shards.json")
    ap.add_argument("--ce", required=True, help="folder with ce_test.npy")
    ap.add_argument("--unseen-keep", action="store_true",
                    help="pairs of test countries absent from held-out keep their LightGBM prob")
    ap.add_argument("--out", required=True, help="new score dump folder")
    args = p.parse_args()
    cmd_fit(args) if args.cmd == "fit" else cmd_apply(args)
