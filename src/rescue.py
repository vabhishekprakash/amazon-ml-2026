"""Rescue stage: find true matches the main blocking never generates, without touching its lists.

The main pipeline keeps each entity's top-40 pre-ranked candidates. Adding more key
families to that competition recovers missing copies but evicts true matches already
found, so this stage runs separately and only adds:

  1. Candidates from three extra key families (pair_model.RESCUE_FIELDS: look-alike
     digit repair, domain-style name prefix, name-token x address-token cross key),
     minus every pair the main blocking already generated.
  2. Records already assigned by the main predictions are excluded, so no accepted
     match is ever changed or reassigned.
  3. Each entity keeps its top RESCUE_TOP candidates by name + address trigram overlap.
  4. A LightGBM on the ordinary pair features, trained on the same candidate
     distribution of training entities, scores them; each record goes to its
     highest-scoring entity and is accepted at the threshold chosen on an inner split
     (lowest value with precision >= 0.85).

Training countries (fit --countries) and rescued countries (apply --countries) are separate, recorded settings; both default to the countries with training labels.

Run:
    python src/rescue.py fit --out outputs/models/rescue --exclude outputs/cache/cv4m5_g127_pseudo.pkl
    python src/rescue.py apply --model outputs/models/rescue --base outputs/submissions/final \\
        --out outputs/submissions/final_rescue
    python src/rescue.py evaluate --model outputs/models/rescue_us --dump outputs/loco_base.pkl --threshold 0.80

evaluate adds the rescue stage to the predictions of a src/loss_breakdown.py dump (target-argmax, a
fixed threshold) and reports macro F0.5 before and after, per country and per fold, on the dump's
held-out entities. With a model fitted on --countries US it measures transfer to an unseen country.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import er_io, pair_model as pm

ENC = pm.ENCODE
RESCUE_TOP = 3
MIN_PRECISION = 0.85
MIN_SUPPORT = 30  # inner accepted pairs needed before a threshold's precision counts
COUNTRIES = ("US", "India")
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
          "verbose": -1, "seed": C.SEED}
ROUNDS = 400


def enc_sets(cands, tid) -> dict:
    """{s1_row: set of encoded target id} from {s1_row: {source: set of target row}}."""
    return {r: {s * ENC + int(tid[s][t]) for s, ts in per.items() for t in ts} for r, per in cands.items()}


def rescue_pool(split, rows, s1, index, tid, text, drop_targets=frozenset()):
    """{s1_row: {source: set(target_row)}} of rescue-family candidates the main blocking never generated."""
    ctry = er_io.country_keys(s1)
    base = enc_sets(pm.gather(rows, s1, ctry, text, index, 200), tid)
    new = pm.gather(rows, s1, ctry, text, index, 200, fields=pm.RESCUE_FIELDS, base_keys=False)
    out = {}
    for r, per in new.items():
        keep = {s: {t for t in ts if (s * ENC + int(tid[s][t])) not in base[r]
                    and (s * ENC + int(tid[s][t])) not in drop_targets} for s, ts in per.items()}
        if any(keep.values()):
            out[r] = keep
    return out


def top_cut(cands, text, split):
    """Each entity's RESCUE_TOP candidates by name + address trigram overlap, with their target text."""
    if not cands:
        return {}, {}
    tt = pm.fetch_target_text(cands, split)
    keys, a_, n_ = pm.prerank_scores(cands, text, tt)
    ent = np.array([k[0] for k in keys])
    order = np.lexsort((-(np.asarray(a_) + np.asarray(n_)), ent))
    es = ent[order]
    first = np.r_[0, np.flatnonzero(np.diff(es)) + 1]
    rank = np.arange(len(es)) - np.repeat(first, np.diff(np.r_[first, len(es)]))
    out = {}
    for i in np.sort(order[rank < RESCUE_TOP]).tolist():
        r, s, t = keys[i]
        out.setdefault(r, {2: set(), 3: set()})[s].add(t)
    return out, tt


def features(cands, text, tt, split):
    """Pair features (with group features over the rescue list) and keys."""
    X, keys = pm.build_features(cands, text, tt, split)
    return pm.add_group_features(X, keys), keys


def assign(prob, encs, thr):
    """Boolean mask: the record's highest-scoring entity, at or above thr."""
    order = np.lexsort((-prob, encs))
    first = np.zeros(len(encs), bool)
    es = encs[order]
    first[order[np.r_[True, es[1:] != es[:-1]]]] = True
    return first & (prob >= thr)


def cmd_fit(a) -> None:
    """Train the rescue scorer on training entities disjoint from an excluded held-out set; freeze its threshold."""
    import lightgbm as lgb
    t0 = time.time()
    s1 = er_io.parse_source("train", 1)
    gt = er_io.parse_ground_truth()
    own = er_io.index_of(s1["ids"], gt["s1"])
    truth = {int(own[g]): {int(gt["match_src"][j]) * ENC + int(gt["match_id"][j])
                           for j in range(gt["offsets"][g], gt["offsets"][g + 1])}
             for g in range(len(gt["s1"]))}
    hold, hold_targets = set(), set()
    if a.exclude:
        with open(a.exclude, "rb") as f:  # our own held-out dump from src/loss_breakdown.py
            res = pickle.load(f)
        hold = set(res["holdout_set"])
        hold_targets = set().union(*(res["truth"][r] for r in hold))
    names = np.array(s1["countries"])[s1["country"]]
    pool = np.array(sorted(r for r in truth if r not in hold and names[r] in a.countries))
    rows = np.sort(np.random.default_rng(C.SEED + 777).choice(pool, a.entities, replace=False))
    tid = {s: er_io.parse_source("train", s)["ids"] for s in (2, 3)}
    index = pm.build_index("train", 300, fields=pm.TOKEN_FIELDS + pm.RESCUE_FIELDS)
    text = pm.read_s1_text("train", rows)
    cands, tt = top_cut(rescue_pool("train", rows, s1, index, tid, text, hold_targets), text, "train")
    X, keys = features(cands, text, tt, "train")
    y = np.array([(s * ENC + int(tid[s][t])) in truth.get(r, ()) for r, s, t in keys])
    ent = np.array([r for r, _, _ in keys])
    inner = (ent * 2654435761 % 4294967296) % 5 == 0
    q = lgb.train(PARAMS, lgb.Dataset(X[~inner], label=y[~inner]), ROUNDS).predict(X[inner])
    thr = next((round(float(t), 2) for t in np.arange(0.30, 0.995, 0.01)
                if (q >= t).sum() >= MIN_SUPPORT and y[inner][q >= t].mean() >= MIN_PRECISION), None)
    if thr is None:
        raise SystemExit("no threshold reaches the inner precision bar; the rescue stage is not used")
    sel = q >= thr
    support = {"inner_pairs": int(inner.sum()), "inner_positives": int(y[inner].sum()),
               "accepted": int(sel.sum()), "tp": int(y[inner][sel].sum()), "fp": int((~y[inner][sel]).sum()),
               "precision": round(float(y[inner][sel].mean()), 4)}
    del truth, cands, tt
    model = lgb.train(PARAMS, lgb.Dataset(X, label=y), ROUNDS)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / "rescue.txt"))
    meta = {"threshold": thr, "threshold_support": support, "min_support": MIN_SUPPORT,
            "countries": list(a.countries), "entities": int(a.entities), "pairs": int(len(y)), "positives": int(y.sum()),
            "top": RESCUE_TOP, "fields": list(pm.RESCUE_FIELDS), "excluded_holdout": len(hold),
            "features": pm.FEATURES + pm.GROUP_FEATURES, "seconds": round(time.time() - t0)}
    (out / "rescue.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "features"}, indent=1))


def read_lists(path: Path, col: int = 1) -> dict:
    """{source1_entity_id string: [ids]} from a two-column submission TSV."""
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            left, _, right = line.rstrip("\n").partition("\t")
            out[left] = [x for x in right.split(",") if x]
    return out


def cmd_apply(a) -> None:
    """Add accepted rescue matches to a finished submission; the rescued pairs join its candidate list."""
    import lightgbm as lgb
    t0 = time.time()
    meta = json.loads((Path(a.model) / "rescue.json").read_text(encoding="utf-8"))
    model = lgb.Booster(model_file=str(Path(a.model) / "rescue.txt"))
    base_dir = Path(a.base)
    matches = read_lists(base_dir / C.MATCHING_FILE)
    cand_lists = read_lists(base_dir / C.CANDIDATE_FILE)
    assigned = set()
    for ids in matches.values():
        for x in ids:
            src, num = x.split("-", 1)
            assigned.add(int(src[1:]) * ENC + int(num))
    s1 = er_io.parse_source("test", 1)
    names = np.array(s1["countries"])[s1["country"]]
    tid = {s: er_io.parse_source("test", s)["ids"] for s in (2, 3)}
    index = pm.build_index("test", 300, fields=pm.TOKEN_FIELDS + pm.RESCUE_FIELDS)
    target = np.flatnonzero(np.isin(names, a.countries))
    R, E, P = [], [], []
    for start in range(0, len(target), a.shard):
        rows = target[start:start + a.shard]
        text = pm.read_s1_text("test", rows)
        cands, tt = top_cut(rescue_pool("test", rows, s1, index, tid, text, assigned), text, "test")
        if not cands:
            continue
        X, keys = features(cands, text, tt, "test")
        R.append(np.array([r for r, _, _ in keys], dtype=np.int64))
        E.append(np.array([s * ENC + int(tid[s][t]) for _, s, t in keys], dtype=np.int64))
        P.append(model.predict(X))
        print(f"  {start + len(rows):,}/{len(target):,} entities | {sum(len(p) for p in P):,} rescue pairs "
              f"| {time.time() - t0:.0f}s", flush=True)
    r, e, p = np.concatenate(R), np.concatenate(E), np.concatenate(P)
    acc = assign(p, e, meta["threshold"])
    ids = s1["ids"]
    decode = lambda x: f"S{x // ENC}-{x % ENC}"
    for rr, ee in zip(r.tolist(), e.tolist()):
        cand_lists.setdefault(f"S1-{ids[rr]}", []).append(decode(ee))
    for rr, ee in zip(r[acc].tolist(), e[acc].tolist()):
        matches.setdefault(f"S1-{ids[rr]}", []).append(decode(ee))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    order = [f"S1-{x}" for x in ids.tolist()]
    for name, lists, col in ((C.MATCHING_FILE, matches, C.GT_MATCH_COL), (C.CANDIDATE_FILE, cand_lists, C.CANDIDATE_COL)):
        with open(out / name, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"{C.GT_ID_COL}\t{col}\n")
            for k in order:
                f.write(f"{k}\t{','.join(lists.get(k, ()))}\n")
    summary = {"apply_countries": list(a.countries), "model_countries": meta.get("countries"), "rescue_pairs": int(len(p)), "per_rescued_country_entity": len(p) / max(len(target), 1),
               "accepted": int(acc.sum()), "threshold": meta["threshold"], "seconds": round(time.time() - t0)}
    (out / "rescue_apply.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def cmd_evaluate(a) -> None:
    """Rescue added to a held-out dump's predictions; macro F0.5 before and after by country and fold."""
    import lightgbm as lgb
    from src.ce_blend import load_heldout
    from src.decide import apply_argmax
    meta = json.loads((Path(a.model) / "rescue.json").read_text(encoding="utf-8"))
    model = lgb.Booster(model_file=str(Path(a.model) / "rescue.txt"))
    with open(a.dump, "rb") as f:  # our own dump from src/loss_breakdown.py
        truth_all = pickle.load(f)["truth"]
    h = load_heldout(a.dump)
    keep = apply_argmax(h["s1_row"], h["enc_id"], h["prob"]) & (h["prob"] >= a.threshold)
    ents = h["ents"].tolist(); pos = {r: k for k, r in enumerate(ents)}
    P = [set() for _ in ents]
    for r, e in zip(h["s1_row"][keep].tolist(), h["enc_id"][keep].tolist()):
        P[pos[r]].add(e)
    T = [set(truth_all[r]) for r in ents]
    del truth_all
    assigned = set().union(*P)
    s1 = er_io.parse_source("train", 1)
    tid = {s: er_io.parse_source("train", s)["ids"] for s in (2, 3)}
    index = pm.build_index("train", 300, fields=pm.TOKEN_FIELDS + pm.RESCUE_FIELDS)
    rows = np.array(ents)
    text = pm.read_s1_text("train", rows)
    cands, tt = top_cut(rescue_pool("train", rows, s1, index, tid, text, assigned), text, "train")
    X, keys = features(cands, text, tt, "train")
    p = model.predict(X)
    e = np.array([s * ENC + int(tid[s][t]) for _, s, t in keys]); r = np.array([k[0] for k in keys])
    acc = assign(p, e, meta["threshold"])
    Q = [set(x) for x in P]
    for rr, ee in zip(r[acc].tolist(), e[acc].tolist()):
        Q[pos[rr]].add(ee)
    f1 = lambda a_, t: 1.0 if not a_ and not t else (1.25 * len(a_ & t) / (len(a_) + 0.25 * len(t)) if a_ & t else 0.0)
    b = np.array([f1(P[k], T[k]) for k in range(len(ents))]); q = np.array([f1(Q[k], T[k]) for k in range(len(ents))])
    c = h["country"].astype(str)
    fold = np.random.default_rng(C.SEED + 11).permutation(len(ents)) % 4
    out = {"threshold": a.threshold, "rescue_threshold": meta["threshold"], "threshold_support": meta.get("threshold_support"),
           "model_countries": meta.get("countries"),
           "pairs_per_entity": len(p) / len(ents), "accepted": int(acc.sum()),
           "added_true": int(sum(len(Q[k] - P[k] & T[k]) for k in range(len(ents)))),
           "added_false": int(sum(len((Q[k] - P[k]) - T[k]) for k in range(len(ents))))}
    for cc in sorted(set(c)):
        m = c == cc
        out[cc] = {"before": round(float(b[m].mean()), 5), "after": round(float(q[m].mean()), 5),
                   "delta": round(float(q[m].mean() - b[m].mean()), 5),
                   "fold_deltas": [round(float(q[m & (fold == k)].mean() - b[m & (fold == k)].mean()), 5) for k in range(4)]}
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--out", required=True)
    f.add_argument("--exclude", default=None, help="held-out dump whose entities and targets stay out of training")
    f.add_argument("--entities", type=int, default=40000)
    f.add_argument("--countries", nargs="+", default=list(COUNTRIES), help="training countries of the scorer")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--model", required=True)
    ev.add_argument("--dump", required=True, help="src/loss_breakdown.py --dump file")
    ev.add_argument("--threshold", type=float, default=0.80)
    ev.add_argument("--out", default=None)
    ap_ = sub.add_parser("apply")
    ap_.add_argument("--model", required=True)
    ap_.add_argument("--base", required=True, help="submission folder with matching_results.tsv and candidate_pairs.tsv")
    ap_.add_argument("--out", required=True)
    ap_.add_argument("--shard", type=int, default=50000)
    ap_.add_argument("--countries", nargs="+", default=list(COUNTRIES), help="countries whose entities are rescued")
    a = ap.parse_args()
    {"fit": cmd_fit, "apply": cmd_apply, "evaluate": cmd_evaluate}[a.cmd](a)
