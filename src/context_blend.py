"""Context-aware rescoring of scored pairs: a regularized logistic model on model scores plus observable context.

features() builds, for pairs of one split, a small set of context columns from record text and the scores
themselves (never from labels). The loco command learns the correction on the training country of an
unseen-country dump and scores the other country with the same assignment rule, reporting paired folds.
"""
import argparse, collections, json, pickle, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import config as C
from src import er_io
from src.ce_blend import ENCODE, entity_scores, load_heldout
from src.decide import apply_argmax
from src.er_norm import normalize_name

GATE = 0.02
CTX = ["t_addr_missing", "t_source3", "t_name_s1_count", "s1_name_s1_count", "name_jac", "num_conflict",
       "t_raw_lower", "owner_margin", "has_rival"]
DIGITS = re.compile(r"\d+")


def logit(p):
    """Clipped log-odds."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def s1_names(split):
    """Normalized Source 1 name and country per row, and the per-country count of each normalized name."""
    names, cty = [], []
    with open(C.source_file(split, 1), encoding="utf-8") as f:
        next(f)
        for line in f:
            q = line.rstrip("\n").split("\t")
            names.append(normalize_name(q[1]) if len(q) == 4 else ""); cty.append(q[-1])
    return names, cty, collections.Counter(zip(cty, names))


def owner_margin(enc_id, score):
    """Score minus the best score of any other entity on the same target, and whether such a rival exists."""
    order = np.lexsort((-score, enc_id)); e = enc_id[order]; s = score[order]
    start = np.concatenate(([True], e[1:] != e[:-1]))
    grp = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    best = s[first][grp]
    nxt = np.full(len(first), np.nan)
    has2 = np.zeros(len(first), dtype=bool)
    second = first + 1
    ok = second < len(e)
    ok[ok] = e[second[ok]] == e[first[ok]]
    nxt[ok] = s[second[ok]]; has2[ok] = True
    rival = np.where(start, nxt[grp], best)
    rival_exists = np.where(start, has2[grp], True)
    m = np.where(rival_exists, s - rival, 10.0)
    out_m = np.empty_like(m); out_r = np.empty(len(m), dtype=bool)
    out_m[order] = np.clip(m, -10, 10); out_r[order] = rival_exists
    return out_m, out_r


def pair_texts(split, s1_row, enc_id):
    """Raw (name, address) of the target record and of the Source 1 record for every pair."""
    n = len(s1_row)
    src = (enc_id // ENCODE).astype(int); num = enc_id % ENCODE
    t_name, t_addr = [""] * n, [""] * n
    for s in (2, 3):
        idx = np.flatnonzero(src == s)
        if not len(idx):
            continue
        rows = er_io.index_of(er_io.parse_source(split, s)["ids"], num[idx])
        ff = er_io.fetch_fields(split, s, np.unique(rows).tolist())
        for i, r in zip(idx.tolist(), rows.tolist()):
            t_name[i], t_addr[i] = ff.get(r, ("", ""))
    want = set(np.unique(s1_row).tolist()); s1t = {}
    with open(C.source_file(split, 1), encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if i in want:
                q = line.rstrip("\n").split("\t"); s1t[i] = (q[1], q[2]) if len(q) == 4 else ("", "")
    return t_name, t_addr, s1t


def features(split, s1_row, enc_id, score, names=None, texts=None):
    """Context matrix (columns CTX) for the given pairs; score is the model log-odds used for owner margins."""
    names, cty, count = names or s1_names(split)
    n = len(s1_row); src = (enc_id // ENCODE).astype(int)
    t_name, t_addr, s1t = texts or pair_texts(split, s1_row, enc_id)
    X = np.zeros((n, len(CTX)), dtype=np.float32)
    for i in range(n):
        r = int(s1_row[i]); c = cty[r]; nm, ad = t_name[i], t_addr[i]
        tn = normalize_name(nm); a, b = set(names[r].split()), set(tn.split())
        sn, sa = s1t.get(r, ("", ""))
        d1, d2 = set(DIGITS.findall(sn + " " + sa)), set(DIGITS.findall(nm + " " + ad))
        letters = [ch for ch in nm if ch.isalpha()]
        X[i, 0] = ad.strip() in ("", "<NULL>")
        X[i, 1] = src[i] == 3
        X[i, 2] = np.log1p(count[(c, tn)]) if tn else 0.0
        X[i, 3] = np.log1p(count[(c, names[r])]) if names[r] else 0.0
        X[i, 4] = len(a & b) / len(a | b) if a | b else 0.0
        X[i, 5] = bool(d1 and d2 and not d1 & d2)
        X[i, 6] = bool(letters) and nm == nm.lower()
    X[:, 7], X[:, 8] = owner_margin(np.asarray(enc_id), np.asarray(score, dtype=np.float64))
    return X


HOP_COLS = ["hop_n_anchor", "hop_is_anchor", "hop_name_sim", "hop_addr_sim", "hop_both_sim", "hop_anchor_logit",
            "hop_support"]
ANCHOR = 0.95


def _grams(s):
    """Character trigram set of a normalized string."""
    s = f"  {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _sim(a, b):
    """Jaccard of two trigram sets."""
    return len(a & b) / len(a | b) if a and b else 0.0


def hop_features(s1_row, enc_id, prob, texts):
    """Agreement of each candidate with the entity's other confident records (argmax-kept, prob >= ANCHOR).

    Columns HOP_COLS: anchor count, whether the pair is itself an anchor, best name / address / joint
    trigram similarity to another anchor, that anchor's log-odds, and how many anchors share its name closely.
    """
    t_name, t_addr = texts[0], texts[1]
    keep = apply_argmax(s1_row, enc_id, prob)
    anchor = keep & (prob >= ANCHOR)
    gn = [_grams(normalize_name(x)) for x in t_name]; ga = [_grams(normalize_name(x)) for x in t_addr]
    X = np.zeros((len(s1_row), len(HOP_COLS)), dtype=np.float32)
    order = np.argsort(s1_row, kind="stable"); r = s1_row[order]
    bounds = np.flatnonzero(np.concatenate(([True], r[1:] != r[:-1], [True])))
    lg = logit(prob)
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        idx = order[b0:b1]; anc = [i for i in idx if anchor[i]]
        for i in idx:
            others = [j for j in anc if j != i]
            best = (0.0, 0.0, 0.0, 0.0); sup = 0
            for j in others:
                ns, as_ = _sim(gn[i], gn[j]), _sim(ga[i], ga[j]); both = (ns + as_) / 2
                sup += ns >= 0.6
                if both > best[2]:
                    best = (ns, as_, both, lg[j])
            X[i] = (len(others), anchor[i], best[0], best[1], best[2], best[3], sup)
    return X


def bundle_features(bundle, split, s1_row, enc_id, score):
    """Context columns (bundle "context") or pair difference columns ("name", "address") for the given pairs."""
    names = s1_names(split); texts = pair_texts(split, s1_row, enc_id)
    if bundle == "context":
        return features(split, s1_row, enc_id, score, names=names, texts=texts)
    if bundle == "hop":
        return hop_features(np.asarray(s1_row), np.asarray(enc_id), 1 / (1 + np.exp(-np.asarray(score))), texts)
    from src.diff_features import diff_features, rarity_tables
    return diff_features(split, s1_row, texts, names[1], rarity_tables(split), bundle)


def design(scores, ctx):
    """Scores, their products, context, and score-by-context interactions."""
    cols = list(scores)
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            cols.append(scores[i] * scores[j])
    if len(scores) > 1:
        cols.append(np.abs(scores[0] - scores[1]))
    cols += [ctx[:, k] for k in range(ctx.shape[1])]
    for s in scores:
        cols += [s * ctx[:, k] for k in range(ctx.shape[1])]
    return np.column_stack(cols)


def fit(X, y, C_=1.0):
    """Standardized L2 logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C_, max_iter=3000)).fit(X, y)


def cmd_loco(a):
    """Learn on the training country's pairs only; report both countries against raw LightGBM at the same rule."""
    h = load_heldout(a.dump)
    with open(a.dump, "rb") as f:  # our own dump from src/loss_breakdown.py
        hold = sorted(pickle.load(f)["holdout_set"])
    perm = np.random.default_rng(C.SEED + 11).permutation(len(hold))
    fold_of = dict(zip(hold, (perm % 4).tolist()))
    ent_fold = np.array([fold_of[int(r)] for r in h["ents"].tolist()])
    c_ent = h["country"].astype(str); c_pair = c_ent[h["ent_idx"]]; pf = ent_fold[h["ent_idx"]]
    lg = logit(h["prob"]); m = h["prob"] >= GATE
    ctx = bundle_features(a.bundle, "train", h["s1_row"][m], h["enc_id"][m], lg[m])
    X = design([lg[m]], ctx); y = h["label"][m]
    tr = c_pair[m] == a.train_country
    p_new = h["prob"].astype(np.float64).copy()
    oof = np.empty(int(m.sum()))
    for k in range(4):  # training country: cross-fitted by entity fold
        te = tr & (pf[m] == k)
        oof[te] = fit(X[tr & ~te], y[tr & ~te]).predict_proba(X[te])[:, 1]
    model = fit(X[tr], y[tr]); oof[~tr] = model.predict_proba(X[~tr])[:, 1]
    p_new[m] = oof

    def f05(p, t):
        keep = apply_argmax(h["s1_row"], h["enc_id"], p)
        pred = (p >= t) & keep
        return entity_scores(h["ent_idx"], h["label"], pred, h["k_true"]), pred
    trm = c_ent == a.train_country
    grid = [0.70, 0.75, 0.80, 0.85, 0.90]
    t_sel = max(grid, key=lambda t: f05(p_new, t)[0][trm].mean())  # chosen on the training country only
    fb, pb = f05(h["prob"].astype(np.float64), a.threshold)
    out = {"dump": a.dump, "bundle": a.bundle, "train_country": a.train_country, "gate": GATE, "pairs_rescored": int(m.sum()),
           "baseline_threshold": a.threshold, "selected_threshold": t_sel}
    for name, t in (("selected", t_sel), ("fixed", a.threshold)):
        fn, pn = f05(p_new, t); res = {}
        for cc in sorted(set(c_ent)):
            e = c_ent == cc; pm_ = c_pair == cc
            res[cc] = {"base": round(float(fb[e].mean()), 5), "new": round(float(fn[e].mean()), 5),
                       "delta": round(float(fn[e].mean() - fb[e].mean()), 5),
                       "fold_deltas": [round(float(fn[e & (ent_fold == k)].mean() - fb[e & (ent_fold == k)].mean()), 5) for k in range(4)],
                       "tp": int((pn & h["label"] & pm_).sum() - (pb & h["label"] & pm_).sum()),
                       "fp": int((pn & ~h["label"] & pm_).sum() - (pb & ~h["label"] & pm_).sum()),
                       "singleton_regressions": int(((fb == 1) & (fn < 1) & e).sum())}
        out[name] = res
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lo = sub.add_parser("loco")
    lo.add_argument("--dump", required=True, help="src/loss_breakdown.py --dump file of an unseen-country run")
    lo.add_argument("--train-country", default="US")
    lo.add_argument("--threshold", type=float, default=0.80)
    lo.add_argument("--bundle", choices=["context", "name", "address", "hop"], default="context")
    lo.add_argument("--out", default=None)
    a = ap.parse_args()
    {"loco": cmd_loco}[a.cmd](a)
