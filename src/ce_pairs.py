"""Export the pair lists a text cross-encoder trains on and rescores.

Three inputs, each optional so the parts can be exported as they appear:

  --fit-pairs     every pair the LightGBM matcher was fitted on, with labels
                  (loss_breakdown.py --dump-pairs). Only entities outside the
                  evaluated sample (in_sample False) become training pairs, and
                  every pair whose target record is also a target of an
                  exported held-out pair is dropped: on test no target record
                  has been seen in training, so a model that learned a record's
                  text as "match" or "non-match" would score held-out pairs in
                  a way test can not reproduce. The held-out pairs must be
                  exported first (same call, or already in --out); a later
                  held-out export that shares an entity or a target with
                  train_pairs.npz is refused.
  --train-countries  keep only training entities of these countries (comma
                  separated country labels), for a model that never sees the
                  others; ce_blend.py fit --loco-ce scores it on them.
  --heldout-dump  the held-out predictions (loss_breakdown.py --dump). Every
                  pair with an out-of-fold probability at or above --floor is
                  exported with its label.
  --test-scores   a test score dump (predict_matches.py --dump-scores). Every
                  pair at or above --floor is exported with the shard and the
                  position it came from, so rescored probabilities can be
                  written back into an identical dump. The held-out dump holds
                  raw booster output, so the test dump must too: it has to come
                  from predict_matches.py --no-calibrate with the same features
                  as the held-out run. A shard whose probs take only a few
                  distinct values (isotonic steps) is refused.
  --test-metrics  results/<run>/metrics.json of the run that wrote
                  --test-scores; required with it. Refused unless the run is
                  uncalibrated, covers the whole test split, and scored exactly
                  as many pairs as the dump holds.
  --test-model    model folder of that run (default outputs/models/<run>). It
                  must hold MODELS boosters over the current feature count
                  (pair_model FEATURES + GROUP_FEATURES), a preranker.txt, and
                  a calibration.pkl without an isotonic map, which is what the
                  held-out run used. A run made with predict_matches.py
                  --load-model saves no model folder, so pass the loaded one.
  --allow-calibrated  accept a calibrated test run (isotonic probs): only the
                  calibration and step-value checks are skipped, and the
                  manifest's test_run records calibrated true and the sha256
                  of the run's calibration.pkl, which ce_blend.py apply checks
                  against a model fitted with --calibrator.

Written to --out:
  train_pairs.npz    s1_id int64, enc_id int64, label int8
  heldout_pairs.npz  s1_row int32, s1_id int64, enc_id int64, prob float32, label int8
                     (sorted by s1_row, then enc_id)
  test_pairs.npz     s1_row int32, s1_id int64, enc_id int64, prob float32,
                     shard int16, pos int32 (shard indexes shards.json)
  shards.json        test shard file names in sorted order
  pseudo_pairs.npz   s1_id int64, enc_id int64, label int8 (export-pseudo; test-split ids)
  manifest.json      floor, sampling settings, training countries, source paths,
                     counts, sha256 per npz, heldout_digest (pairs_digest of the
                     held-out pairs)

enc_id is source * 2_000_000_000 + the numeric target id, as in pair_model.

export-pseudo writes pseudo_pairs.npz (s1_id int64, enc_id int64, label int8)
from a pseudo_labels.py file: self-labelled pairs of test records, so s1_id is
a TEST Source 1 id and src/ce_rerank.py reads their text from the test split.
All positives are kept and negatives are sampled as for train_pairs.npz.

raw-model copies a saved model folder with its isotonic map removed.
predict_matches.py --load-model applies the saved map even with
--no-calibrate, so a raw dump from a calibrated run's boosters needs such a
copy (the boosters were fitted without the calibration split's entities).

Run:
    python src/predict_matches.py --split test --no-calibrate --entities 60000 \
        --extra-train 180000 --models 5 --preranker 30000 --top-k 40 --workers 9 \
        --dump-scores outputs/scores/test_v8_raw --run-name sub08_raw
    python src/ce_pairs.py export --fit-pairs outputs/cache/pairs_addridf.npz \
        --heldout-dump outputs/cache/cv4m5_addridf.pkl \
        --test-scores outputs/scores/test_v8_raw --test-metrics results/sub08_raw/metrics.json \
        --out outputs/ce/export
    # training pairs of one country only, for a leave-one-country-out model:
    python src/ce_pairs.py export --fit-pairs outputs/cache/pairs_addridf.npz \
        --heldout-dump outputs/cache/cv4m5_addridf.pkl --train-countries US \
        --out outputs/ce/export_us
    # self-labelled pairs of the unseen-country test records:
    python src/ce_pairs.py export-pseudo --pseudo outputs/cache/pseudo_v8.npz \
        --out outputs/ce/france_export
    # raw dump from an already fitted calibrated run, without refitting:
    python src/ce_pairs.py raw-model --src outputs/models/sub08_addridf \
        --dst outputs/models/sub08_addridf_raw
    python src/predict_matches.py --split test --no-calibrate --load-model \
        outputs/models/sub08_addridf_raw --top-k 40 --workers 9 \
        --dump-scores outputs/scores/test_v8_raw --run-name sub08_raw
    python src/ce_pairs.py export --test-scores outputs/scores/test_v8_raw \
        --test-metrics results/sub08_raw/metrics.json \
        --test-model outputs/models/sub08_addridf_raw --out outputs/ce/export
    # calibrated test run, blended on the calibrated scale (ce_blend.py fit --calibrator):
    python src/ce_pairs.py export --test-scores outputs/scores/test_v8 \
        --test-metrics results/sub08_addridf/metrics.json --allow-calibrated --out outputs/ce/export
"""
import argparse
import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import er_io

MIN_DISTINCT = 0.01
MODELS = 5  # boosters averaged per prediction in the held-out run (--models 5)


def distinct_frac(p) -> float:
    """Distinct probs per pair (1.0 when empty).

    Isotonic calibration maps scores onto a few hundred steps (279 distinct
    values over 1.16M pairs in one shard, 0.0002), while averaged booster
    output is nearly all distinct (0.9). Below MIN_DISTINCT the probs are
    step-valued, and a blend fitted on raw probs must not be applied to them.
    """
    return len(np.unique(p)) / len(p) if len(p) else 1.0


class _HeldoutUnpickler(pickle._Unpickler):
    """Pure-Python unpickler for the held-out dump that leaves raw_cands empty.

    raw_cands holds every raw candidate id of every held-out entity: about
    three quarters of the file and most of the memory a plain load needs. Its
    sets are left empty, and leaf values (numbers, tuples, bytes) are not kept
    in the memo, so only the parts that are used stay resident.
    """

    dispatch = dict(pickle._Unpickler.dispatch)
    _LEAVES = (bytes, tuple, int, float, np.generic)

    def __init__(self, f):
        """Open the stream and reset the memo counter and top-level key."""
        super().__init__(f)
        self._n_memo = 0
        self._top_key = None

    def load_memoize(self):
        """Advance the memo index; keep the object unless it is a leaf value."""
        obj = self.stack[-1]
        if not isinstance(obj, self._LEAVES):
            self.memo[self._n_memo] = obj
        self._n_memo += 1

    def load_short_binunicode(self):
        """Push a string and remember it when it is a key of the top-level dict."""
        pickle._Unpickler.load_short_binunicode(self)
        if len(self.metastack) == 1:
            self._top_key = self.stack[-1]

    def load_additems(self):
        """Fill a set, except the candidate sets under the raw_cands key."""
        if self._top_key == "raw_cands":
            self.pop_mark()
            return
        pickle._Unpickler.load_additems(self)

    dispatch[pickle.MEMOIZE[0]] = load_memoize
    dispatch[pickle.SHORT_BINUNICODE[0]] = load_short_binunicode
    dispatch[pickle.ADDITEMS[0]] = load_additems


def load_heldout(path) -> dict:
    """Held-out dump without raw_cands: val_by_ent, truth, holdout_set, country, countries.

    The file is the one loss_breakdown.py --dump writes in this repo.
    Falls back to a plain pickle.load if the lean reader meets a layout it
    does not expect (a memo reference to a skipped leaf).
    """
    with open(path, "rb") as f:
        try:
            res = _HeldoutUnpickler(f).load()
        except KeyError:
            print("lean read failed; loading the whole dump", flush=True)
            f.seek(0)
            res = pickle.load(f)
    res.pop("raw_cands", None)
    return res


def pairs_digest(s1_row, enc_id) -> str:
    """Hex SHA-256 over the int64 bytes of a held-out pair list's S1 rows then target ids.

    src/ce_rerank.py records it for the held-out pairs a model's training
    pairs were checked against; src/ce_blend.py compares it with the pairs it
    blends.
    """
    h = hashlib.sha256(np.ascontiguousarray(s1_row, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(enc_id, dtype=np.int64).tobytes())
    return h.hexdigest()


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file, read in 1 MB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def booster_width(path: Path) -> int:
    """Feature count of a LightGBM text model, from its max_feature_idx header line."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("max_feature_idx="):
                return int(line.split("=", 1)[1]) + 1
            if line.startswith("Tree="):
                break
    raise SystemExit(f"{path}: no max_feature_idx line")


def check_test_run(metrics_path, model_dir, allow_calibrated=False) -> dict:
    """Refuse a test run whose model differs in kind from the held-out run's.

    The run must be uncalibrated and cover the whole test split. Its model
    folder must hold MODELS boosters, each over len(FEATURES) +
    len(GROUP_FEATURES) columns, a preranker.txt, and a calibration.pkl whose
    isotonic map is None (predict_matches.py applies a saved map even with
    --no-calibrate when it loads a model). With allow_calibrated both
    calibration checks are skipped and the record carries the sha256 of
    calibration.pkl. Returns the manifest record.
    """
    from src import pair_model as pm
    run = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    if run.get("calibrated") and not allow_calibrated:
        raise SystemExit(f"{metrics_path}: run {run.get('run')} is calibrated; the held-out dump is raw")
    if run.get("split") != "test" or run.get("limit") is not None:
        raise SystemExit(f"{metrics_path}: not a full test run (split {run.get('split')}, "
                         f"limit {run.get('limit')})")
    md = Path(model_dir) if model_dir else C.MODEL_DIR / str(run.get("run"))
    if not md.is_dir():
        raise SystemExit(f"no model folder {md}; pass --test-model (a --load-model run saves none)")
    want = len(pm.FEATURES) + len(pm.GROUP_FEATURES)
    widths = {b.name: booster_width(b) for b in sorted(md.glob("booster_*.txt"))}
    if len(widths) != MODELS or set(widths.values()) != {want}:
        raise SystemExit(f"{md}: boosters {widths}; the held-out run averages {MODELS} boosters "
                         f"of {want} features")
    if not (md / "preranker.txt").exists():
        raise SystemExit(f"{md}: no preranker.txt; the held-out run uses the learned pre-ranker")
    with open(md / "calibration.pkl", "rb") as f:
        cal = pickle.load(f)
    if cal.get("iso") is not None and not allow_calibrated:
        raise SystemExit(f"{md}/calibration.pkl holds an isotonic map, which predict_matches.py "
                         "applies before dumping; make a copy with ce_pairs.py raw-model")
    return {"metrics": str(Path(metrics_path).resolve()), "run": run.get("run"),
            "calibrated": run.get("calibrated"), "candidate_pairs": run.get("candidate_pairs"),
            "rows": run.get("rows"), "model_dir": str(md.resolve()), "n_features": want,
            "boosters": len(widths), "iso_removed_from": cal.get("iso_removed_from"),
            "calibration_sha256": sha256(md / "calibration.pkl") if run.get("calibrated") else None}


def raw_model(a) -> None:
    """Copy a model folder with its isotonic map set to None, marking where it came from."""
    src, dst = Path(a.src), Path(a.dst)
    if dst.exists():
        raise SystemExit(f"{dst} already exists")
    with open(src / "calibration.pkl", "rb") as f:
        cal = pickle.load(f)
    dst.mkdir(parents=True)
    for f in sorted(src.glob("booster*.txt")) + sorted(src.glob("preranker.txt")):
        shutil.copy2(f, dst / f.name)
    with open(dst / "calibration.pkl", "wb") as f:
        pickle.dump(dict(cal, iso=None, iso_removed_from=str(src.resolve())), f)
    print(f"wrote {dst} ({len(list(dst.glob('booster*.txt')))} boosters, isotonic map removed)")


def export_train(path, neg_per_pos: int, seed: int, held_enc, countries=None):
    """Training pairs from entities outside the evaluated sample.

    Pairs whose target is in held_enc (the targets of the held-out pairs) are
    dropped first, and with countries set only entities of those countries
    are kept. All remaining positives are kept. Negatives are sampled
    uniformly within each entity, at most neg_per_pos per positive; an entity
    with no true match counts as one positive, so unmatched entities still
    contribute negatives. Returns the arrays, sorted by s1_id then enc_id.
    """
    f = np.load(path)
    extra = ~f["in_sample"]
    keep = extra.copy()
    n_country = 0
    if countries:
        t = er_io.parse_source("train", 1)
        unknown = set(countries) - set(t["countries"])
        if unknown:
            raise SystemExit(f"--train-countries {sorted(unknown)} not in train {t['countries']}")
        codes = [t["countries"].index(c) for c in countries]
        keep &= np.isin(t["country"][f["s1_row"]], codes)
        n_country = int(extra.sum() - keep.sum())
    enc_all = f["enc_id"]
    shared = keep & np.isin(enc_all, held_enc)
    n_drop, n_drop_pos = int(shared.sum()), int(f["label"][shared].sum())
    keep &= ~shared
    del shared
    arrays, n_ent = sample_negatives(f["s1_id"][keep], enc_all[keep], f["label"][keep], neg_per_pos, seed)
    lab = arrays["label"]
    return arrays, \
        {"pairs": int(len(lab)), "positives": int(lab.sum()), "entities": n_ent,
         "dump_pairs": int(len(keep)), "dump_pairs_extra_train": int(extra.sum()),
         "dropped_other_country": n_country, "dropped_heldout_target_pairs": n_drop,
         "dropped_heldout_target_positives": n_drop_pos,
         "heldout_targets": int(len(np.unique(held_enc)))}


def sample_negatives(s1_id, enc, label, neg_per_pos: int, seed: int):
    """All positives plus at most neg_per_pos uniformly drawn negatives per positive, per entity.

    An entity with no positive counts as one, so it still contributes
    negatives. Returns ({s1_id, enc_id, label int8} sorted by s1_id then
    enc_id, number of entities in the input).
    """
    lab = np.asarray(label).astype(np.int8)
    u = np.random.default_rng(seed).random(len(lab))
    order = np.lexsort((u, lab == 0, s1_id))  # per entity: positives, then negatives in random order
    s1_id, enc, lab = s1_id[order], enc[order], lab[order]
    new = np.r_[True, s1_id[1:] != s1_id[:-1]]
    ent = np.cumsum(new) - 1
    rank = np.arange(len(lab)) - np.flatnonzero(new)[ent]
    n_pos = np.bincount(ent, weights=lab).astype(np.int64)
    cap = neg_per_pos * np.maximum(n_pos, 1)
    sel = (lab == 1) | (rank - n_pos[ent] < cap[ent])
    s1_id, enc, lab = s1_id[sel], enc[sel], lab[sel]
    order = np.lexsort((enc, s1_id))
    return {"s1_id": s1_id[order], "enc_id": enc[order], "label": lab[order]}, int(new.sum())


def export_pseudo(a) -> None:
    """Write pseudo_pairs.npz from a pseudo_labels.py file and add it to the manifest.

    The file's s1_row indexes the test Source 1 file, so s1_id is read from
    there; negatives are sampled as for train_pairs.npz. The manifest in --out
    is updated in place, and sha256 is refreshed for every pair file present.
    """
    with np.load(a.pseudo) as b:
        s1_row, enc, label = b["s1_row"], b["enc_id"], b["label"]
        meta = json.loads(str(b["meta"]))
    ids = er_io.parse_source("test", 1)["ids"]
    if len(s1_row) and (s1_row.min() < 0 or s1_row.max() >= len(ids)):
        raise SystemExit(f"{a.pseudo}: s1_row outside the test Source 1 file")
    arrays, n_ent = sample_negatives(ids[s1_row].astype(np.int64), enc.astype(np.int64), label,
                                     a.neg_per_pos, a.seed)
    s1_id, e = arrays["s1_id"], arrays["enc_id"]
    if len(e) > 1 and ((s1_id[1:] == s1_id[:-1]) & (e[1:] == e[:-1])).any():
        raise SystemExit(f"{a.pseudo} has a repeated (s1_row, enc_id) pair")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "pseudo_pairs.npz", **arrays)
    lab = arrays["label"]
    counts = {"pairs": int(len(lab)), "positives": int(lab.sum()), "entities": n_ent,
              "entities_with_positive": int(len(np.unique(s1_id[lab == 1]))),
              "source_pairs": int(len(label)), "source_positives": int(np.sum(label == 1))}
    mpath = out / "manifest.json"
    man = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    man.setdefault("sources", {})["pseudo"] = str(Path(a.pseudo).resolve())
    man.setdefault("counts", {})["pseudo_pairs"] = counts
    man["pseudo"] = {"split": "test", "neg_per_pos": a.neg_per_pos, "seed": a.seed,
                     "countries": meta.get("countries"), "labels": meta}
    shas = man.setdefault("sha256", {})
    for name in ("train_pairs.npz", "heldout_pairs.npz", "test_pairs.npz", "pseudo_pairs.npz"):
        if (out / name).exists():
            shas[name] = sha256(out / name)
    mpath.write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"pseudo_pairs: {counts}\nwrote {out}", flush=True)


def export_heldout(path, floor: float):
    """Held-out pairs with out-of-fold probability >= floor, labelled from truth.

    Also returns the entity ids of the whole held-out set.
    """
    res = load_heldout(path)
    truth, hold = res["truth"], res["holdout_set"]
    rows, encs, probs, labels = [], [], [], []
    n_dump = 0
    for r, lst in res["val_by_ent"].items():
        if r not in hold:
            continue
        n_dump += len(lst)
        t = truth[r]
        for p, e in lst:
            if p >= floor:
                rows.append(r)
                encs.append(int(e))
                probs.append(float(p))
                labels.append(int(e) in t)
    ids = er_io.parse_source("train", 1)["ids"]
    hold_ids = np.unique(ids[np.fromiter(hold, dtype=np.int64, count=len(hold))])
    del res, truth, hold
    s1_row = np.array(rows, dtype=np.int32)
    enc = np.array(encs, dtype=np.int64)
    order = np.lexsort((enc, s1_row))
    s1_row, enc = s1_row[order], enc[order]
    if len(s1_row) > 1 and ((s1_row[1:] == s1_row[:-1]) & (enc[1:] == enc[:-1])).any():
        raise SystemExit("held-out dump has a repeated (s1_row, enc_id) pair")
    lab = np.array(labels, dtype=np.int8)[order]
    out = {"s1_row": s1_row, "s1_id": ids[s1_row],
           "enc_id": enc, "prob": np.array(probs, dtype=np.float32)[order], "label": lab}
    return out, {"pairs": int(len(lab)), "positives": int(lab.sum()),
                 "entities": int(len(np.unique(s1_row))), "heldout_entities": len(hold_ids),
                 "dump_pairs": n_dump}, hold_ids


def export_test(score_dir, floor: float, allow_calibrated=False):
    """Test pairs with probability >= floor, streamed one shard at a time.

    Refuses step-valued (isotonic-calibrated) shards unless allow_calibrated.
    Returns the arrays, the shard file names, and counts.
    """
    files = sorted(Path(score_dir).glob("shard_*.npz"))
    if not files:
        raise SystemExit(f"no shards in {score_dir}")
    if len(files) > np.iinfo(np.int16).max:
        raise SystemExit("too many shards for an int16 shard index")
    ids = er_io.parse_source("test", 1)["ids"]
    parts = {k: [] for k in ("s1_row", "enc_id", "prob", "shard", "pos")}
    n_dump = n_ent = 0
    min_distinct = 1.0
    for k, f in enumerate(files):
        b = np.load(f)
        s1_row, enc, p = b["s1_row"], b["enc_id"], b["prob"]
        if not len(s1_row) == len(enc) == len(p):
            raise SystemExit(f"{f.name}: array lengths differ")
        if len(s1_row) and (s1_row.min() < 0 or s1_row.max() >= len(ids)):
            raise SystemExit(f"{f.name}: s1_row outside the test Source 1 file")
        frac = distinct_frac(p)
        if frac < MIN_DISTINCT and not allow_calibrated:
            raise SystemExit(f"{f.name}: {frac:.4f} distinct probs per pair (isotonic steps); "
                             "the held-out dump is raw, so dump test with predict_matches.py "
                             "--no-calibrate")
        min_distinct = min(min_distinct, frac)
        n_dump += len(p)
        n_ent += len(np.unique(s1_row))
        pos = np.flatnonzero(p >= floor).astype(np.int32)
        parts["s1_row"].append(s1_row[pos].astype(np.int32))
        parts["enc_id"].append(enc[pos].astype(np.int64))
        parts["prob"].append(p[pos].astype(np.float32))
        parts["shard"].append(np.full(len(pos), k, dtype=np.int16))
        parts["pos"].append(pos)
        print(f"  {f.name}: {len(pos):,} of {len(p):,} pairs kept", flush=True)
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out["s1_id"] = ids[out["s1_row"]]
    return out, [f.name for f in files], {
        "pairs": int(len(out["prob"])), "entities": int(len(np.unique(out["s1_row"]))),
        "shards": len(files), "dump_pairs": n_dump, "dump_entities": n_ent,
        "test_s1_rows": int(len(ids)), "min_distinct_prob_frac": round(min_distinct, 4)}


def export(a) -> None:
    """Write whichever of the three pair files have inputs, then the manifest."""
    if not (a.fit_pairs or a.heldout_dump or a.test_scores):
        raise SystemExit("give at least one of --fit-pairs, --heldout-dump, --test-scores")
    if a.test_scores and not a.test_metrics:
        raise SystemExit("--test-scores needs --test-metrics (metrics.json of the run that wrote it)")
    run = check_test_run(a.test_metrics, a.test_model, a.allow_calibrated) if a.test_scores else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    man = (json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists()
           else {"floor": a.floor, "sources": {}, "counts": {}, "sha256": {}})
    if (a.heldout_dump or a.test_scores) and man.setdefault("floor", a.floor) != a.floor:
        raise SystemExit(f"{out} was exported with floor {man['floor']}, not {a.floor}")

    held_ids = held_enc = None
    if a.heldout_dump:
        arrays, counts, held_ids = export_heldout(a.heldout_dump, a.floor)
        held_enc = np.unique(arrays["enc_id"])
        if not a.fit_pairs and (out / "train_pairs.npz").exists():
            with np.load(out / "train_pairs.npz") as tr:
                if np.isin(tr["s1_id"], held_ids).any() or np.isin(tr["enc_id"], held_enc).any():
                    raise SystemExit(f"{out}/train_pairs.npz shares entities or target records with "
                                     "these held-out pairs; export --fit-pairs again with them")
        np.savez_compressed(out / "heldout_pairs.npz", **arrays)
        man["sources"]["heldout_dump"] = str(Path(a.heldout_dump).resolve())
        man["counts"]["heldout_pairs"] = counts
        man["heldout_digest"] = pairs_digest(arrays["s1_row"], arrays["enc_id"])
        print(f"heldout_pairs: {counts}", flush=True)
        del arrays
    if a.fit_pairs:
        if held_enc is None:
            if not (out / "heldout_pairs.npz").exists():
                raise SystemExit("--fit-pairs needs the held-out pairs (--heldout-dump, or "
                                 "heldout_pairs.npz in --out) to leave their target records out")
            with np.load(out / "heldout_pairs.npz") as hp:
                held_ids, held_enc = np.unique(hp["s1_id"]), np.unique(hp["enc_id"])
        countries = [c.strip() for c in a.train_countries.split(",")] if a.train_countries else None
        arrays, counts = export_train(a.fit_pairs, a.neg_per_pos, a.seed, held_enc, countries)
        if np.isin(np.unique(arrays["s1_id"]), held_ids).any():
            raise SystemExit("training pairs share entities with the held-out pairs")
        np.savez_compressed(out / "train_pairs.npz", **arrays)
        man["sources"]["fit_pairs"] = str(Path(a.fit_pairs).resolve())
        man["counts"]["train_pairs"] = counts
        man["neg_per_pos"], man["seed"] = a.neg_per_pos, a.seed
        man["train_countries"] = countries
        print(f"train_pairs: {counts}", flush=True)
        del arrays
    if a.test_scores:
        man.pop("test_run", None)
        arrays, shards, counts = export_test(a.test_scores, a.floor, a.allow_calibrated)
        if (counts["dump_pairs"], counts["test_s1_rows"]) != (run["candidate_pairs"], run["rows"]):
            raise SystemExit(f"{a.test_scores} holds {counts['dump_pairs']:,} pairs over "
                             f"{counts['test_s1_rows']:,} rows; run {run['run']} scored "
                             f"{run['candidate_pairs']} pairs over {run['rows']} rows")
        man["test_run"] = run
        np.savez_compressed(out / "test_pairs.npz", **arrays)
        (out / "shards.json").write_text(json.dumps(shards, indent=1), encoding="utf-8")
        man["sources"]["test_scores"] = str(Path(a.test_scores).resolve())
        man["counts"]["test_pairs"] = counts
        print(f"test_pairs: {counts}", flush=True)
        del arrays

    for name in ("train_pairs.npz", "heldout_pairs.npz", "test_pairs.npz"):
        if (out / name).exists():
            man["sha256"][name] = sha256(out / name)
    mpath.write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="write the train, held-out and test pair files")
    e.add_argument("--fit-pairs", default=None, help="loss_breakdown.py --dump-pairs output (.npz)")
    e.add_argument("--heldout-dump", default=None, help="loss_breakdown.py --dump output (.pkl)")
    e.add_argument("--test-scores", default=None,
                   help="predict_matches.py --dump-scores directory of an uncalibrated run")
    e.add_argument("--test-metrics", default=None,
                   help="metrics.json of the run that wrote --test-scores (required with it)")
    e.add_argument("--test-model", default=None,
                   help="model folder of that run (default outputs/models/<run>)")
    e.add_argument("--allow-calibrated", action="store_true",
                   help="accept a calibrated test run; its calibration.pkl sha256 goes in the manifest")
    e.add_argument("--floor", type=float, default=0.02,
                   help="held-out and test pairs below this LightGBM probability are left out")
    e.add_argument("--out", required=True)
    e.add_argument("--neg-per-pos", type=int, default=4)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--train-countries", default=None,
                   help="comma separated country labels; training pairs of other countries are left out")
    r = sub.add_parser("raw-model", help="copy a model folder with its isotonic map removed")
    r.add_argument("--src", required=True)
    r.add_argument("--dst", required=True)
    q = sub.add_parser("export-pseudo", help="write pseudo_pairs.npz from a pseudo_labels.py file")
    q.add_argument("--pseudo", required=True, help="pseudo_labels.py output (.npz)")
    q.add_argument("--out", required=True)
    q.add_argument("--neg-per-pos", type=int, default=4)
    q.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    {"raw-model": raw_model, "export-pseudo": export_pseudo}.get(args.cmd, export)(args)
