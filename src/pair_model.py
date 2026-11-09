"""Pairwise matcher: candidates -> features -> LightGBM -> admission -> assignment.

The blocking ceiling says a perfect matcher over address-plus-name candidates
would score about 0.89. This turns that ceiling into an actual score.

Four stages:
  1. Candidates from the cached token postings (name tokens, address tokens,
     address number-word keys, whole-name keys, house numbers), within country.
  2. Features per candidate pair, from the raw name and address text.
  3. A LightGBM classifier trained on labelled pairs, with isotonic calibration
     fitted on a separate slice of training entities so the probabilities can
     feed the admission bar directly (--no-calibrate keeps the raw output).
  4. Per-entity admission under the metric's own rule, then a one-to-one pass:
     every target record belongs to at most one Source 1 entity, so a target
     claimed by two entities is kept only by the one that scores it highest.

Admission follows from F0.5 = 1.25c / (n + 0.25k): adding a candidate of
probability p raises the expected score only while p exceeds the current score
divided by 1.25, so the bar rises with each id admitted.

Run:
    python src/pair_model.py --entities 40000 --max-df 300
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import re

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import er_io
from src.candidates import build_token_postings, lookup_postings
from src.er_norm import (FILLER, concat_keys, core_name, cross_keys, domain_prefix_key, hash64,
                         house_numbers, name_pair_keys, normalize_name, number_word_keys,
                         number_word_keys_from_tokens, number_word_keys_loose, postcodes,
                         repair_name, strip_accents, word_pair_keys)

ENCODE = 2_000_000_000

# Token postings families: name tokens, address tokens, address number-word
# keys that skip words under 4 characters ("36 rue de segur" -> "36 segur"), and
# pairs of core name words, which survive an edit to any one word.
#
# The last three reach copies corrupted in the name and the address at once,
# where every single-field key either breaks or is too common to keep: "rn"
# repairs look-alike digits inside name tokens, "dp" keys on the opening
# characters of the spaceless name so a domain-style rewrite still meets its
# spelled-out form, and "xk" pairs a distinctive name word with an address word.
TOKEN_FIELDS = (1, 2, "nw4", "core", "ww", "np", "cc")
# Used only by the rescue stage (src/rescue.py), never mixed into the top-k lists.
RESCUE_FIELDS = ("rn", "dp", "xk")

# A cross key is only worth keeping while it is rare: at the global cap it adds
# 41% more raw candidates for 31% of the never-blocked pairs, at 50 it adds 12%
# for 26%, at 20 it adds 5% for 20%. Measured in results/blocking_keys/recall.json.
#
# These four families are NOT enabled on main. They recover 39.5% of the pairs
# no current key reaches, but the pairs arrive into a fixed top-k pre-ranker and
# evict true matches already found: held-out F0.5 falls 0.00080 at df 50 and
# 0.00076 at df 20, every fold down both times, while the loss to prefiltering
# rises from 0.00104 to 0.00595 and 0.00568. Cutting the added candidates in
# half barely moves that, so the volume is not what does the damage - the added
# pairs score well on trigram overlap, which is exactly what the pre-ranker
# ranks by. See results/blocking_keys/cv4m5_keys{,_df20}.json.
FIELD_MAX_DF = {"xk": 20}
THRESHOLDS = (0.6, 0.7, 0.75, 0.8)
NW_MIN_LEN = 4


def sample_entities(n: int, seed: int, extra: int = 0):
    """Fixed sample of Source 1 rows with their true match sets.

    extra draws that many further rows from outside the sample (a separate
    generator, so the first n rows are the same whatever extra is). They are
    returned in `rows` as well; the caller uses them for training only.
    """
    rng = np.random.default_rng(seed)
    s1 = er_io.parse_source("train", 1)
    gt = er_io.parse_ground_truth()
    rows = np.sort(rng.choice(len(s1["ids"]), size=min(n, len(s1["ids"])), replace=False))
    if extra:
        rest = np.setdiff1d(np.arange(len(s1["ids"])), rows)
        more = np.random.default_rng(seed + 1).choice(rest, size=min(extra, len(rest)), replace=False)
        rows = np.sort(np.concatenate([rows, more]))
    row_of = er_io.index_of(s1["ids"], gt["s1"])
    gt_of_row = np.full(len(s1["ids"]), -1, dtype=np.int64)
    gt_of_row[row_of] = np.arange(len(gt["s1"]))
    truth = {}
    for row in rows.tolist():
        g = gt_of_row[row]
        lo, hi = gt["offsets"][g], gt["offsets"][g + 1]
        truth[row] = set((gt["match_src"][lo:hi].astype(np.int64) * ENCODE
                          + gt["match_id"][lo:hi]).tolist())
    return s1, rows, truth


def build_index(split: str, max_df: int, fields=None) -> dict:
    """Sorted blocking keys for both target sources, built once per split.

    Per source: whole-name and token-sorted keys, house-number postings, and
    the token postings in TOKEN_FIELDS, all folded with the country. `fields`
    narrows the token families to a subset, which lets one family be measured
    on its own.
    """
    fields = TOKEN_FIELDS if fields is None else fields
    index = {}
    for source in (2, 3):
        target = er_io.parse_source(split, source)
        ctry_t = er_io.country_keys(target)
        entry = {}
        for col in ("name_key", "sort_key"):
            tk = er_io.combine_keys(target[col], ctry_t)
            order = np.argsort(tk, kind="stable")
            entry[col] = (tk[order], order)

        hr, hk = [], []
        for slot in range(target["house"].shape[1]):
            v = target["house"][:, slot]
            keep = np.flatnonzero(v != er_io.EMPTY)
            hr.append(keep.astype(np.int32))
            hk.append(er_io.combine_keys(v[keep].astype(np.int64), ctry_t[keep]))
        hkey = np.concatenate(hk)
        hrow = np.concatenate(hr)
        o = np.argsort(hkey, kind="stable")
        entry["house"] = (hkey[o], hrow[o])

        for field in fields:
            entry[f"tok{field}"] = build_token_postings(
                split, source, min(FIELD_MAX_DF.get(field, max_df), max_df), field=field)
        index[source] = entry
        del target
    return index


def read_s1_text(split: str, rows) -> dict:
    """Raw (name, address, country) for the given Source 1 rows, from one streaming pass."""
    want = set(rows.tolist())
    text = {}
    with open(C.source_file(split, 1), encoding="utf-8") as f:
        next(f, None)
        for i, line in enumerate(f):
            if i in want:
                p = line.rstrip("\n").split("\t")
                if len(p) == 4:
                    text[i] = (p[1], p[2], p[3])
                if len(text) == len(want):
                    break
    return text


def _query_tokens(field, name: str, address: str):
    """The Source 1 side's tokens for one postings family."""
    if field == "nw4":
        return number_word_keys(address, min_len=NW_MIN_LEN)
    if field == "core":
        core = core_name(normalize_name(name))
        return [core] if core else []
    if field == "np":
        return name_pair_keys(core_name(normalize_name(name)))
    if field == "cc":
        return concat_keys(normalize_name(name))
    if field == "ww":
        return word_pair_keys(address)
    if field == "rn":
        core = core_name(repair_name(normalize_name(name)))
        return [core] if core else []
    if field == "dp":
        dp = domain_prefix_key(repair_name(normalize_name(name)))
        return [dp] if dp else []
    if field == "nx":
        return number_word_keys_loose(address, min_len=NW_MIN_LEN)
    if field == "xk":
        return cross_keys(name, address)
    return normalize_name(name if field == 1 else address).split()


def gather(rows, s1, ctry_s1, text, index, max_group: int, fields=None, base_keys=True):
    """Candidate target rows per entity from a prebuilt index.

    Returns {s1_row: {source: set(target_row)}} using whole-name, token-sorted,
    house-number, name-token, address-token and number-word keys, all within
    country. `fields` narrows the token families and base_keys=False drops the
    whole-name and house-number families, so one family's own contribution can
    be measured.
    """
    fields = TOKEN_FIELDS if fields is None else fields
    out = {r: {} for r in rows.tolist()}
    for source in (2, 3):
        e = index[source]
        for r in rows.tolist():
            out[r][source] = set()

        for col in (("name_key", "sort_key") if base_keys else ()):
            sk, order = e[col]
            qk = er_io.combine_keys(s1[col][rows], ctry_s1[rows])
            for i, hits in enumerate(lookup_postings(sk, order, qk, max_group)):
                if len(hits):
                    out[int(rows[i])][source] |= set(hits.tolist())

        hkey, hrow = e["house"]
        for slot in (range(s1["house"].shape[1]) if base_keys else ()):
            v = s1["house"][rows, slot]
            live = v != er_io.EMPTY
            qk = er_io.combine_keys(v.astype(np.int64), ctry_s1[rows])
            for i, hits in enumerate(lookup_postings(hkey, hrow, qk, max_group)):
                if live[i] and len(hits):
                    out[int(rows[i])][source] |= set(hits.tolist())

        for field in fields:
            tkey, trow = e[f"tok{field}"]
            for i, r in enumerate(rows.tolist()):
                toks = sorted({hash64(t) for t in _query_tokens(field, *text[r][:2])})
                if not toks:
                    continue
                qk = er_io.combine_keys(np.array(toks, dtype=np.int64),
                                        np.full(len(toks), ctry_s1[r], dtype=np.int64))
                for hits in lookup_postings(tkey, trow, qk, max_group):
                    if len(hits):
                        out[r][source] |= set(hits.tolist())
    return out


def gather_candidates(rows, s1, max_df: int, max_group: int, split="train"):
    """Candidate target rows per sampled entity, plus the entities' raw text."""
    text = read_s1_text(split, rows)
    index = build_index(split, max_df)
    return gather(rows, s1, er_io.country_keys(s1), text, index, max_group), text


FEATURES = ["name_jac", "name_cont", "name_eq", "name_sort_eq", "name_len_ratio",
            "addr_jac", "addr_cont", "addr_eq", "house_hit", "pc_hit",
            "name_char_jac", "addr_char_jac", "n_cand", "best_rank", "nw_shared",
            "addr_empty_s1", "addr_empty_t", "name_nonlatin_s1", "name_nonlatin_t",
            "num_jac", "num_cont", "num_both", "name_only_s1", "name_only_t",
            "num_trunc", "num_sibling",
            "name_tset", "name_partial", "addr_tset", "addr_partial",
            "t_name_min_df", "s1_only_max_df", "t_only_max_df", "only_tok_ratio",
            "s1_core_in_s1", "t_core_in_s1", "t_core_in_targets",
            "t_addr_in_s1", "s1_addr_coloc", "nw_s1_mult", "num_offset",
            "addr_idf_cont", "addr_idf_shared_max", "addr_idf_s1only_max",
            "legal_conflict", "legal_one_side", "added_real", "dropped_real", "bracket_added",
            "added_filler"]
# Optional pair-difference columns (src/diff_features.py): ER_DIFF=name or ER_DIFF=address.
DIFF = os.environ.get("ER_DIFF", "")
if DIFF:
    from src.diff_features import ADDR_COLS, NAME_COLS, addr_diff, name_align
    FEATURES += {"name": NAME_COLS, "address": ADDR_COLS}[DIFF]


def _num_offset(left, right) -> float:
    """Signed gap (right minus left) of the closest address numbers when none is shared.

    Decoy records sit at small positive offsets from the real number (3423 ->
    3434), while noise on true copies moves in both directions; the sign and
    size together carry the signal. NaN when a number is shared or missing.
    """
    if not left or not right or left & right:
        return float("nan")
    a, b = min(((x, y) for x in left for y in right), key=lambda p: abs(p[1] - p[0]))
    return float(max(-50, min(50, b - a)))


_ADDR_COUNTS = {}


def _addr_key(norm_addr: str) -> str:
    """Order-free exact address key: the sorted distinct normalized tokens."""
    return " ".join(sorted(set(norm_addr.split())))


def _loose_addr_key(norm_addr: str) -> str:
    """First number (zeros stripped) plus the sorted alphabetic tokens."""
    toks = norm_addr.split()
    num = next((str(int(t)) for t in toks if t.isdigit() and len(t) <= 6), "")
    return num + "|" + " ".join(sorted({t for t in toks if not t.isdigit()}))


def addr_counts(split: str) -> tuple:
    """Source 1 counts per exact address, loose address and number-word key.

    The address counterpart of core_counts. A candidate whose address belongs
    to exactly one Source 1 business is far more likely to be that business
    than one at an address several Source 1 entities share, where a record
    with an unrelated name is a coin flip between them.
    """
    if split in _ADDR_COUNTS:
        return _ADDR_COUNTS[split]
    import collections
    cache = er_io.CACHE_DIR / f"{split}_addr_counts.v{er_io.CACHE_VERSION}.json"
    if cache.exists():
        blob = json.loads(cache.read_text(encoding="utf-8"))
    else:
        exact, loose, nw = collections.Counter(), collections.Counter(), collections.Counter()
        with open(C.source_file(split, 1), encoding="utf-8") as f:
            next(f, None)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 4:
                    continue
                na = normalize_name(parts[2])
                if not na:
                    continue
                exact[str(hash64(_addr_key(na)))] += 1
                loose[str(hash64(_loose_addr_key(na)))] += 1
                for k in number_word_keys_from_tokens(na.split(), min_len=NW_MIN_LEN):
                    nw[str(hash64(k))] += 1
        blob = [dict(exact), dict(loose), dict(nw)]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(blob), encoding="utf-8")
    _ADDR_COUNTS[split] = tuple(blob)
    return _ADDR_COUNTS[split]

_CORE_COUNTS = {}


def core_counts(split: str) -> tuple:
    """({core hash: Source 1 entities}, {core hash: Source 2/3 records}) for a split.

    A matching name means much more when no other Source 1 entity carries it:
    a blank-address record named like a unique business is almost certainly
    that business, while one named like fifty branches is a coin flip.
    """
    if split in _CORE_COUNTS:
        return _CORE_COUNTS[split]
    import collections
    cache = er_io.CACHE_DIR / f"{split}_core_counts.v{er_io.CACHE_VERSION}.json"
    if cache.exists():
        blob = json.loads(cache.read_text(encoding="utf-8"))
    else:
        blob = []
        for sources in ((1,), (2, 3)):
            cnt = collections.Counter()
            for source in sources:
                with open(C.source_file(split, source), encoding="utf-8") as f:
                    next(f, None)
                    for line in f:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) == 4:
                            core = core_name(normalize_name(parts[1]))
                            if core:
                                cnt[str(hash64(core))] += 1
            # Source 1 keeps every count (1 vs 0 is the unique-name signal);
            # targets keep 2+, and a missing target name counts as 1.
            blob.append(dict(cnt) if sources == (1,) else {k: v for k, v in cnt.items() if v >= 2})
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(blob), encoding="utf-8")
    _CORE_COUNTS[split] = (blob[0], blob[1])
    return _CORE_COUNTS[split]

_NAME_DF = {}


def name_token_df(split: str) -> dict:
    """{token: records} over the split's Source 2 and 3 names, tokens seen 2+ times.

    A token missing from the table occurs once. Cached to disk, and once per
    process, since feature workers call it for every chunk.
    """
    if split in _NAME_DF:
        return _NAME_DF[split]
    import collections
    cache = er_io.CACHE_DIR / f"{split}_name_df.v{er_io.CACHE_VERSION}.json"
    if cache.exists():
        table = json.loads(cache.read_text(encoding="utf-8"))
    else:
        cnt = collections.Counter()
        for source in (2, 3):
            with open(C.source_file(split, source), encoding="utf-8") as f:
                next(f, None)
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) == 4:
                        cnt.update(set(normalize_name(parts[1]).split()))
        table = {t: n for t, n in cnt.items() if n >= 2}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(table, ensure_ascii=False), encoding="utf-8")
    _NAME_DF[split] = table
    return table

_ADDR_W = {}


def addr_token_w(split: str) -> dict:
    """{country: {address token: log10(records / records with the token)}} over the split's three sources.

    Counted per country label over Source 1, 2 and 3 addresses, tokens that
    are not all digits, seen in 2+ records; a token missing from the table occurs once in
    the split (a typo or a one-off word) and weighs 0. The weight depends only
    on how common a token is within its own country, so region, state, city
    and street-type words come out around 1 and street names 2-5 in every
    country, and features built on it read the same in a country that never
    appears in training. Weights are capped at 5.0, reached by any token in
    fewer than 1 in 100,000 of the country's records: below that the count
    is set by one or a few businesses, so an uncapped weight would move with
    the size of the split and the country rather than with the token. Cached
    to disk as counts plus the record total per country, and converted to
    weights once per process.
    """
    if split in _ADDR_W:
        return _ADDR_W[split]
    import collections
    cache = er_io.CACHE_DIR / f"{split}_addr_df.v{er_io.CACHE_VERSION}.json"
    if cache.exists():
        blob = json.loads(cache.read_text(encoding="utf-8"))
    else:
        cnt, n = collections.defaultdict(collections.Counter), collections.Counter()
        for source in (1, 2, 3):
            with open(C.source_file(split, source), encoding="utf-8") as f:
                next(f, None)
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) != 4:
                        continue
                    toks = {t for t in normalize_name(parts[2]).split() if not t.isdigit()}
                    if toks:
                        n[parts[3]] += 1
                        cnt[parts[3]].update(toks)
        blob = {c: {"n": n[c], "df": {t: k for t, k in cnt[c].items() if k >= 2}} for c in n}
        cache.parent.mkdir(parents=True, exist_ok=True)
        # Written to a temporary name and renamed, so a reader never sees a
        # partial file.
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
        tmp.replace(cache)
    _ADDR_W[split] = {c: {t: min(math.log10(v["n"] / k), 5.0) for t, k in v["df"].items()}
                      for c, v in blob.items()}
    return _ADDR_W[split]


def _number_relation(left, right):
    """(truncation, sibling) flags between two sets of address numbers.

    The noise drops a leading or trailing digit ("414" -> "14", "63/67" ->
    "63/6"), so one number being a proper prefix or suffix of the other is
    agreement. A different business at the same site has a number of the same
    length with one or two digits changed ("2251" vs "2256"), which is not.
    """
    trunc = sib = 0.0
    shared = left & right
    for a in left - shared:
        sa = str(a)
        for b in right - shared:
            sb = str(b)
            if len(sa) != len(sb):
                short, long_ = (sa, sb) if len(sa) < len(sb) else (sb, sa)
                if long_.startswith(short) or long_.endswith(short):
                    trunc = 1.0
            elif sum(x != y for x, y in zip(sa, sb)) <= 2:
                sib = 1.0
    return trunc, sib


def _core_feats(core: str, in_s1: dict, in_t: dict):
    """log counts of Source 1 entities and target records sharing a core name."""
    if not core:
        return 0.0, 0.0
    h = str(hash64(core))
    return np.log1p(in_s1.get(h, 0)), np.log1p(in_t.get(h, 1))


def _addr_w(lwa: dict, l_mass: float, rwa: dict):
    """(weighted containment, rarest shared weight, rarest Source 1-only weight) of two addresses.

    Both sides are {token: weight} maps from addr_token_w. Containment is the
    shared weight over the lighter side's total, so a dropped or swapped
    region, state or city (weight ~1) barely moves it while a different street
    (weight 2-5) collapses it. The rarest shared token says whether the
    agreement reaches street level; the rarest Source 1 token the target lacks
    says whether only ballast or the street itself is missing. With no
    weighted token on either side the first two are 0.0, like _jac, and the
    third is NaN: missing text is not disagreement.
    """
    r_mass = sum(rwa.values())
    if not l_mass or not r_mass:
        return 0.0, 0.0, float("nan")
    shared = [w for t, w in rwa.items() if t in lwa]
    return (sum(shared) / min(l_mass, r_mass), max(shared, default=0.0),
            max((w for t, w in lwa.items() if t not in rwa), default=0.0))


def _nonlatin(s: str) -> float:
    """1.0 when a normalized string holds non-ASCII letters (Devanagari, Bengali, ...)."""
    return float(any(ord(c) > 127 and c.isalpha() for c in s))


_LEGAL_US = {"inc": "inc", "incorporated": "inc", "llc": "llc", "corp": "corp", "corporation": "corp",
             "pllc": "pllc", "lp": "lp", "plc": "plc", "pc": "pc", "llp": "llp", "co": "co", "company": "co"}
_LEGAL_FR = frozenset("sarl sas sasu sa eurl snc sci selarl scop gie sca scs selas".split())
_LEGAL_TOK = frozenset(_LEGAL_US) | _LEGAL_FR | {"private", "pvt", "limited", "ltd", "opc"}
_ALNUM = re.compile(r"[^a-z0-9]+")
_BRACKET = re.compile(r"[\[(]")


def _legal_forms(raw: str) -> frozenset:
    """Canonical legal forms a raw business name carries (US, Indian and French forms).

    Spelled-out and dotted variants collapse ("L.L.C." -> llc, "S.A.R.L." -> sarl), and
    "Private Limited" / "Pvt Ltd" are one form, distinct from a bare "Limited".
    """
    s = _ALNUM.sub(" ", strip_accents(raw.lower()))
    for pat, rep in ((r"\bl l c\b", "llc"), (r"\bl l p\b", "llp"), (r"\bs a r l\b", "sarl"),
                     (r"\bs a s u\b", "sasu"), (r"\bs a s\b", "sas"), (r"\bs a\b", "sa")):
        s = re.sub(pat, rep, s)
    toks = s.split(); ts = set(toks); forms = set()
    private = bool(ts & {"private", "pvt"}) or " p ltd" in f" {s}"
    if ts & {"ltd", "limited"}:
        forms.add("pvt_ltd" if private else "ltd")
    elif private:
        forms.add("pvt_ltd")
    for w in toks:
        if w in _LEGAL_US:
            forms.add(_LEGAL_US[w])
        elif w in _LEGAL_FR:
            forms.add(w)
    return frozenset(forms)


def _real_tokens(normalized: str) -> list:
    """Name tokens that identify the business: no legal forms, filler words or single letters."""
    return [w for w in normalized.split() if len(w) > 1 and w not in FILLER and w not in _LEGAL_TOK]


def _unmatched(a, b, fuzz) -> int:
    """How many tokens of a have no exact, prefix or near-spelling counterpart in b."""
    return sum(1 for w in a if not any(w == v or (len(w) >= 3 and (v.startswith(w) or w.startswith(v)))
                                       or fuzz.ratio(w, v) >= 80 for v in b))


def _decoy_feats(l_legal, l_real, l_br, l_toks, raw_t: str, norm_t: str, fuzz):
    """Legal-form conflict, one-sided legal form, added and dropped identifying words, added brackets, added filler words.

    Near-duplicate records of a different business keep the name but swap the legal
    form or add a real word; true copies abbreviate the form or tag the name in brackets.
    """
    t_legal = _legal_forms(raw_t); t_real = _real_tokens(norm_t)
    return (float(bool(l_legal) and bool(t_legal) and l_legal.isdisjoint(t_legal)),
            float(bool(l_legal) != bool(t_legal)),
            float(_unmatched(t_real, l_real, fuzz)), float(_unmatched(l_real, t_real, fuzz)),
            float(bool(_BRACKET.search(raw_t)) and not l_br),
            float(sum(1 for w in set(norm_t.split()) if w in FILLER and w not in _LEGAL_TOK and w not in l_toks)))


def _diff(kind, ln, nn, s1_addr, t_addr, df, lw):
    """Name-token alignment or address-number-role columns; empty unless ER_DIFF selects a bundle."""
    if not kind:
        return ()
    if kind == "name":
        a = [t for t in ln.split() if t not in FILLER]; b = [t for t in nn.split() if t not in FILLER]
        w = lambda t: float(np.log10(1e7 / df.get(t, 1)))
        return tuple(name_align(a, b, [w(t) for t in a], [w(t) for t in b]))
    return tuple(addr_diff(s1_addr, t_addr, "", None, None, wf=lambda t: lw.get(t, 3.0), rare=2.0))


def _jac(a: set, b: set):
    """Jaccard and containment of two token sets."""
    if not a or not b:
        return 0.0, 0.0
    inter = len(a & b)
    return inter / len(a | b), inter / min(len(a), len(b))


def _chars(s: str, n=3):
    """Character n-gram set, for typo tolerance."""
    s = f"  {s} "
    return {s[i:i + n] for i in range(len(s) - n + 1)}


PRERANK_FEATURES = ["addr_char_jac", "name_char_jac", "addr_jac", "name_jac",
                    "addr_empty_t", "house_hit", "n_raw"]


def prerank_scores(cands, s1_text, targets_text, full=False):
    """(keys, addr trigram Jaccard, name trigram Jaccard) for every raw pair.

    The two numbers the pre-ranker needs, in the exact order build_features
    walks the pairs, at a fraction of the cost of the full feature set. With
    full, a third value is returned: a small matrix of PRERANK_FEATURES for a
    learned pre-ranker (still cheap: token sets and one digit scan per pair).
    """
    keys, a, n, extra = [], [], [], []
    for r, per_source in cands.items():
        ln, la = normalize_name(s1_text[r][0]), normalize_name(s1_text[r][1])
        lcn, lca = _chars(ln), _chars(la)
        ltn, lta = set(ln.split()), set(la.split())
        lh = house_numbers(s1_text[r][1]) if full else None
        n_raw = sum(len(v) for v in per_source.values())
        for source, rows in per_source.items():
            for tr in rows:
                rn, ra = targets_text[source][tr]
                nn, na = normalize_name(rn), normalize_name(ra)
                n.append(_jac(lcn, _chars(nn))[0])
                a.append(_jac(lca, _chars(na))[0])
                keys.append((r, source, tr))
                if full:
                    extra.append((_jac(lta, set(na.split()))[0], _jac(ltn, set(nn.split()))[0],
                                  float(not na), float(bool(lh & house_numbers(ra))),
                                  float(n_raw)))
    a, n = np.array(a, dtype=np.float32), np.array(n, dtype=np.float32)
    if not full:
        return keys, a, n
    P = np.column_stack([a, n, np.array(extra, dtype=np.float32).reshape(-1, 5)])
    return keys, a, n, P


def build_features(cands, s1_text, targets_text, split="train", top_k=None, n_cand_of=None,
                   preranker=None):
    """Feature matrix for every candidate pair, plus its (s1_row, source, target_row).

    With top_k, only the pairs the pre-ranker keeps get the full feature set:
    the two pre-ranker similarities are computed for every raw pair first,
    the survivors are chosen exactly as prefilter would, and the result is
    identical to building everything and then prefiltering, at about half
    the cost. n_cand_of carries each entity's raw candidate count through.
    """
    if top_k is not None:
        if preranker is None:
            keys, a_, n_ = prerank_scores(cands, s1_text, targets_text)
            keep = prefilter_index(a_, n_, keys, top_k)
        else:
            import lightgbm as lgb
            keys, a_, n_, P = prerank_scores(cands, s1_text, targets_text, full=True)
            score = lgb.Booster(model_str=preranker).predict(P) if len(keys) else a_
            keep = prefilter_index(a_, n_, keys, top_k, score)
        sub = {}
        for i in keep:
            r, src, tr = keys[i]
            sub.setdefault(r, {}).setdefault(src, []).append(tr)
        raw = {r: sum(len(v) for v in per.values()) for r, per in cands.items()}
        return build_features(sub, s1_text, targets_text, split, n_cand_of=raw)
    from rapidfuzz import fuzz
    df = name_token_df(split)
    in_s1, in_t = core_counts(split)
    a_exact, a_loose, a_nw = addr_counts(split)
    W = addr_token_w(split)
    feats, keys = [], []
    for r, per_source in cands.items():
        n_cand = (n_cand_of[r] if n_cand_of is not None
                  else sum(len(v) for v in per_source.values()))
        ln, la = normalize_name(s1_text[r][0]), normalize_name(s1_text[r][1])
        ltn, lta = set(ln.split()), set(la.split())
        lcn, lca = _chars(ln), _chars(la)
        lh, lp = house_numbers(s1_text[r][1]), postcodes(s1_text[r][1])
        lnw = number_word_keys_from_tokens(la.split(), min_len=NW_MIN_LEN)
        l_nl = _nonlatin(ln)
        l_core = core_name(ln)
        l_core_n = np.log1p(in_s1.get(str(hash64(l_core)), 1)) if l_core else 0.0
        l_coloc = (np.log1p(a_loose.get(str(hash64(_loose_addr_key(la))), 1) - 1)
                   if la else 0.0)
        # Address token weights of the entity's own country; digits and tokens
        # seen once in the split are not in the table and drop out here.
        l_legal = _legal_forms(s1_text[r][0]); l_real = _real_tokens(ln)
        l_br = bool(_BRACKET.search(s1_text[r][0]))
        lw = W.get(s1_text[r][2], {})
        lwa = {t: lw[t] for t in lta if t in lw}
        l_mass = sum(lwa.values())
        for source, rows in per_source.items():
            for tr in rows:
                rn, ra = targets_text[source][tr]
                nn, na = normalize_name(rn), normalize_name(ra)
                rtn, rta = set(nn.split()), set(na.split())
                nj, nc = _jac(ltn, rtn)
                aj, ac = _jac(lta, rta)
                ncj, _ = _jac(lcn, _chars(nn))
                acj, _ = _jac(lca, _chars(na))
                th = house_numbers(ra)
                nums = len(lh & th)
                w_cont, w_shared, w_s1only = _addr_w(lwa, l_mass,
                                                     {t: lw[t] for t in rta if t in lw})
                feats.append((
                    nj, nc, float(ln == nn),
                    float(" ".join(sorted(ltn)) == " ".join(sorted(rtn))),
                    min(len(ln), len(nn)) / max(len(ln), len(nn), 1),
                    aj, ac, float(la == na),
                    float(bool(nums)),
                    float(bool(lp & postcodes(ra))),
                    ncj, acj, float(n_cand), 0.0,
                    float(len(lnw & number_word_keys_from_tokens(na.split(),
                                                                 min_len=NW_MIN_LEN))),
                    # Missing text is not disagreement: a blank address or a
                    # name in another script zeroes the similarities above,
                    # and these flags let the model lean on the other field.
                    float(not la), float(not na), l_nl, _nonlatin(nn),
                    # Sibling records differ by one number ("25-112" vs
                    # "25-133"); one shared number is not agreement.
                    nums / len(lh | th) if lh and th else 0.0,
                    nums / len(lh) if lh else 0.0,
                    float(bool(lh) and bool(th)),
                    float(len(ltn - rtn)), float(len(rtn - ltn)),
                    *_number_relation(lh, th),
                    # Typo-tolerant token matching ("Memblell" / "Merrell",
                    # "CIYT" / "City") that exact-token Jaccard cannot see.
                    fuzz.token_set_ratio(ln, nn) / 100, fuzz.partial_ratio(ln, nn) / 100,
                    fuzz.token_set_ratio(la, na) / 100, fuzz.partial_ratio(la, na) / 100,
                    # True copies keep the name up to typos or replace it with
                    # a one-off nonsense word; a different business at the
                    # same site swaps in another real, common word.
                    np.log1p(min((df.get(t, 1) for t in rtn), default=0)),
                    np.log1p(max((df.get(t, 1) for t in ltn - rtn), default=0)),
                    np.log1p(max((df.get(t, 1) for t in rtn - ltn), default=0)),
                    max((fuzz.ratio(a, b) for a in ltn - rtn for b in rtn - ltn),
                        default=0) / 100,
                    l_core_n, *_core_feats(core_name(nn), in_s1, in_t),
                    np.log1p(a_exact.get(str(hash64(_addr_key(na))), 0)) if na else -1.0,
                    l_coloc,
                    np.log1p(min((a_nw.get(str(hash64(k)), 1) for k in lnw
                                  & number_word_keys_from_tokens(na.split(),
                                                                 min_len=NW_MIN_LEN)),
                                 default=0)),
                    _num_offset(lh, th),
                    # Address agreement weighted by how rare each token is in
                    # the country: region and city words count for little,
                    # the street name for a lot.
                    w_cont, w_shared, w_s1only,
                    *_decoy_feats(l_legal, l_real, l_br, ltn, rn, nn, fuzz),
                    *_diff(DIFF, ln, nn, s1_text[r][1], ra, df, lw)))
                keys.append((r, source, tr))
    return np.array(feats, dtype=np.float32), keys




PREFILTER_SOLO = 8
GROUP_BASE = ("name_char_jac", "addr_char_jac")
GROUP_FEATURES = [f"{f}_{kind}" for f in GROUP_BASE for kind in ("gap", "rank")]
DECOY_GROUP = ("legal_conflict", "added_real", "added_filler", "legal_one_side")
GROUP_FEATURES += [f"{f}_over_min" for f in DECOY_GROUP]


def add_group_features(X, keys):
    """Append each GROUP_BASE similarity's gap to the entity's best and its rank.

    Computed over the entity's pre-ranked candidates, so train and test see the
    same set. Being the closest of ~50 candidates says more than an absolute
    similarity does; judged under the threshold rule this is worth about
    +0.007 held out (it looked harmful only under greedy admission).
    """
    if not len(keys):
        return np.zeros((0, X.shape[1] + len(GROUP_FEATURES)), dtype=np.float32)
    ent = np.fromiter((k[0] for k in keys), dtype=np.int64, count=len(keys))
    _, g = np.unique(ent, return_inverse=True)
    cols = []
    for f in GROUP_BASE:
        col = X[:, FEATURES.index(f)].astype(np.float64)
        best = np.full(g.max() + 1, -np.inf)
        np.maximum.at(best, g, col)
        order = np.lexsort((-col, g))
        sg = g[order]
        first = np.r_[0, np.flatnonzero(np.diff(sg)) + 1]
        within = np.arange(len(col)) - np.repeat(first, np.diff(np.r_[first, len(col)]))
        rank = np.empty(len(col))
        rank[order] = within
        cols += [col - best[g], rank]
    for f in DECOY_GROUP:
        # How much more of a decoy sign this candidate shows than the entity's cleanest one.
        col = X[:, FEATURES.index(f)].astype(np.float64)
        low = np.full(g.max() + 1, np.inf)
        np.minimum.at(low, g, col)
        cols.append(col - low[g])
    return np.hstack([X, np.column_stack(cols).astype(np.float32)])


def prefilter(X, keys, k):
    """Keep each entity's pre-ranked candidates; see prefilter_index."""
    if k is None or not len(keys):
        return X, keys
    keep = prefilter_index(X[:, FEATURES.index("addr_char_jac")],
                           X[:, FEATURES.index("name_char_jac")], keys, k)
    return X[keep], [keys[i] for i in keep]


def prefilter_index(a_, n_, keys, k, score=None):
    """Keep each entity's k candidates with the highest name + address trigram overlap.

    This is the last blocking stage: the matcher is trained and run only on
    what survives, and candidate_pairs.tsv lists exactly that set. Dropping
    the long tail of easy negatives also scores higher than keeping it (on the
    25k sample, top 50 of ~290 kept 98.7% of true pairs). Order is preserved;
    ties go to the earlier row. k=None keeps everything.
    """
    ent = np.fromiter((kk[0] for kk in keys), dtype=np.int64, count=len(keys))
    _, g = np.unique(ent, return_inverse=True)

    def rank_by(score):
        """Rank of each pair within its entity by score, 0 = best."""
        order = np.lexsort((-score, g))
        sg = g[order]
        first = np.r_[0, np.flatnonzero(np.diff(sg)) + 1]
        within = np.arange(len(score)) - np.repeat(first, np.diff(np.r_[first, len(score)]))
        rank = np.empty(len(score), dtype=np.int64)
        rank[order] = within
        return rank

    # The combined score decides most slots; a few are reserved for the best
    # candidates on each field alone, so a copy whose name was replaced by a
    # nonsense word (or whose address was blanked) is not crowded out.
    main = a_ + n_ if score is None else score
    keep = np.flatnonzero((rank_by(main) < k - 2 * PREFILTER_SOLO)
                          | (rank_by(a_) < PREFILTER_SOLO) | (rank_by(n_) < PREFILTER_SOLO))
    return keep


def prerank_matrix(cands, s1_text, targets_text):
    """(keys, PRERANK_FEATURES matrix) for every raw pair; a pool worker entry point."""
    keys, _, _, P = prerank_scores(cands, s1_text, targets_text, full=True)
    return P, keys


def train_preranker(n_entities: int, exclude, max_df: int, max_group: int, pool=None,
                    workers=1) -> str:
    """Fit a small LightGBM that ranks raw candidates for the top-k cut.

    Trained on its own entities, disjoint from `exclude` (the entities being
    evaluated), so held-out scores never see a ranker fitted on them. Returns
    the model as a string so pool workers can rebuild it.
    """
    import lightgbm as lgb
    s1 = er_io.parse_source("train", 1)
    gt = er_io.parse_ground_truth()
    pool_rows = np.setdiff1d(np.arange(len(s1["ids"])), np.fromiter(exclude, dtype=np.int64))
    rows = np.sort(np.random.default_rng(C.SEED + 101).choice(pool_rows, size=n_entities,
                                                               replace=False))
    row_of = er_io.index_of(s1["ids"], gt["s1"])
    gt_of_row = np.full(len(s1["ids"]), -1, dtype=np.int64)
    gt_of_row[row_of] = np.arange(len(gt["s1"]))
    enc = gt["match_src"].astype(np.int64) * ENCODE + gt["match_id"]
    truth = {r: set(enc[gt["offsets"][gt_of_row[r]]:gt["offsets"][gt_of_row[r] + 1]].tolist())
             for r in rows.tolist()}
    cands, text = gather_candidates(rows, s1, max_df, max_group)
    tgt = fetch_target_text(cands)
    if pool is None or workers <= 1:
        P, keys = prerank_matrix(cands, text, tgt)
    else:
        ents = list(cands)
        size = max(1, -(-len(ents) // (workers * 4)))
        jobs = []
        for i in range(0, len(ents), size):
            part = ents[i:i + size]
            tt = {s: {tr: tgt[s][tr] for r in part for tr in cands[r].get(s, ())} for s in (2, 3)}
            jobs.append(({r: cands[r] for r in part}, {r: text[r] for r in part}, tt))
        parts = [x for x in pool.starmap(prerank_matrix, jobs) if x[1]]
        P = np.concatenate([x[0] for x in parts])
        keys = [k for x in parts for k in x[1]]
    tid = {s: er_io.parse_source("train", s)["ids"] for s in (2, 3)}
    y = np.array([1 if (src * ENCODE + tid[src][tr]) in truth[r] else 0 for r, src, tr in keys],
                 dtype=np.int8)
    booster = lgb.train({"objective": "binary", "learning_rate": 0.1, "num_leaves": 31,
                         "min_data_in_leaf": 100, "verbose": -1, "seed": C.SEED},
                        lgb.Dataset(P, label=y), num_boost_round=150)
    print(f"pre-ranker trained on {len(rows):,} entities, {len(y):,} raw pairs "
          f"({y.mean() * 100:.2f}% positive)", flush=True)
    return booster.model_to_string()


def build_features_parallel(cands, s1_text, targets_text, pool=None, workers=1, split="train",
                            top_k=None, preranker=None):
    """build_features split by entity across a process pool.

    Chunks keep the entity order, so rows come back in the same order as the
    serial call and the matrix is identical.
    """
    if pool is None or workers <= 1:
        return build_features(cands, s1_text, targets_text, split, top_k, None, preranker)
    # The per-split tables are built and cached here, once, so the workers
    # only ever read them.
    addr_token_w(split)
    ents = list(cands)
    size = max(1, -(-len(ents) // (workers * 4)))
    jobs = []
    for i in range(0, len(ents), size):
        part = ents[i:i + size]
        tt = {s: {tr: targets_text[s][tr] for r in part for tr in cands[r].get(s, ())}
              for s in (2, 3)}
        jobs.append(({r: cands[r] for r in part}, {r: s1_text[r] for r in part}, tt, split, top_k,
                     None, preranker))
    parts = [p for p in pool.starmap(build_features, jobs) if p[1]]
    if not parts:
        return np.zeros((0, len(FEATURES)), dtype=np.float32), []
    return (np.concatenate([p[0] for p in parts]),
            [k for p in parts for k in p[1]])


def fetch_target_text(cands, split="train"):
    """Raw name and address for every candidate target row, one pass per source."""
    out = {}
    for source in (2, 3):
        rows = sorted({tr for per in cands.values() for tr in per.get(source, ())})
        out[source] = er_io.fetch_fields(split, source, np.array(rows, dtype=np.int64)) \
            if rows else {}
    return out


def admit(probs, ids, k_hat):
    """Greedy admission under F0.5: add while p > current score / 1.25.

    From F = 1.25c / (n + 0.25k), admitting a candidate of probability p raises
    the expected score only while p exceeds c / (n + 0.25k), so the bar rises
    with every id already admitted. c is accumulated as the sum of admitted
    probabilities, which is the expected number correct.
    """
    order = np.argsort(-probs)
    chosen, c, n = [], 0.0, 0
    for idx in order:
        p = probs[idx]
        if n and p <= c / (n + 0.25 * k_hat):
            break
        chosen.append(ids[idx])
        c += p
        n += 1
    return chosen


def select(by_ent, k_global=None):
    """Admit per entity, then keep each target only for its highest scorer.

    by_ent maps s1_row to a list of (probability, encoded target id). With
    k_global unset, k is estimated per entity as the sum of its probabilities,
    the expected number of true matches among its candidates; that needs
    calibrated probabilities. Returns (admitted, assigned), both
    {s1_row: [encoded id]}.
    """
    picks = {}
    for r, lst in by_ent.items():
        probs = np.array([x[0] for x in lst])
        ids = [x[1] for x in lst]
        k_hat = k_global if k_global is not None else max(float(probs.sum()), 1.0)
        picks[r] = admit(probs, ids, k_hat)

    best = {}
    for r, chosen in picks.items():
        lookup = {i: pr for pr, i in by_ent[r]}
        for i in chosen:
            if i not in best or lookup[i] > best[i][1]:
                best[i] = (r, lookup[i])
    assigned = {r: [] for r in picks}
    for i, (r, _) in best.items():
        assigned[r].append(i)
    return picks, assigned


def macro_f05(truth, pred, entities):
    """Mean per-entity F0.5 over the given entities; empty-empty scores 1.0."""
    tot = 0.0
    for r in entities:
        t, q = truth[r], set(pred.get(r, ()))
        if not t and not q:
            tot += 1.0
        elif t and q:
            c = len(t & q)
            tot += 1.25 * c / (len(q) + 0.25 * len(t)) if c else 0.0
    return tot / len(entities)


def fit(n_entities: int, max_df: int, max_group: int, calibrate=True, pool=None, workers=1,
        top_k=None, extra_train=0, lgb_params=None, rounds=300, cv_folds=0, n_models=1,
        drop=(), preranker_entities=0, pairs_out=None, pseudo=None):
    """Train on one entity split and score the held-out split end to end.

    top_k applies prefilter before training. extra_train adds that many
    entities from outside the sample to training only, leaving the held-out
    entities unchanged. lgb_params overrides LightGBM settings. pairs_out saves
    every pair the model scores, with its label, to that .npz path. pseudo adds the
    labelled pairs of a pseudo_labels.py file to training only. Returns a dict with the
    booster, the isotonic calibrator (or None), the global k used when
    uncalibrated, and the held-out scores.
    """
    t0 = time.time()
    import lightgbm as lgb

    _, base_rows, _ = sample_entities(n_entities, C.SEED)
    s1, rows, truth = sample_entities(n_entities, C.SEED, extra_train)
    print(f"entities: {len(rows):,} | mean true matches "
          f"{np.mean([len(v) for v in truth.values()]):.3f}", flush=True)

    cands, s1_text = gather_candidates(rows, s1, max_df, max_group)
    tgt_text = fetch_target_text(cands)
    print(f"candidates gathered ({time.time() - t0:.0f}s), "
          f"{sum(len(v) for p in cands.values() for v in p.values()):,} pairs", flush=True)

    n_raw = sum(len(v) for per in cands.values() for v in per.values())
    preranker = (train_preranker(preranker_entities, set(base_rows.tolist()), max_df, max_group,
                                 pool, workers) if preranker_entities and top_k else None)
    X, keys = build_features_parallel(cands, s1_text, tgt_text, pool, workers, "train", top_k,
                                      preranker)
    del tgt_text
    X = add_group_features(X, keys)
    tid = {s: er_io.parse_source("train", s)["ids"] for s in (2, 3)}
    y = np.array([1 if (src * ENCODE + tid[src][tr]) in truth[r] else 0
                  for r, src, tr in keys], dtype=np.int8)
    print(f"features: {X.shape} of {n_raw:,} raw pairs | positives {y.sum():,} ({y.mean() * 100:.2f}%) "
          f"({time.time() - t0:.0f}s)", flush=True)

    ent = np.array([k[0] for k in keys])
    if pairs_out:
        # in_sample marks the sampled entities the held-out score is taken on;
        # the others were drawn for training only.
        np.savez(pairs_out, s1_row=ent.astype(np.int32), s1_id=s1["ids"][ent],
                 enc_id=np.array([src * ENCODE + int(tid[src][tr]) for _, src, tr in keys],
                                 dtype=np.int64),
                 label=y, in_sample=np.isin(ent, base_rows))
    if pseudo:
        # Pairs of records with no ground truth, labelled from an earlier
        # model's confident scores. Their entity is -1, which is never held
        # out, calibrated on or given a fold, so they only ever train.
        pl = np.load(pseudo)
        if [str(f) for f in pl["features"]] != FEATURES + GROUP_FEATURES:
            raise SystemExit(f"{pseudo}: feature columns differ from this code")
        X = np.vstack([X, pl["X"].astype(np.float32)])
        y = np.concatenate([y, pl["label"].astype(np.int8)])
        ent = np.concatenate([ent, np.full(len(pl["label"]), -1)])
        keys = keys + [(-1, 0, 0)] * len(pl["label"])
        print(f"pseudo-labelled pairs added: {len(pl['label']):,} "
              f"({pl['label'].mean() * 100:.1f}% positive) from {pseudo}", flush=True)
    uniq_ent = np.unique(ent)
    rng = np.random.default_rng(C.SEED)
    # Held-out entities are drawn from every sampled entity, including those
    # with no candidate at all: they score 0 (or 1 if truly unmatched) and
    # must count, or the held-out score flatters the pipeline.
    holdout = set(rng.choice(base_rows, size=len(base_rows) // 4, replace=False).tolist())
    is_val = np.array([e in holdout for e in ent])

    iso, k_global = None, None
    if calibrate:
        # Split the training entities again so calibration never sees the data
        # the trees were fitted on; an uncalibrated score would bias the bar.
        train_ent = np.array(sorted(set(uniq_ent.tolist()) - holdout - {-1}))
        calib_ent = set(rng.choice(train_ent, size=max(len(train_ent) // 5, 1),
                                   replace=False).tolist())
        is_cal = np.array([e in calib_ent for e in ent])
        fit_mask = ~is_val & ~is_cal
    else:
        fit_mask = ~is_val
        k_global = float(np.mean([len(truth[r]) for r in rows.tolist() if r not in holdout]))

    if drop:
        names = FEATURES + GROUP_FEATURES
        X = X[:, [j for j, n in enumerate(names) if n not in set(drop)]]
    params = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63,
              "min_data_in_leaf": 50, "verbose": -1, "seed": C.SEED}
    params.update(lgb_params or {})

    def train_predict(tr, te):
        """(models, mean prediction): one model, or n_models bagged with different seeds."""
        if n_models == 1:
            m = lgb.train(params, lgb.Dataset(X[tr], label=y[tr]), num_boost_round=rounds)
            return [m], m.predict(X[te])
        models = []
        for j in range(n_models):
            pj = dict(params, seed=C.SEED + j, bagging_fraction=0.8, bagging_freq=1,
                      feature_fraction=0.8)
            models.append(lgb.train(pj, lgb.Dataset(X[tr], label=y[tr]), num_boost_round=rounds))
        return models, np.mean([m.predict(X[te]) for m in models], axis=0)

    models, p = train_predict(fit_mask, is_val)
    model = models[-1]
    fold_of = None
    if cv_folds > 1 and not calibrate:
        # Cross-fitting: every sampled entity is scored by a model that never saw
        # it, so the held-out set is the whole sample, not a quarter of it. A
        # single split moves by about +-0.0015 whenever the feature set changes,
        # which is as large as the effects being measured.
        perm = np.random.default_rng(C.SEED + 11).permutation(len(base_rows))
        fold_of = dict(zip(base_rows.tolist(), (perm % cv_folds).tolist()))
        ef = np.array([fold_of.get(e, -1) for e in ent])
        holdout = set(base_rows.tolist())
        is_val = ef >= 0
        p_all = np.zeros(len(ent))
        for k in range(cv_folds):
            tr, te = ef != k, ef == k
            p_all[te] = train_predict(tr, te)[1]
        p = p_all[is_val]
    if calibrate:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(model.predict(X[is_cal]), y[is_cal])
        p = iso.predict(p)
    print(f"model trained ({time.time() - t0:.0f}s)", flush=True)

    val_keys = [k for k, v in zip(keys, is_val) if v]
    by_ent = {}
    for (r, src, tr), prob in zip(val_keys, p):
        by_ent.setdefault(r, []).append((prob, src * ENCODE + tid[src][tr]))
    picks, assigned = select(by_ent, k_global)

    raw, one2one = macro_f05(truth, picks, holdout), macro_f05(truth, assigned, holdout)
    # A flat probability threshold beats greedy admission on uncalibrated
    # scores, so the held-out score is also reported under that rule.
    thr = {t: macro_f05(truth, {r: [i for pr, i in lst if pr >= t] for r, lst in by_ent.items()},
                        holdout) for t in THRESHOLDS}
    imp = sorted(zip([n for n in FEATURES + GROUP_FEATURES if n not in set(drop)],
                     model.feature_importance("gain")),
                 key=lambda x: -x[1])
    raw_cands = {r: {src * ENCODE + int(tid[src][tr]) for src, trs in cands[r].items() for tr in trs}
                 for r in holdout}
    fold_scores = None
    if fold_of is not None:
        fold_scores = []
        for k in range(cv_folds):
            members = {r for r in holdout if fold_of[r] == k}
            fold_scores.append(macro_f05(truth, {r: [i for pr, i in lst if pr >= 0.75]
                                                 for r, lst in by_ent.items() if r in members},
                                         members))
    return {"fold_scores": fold_scores, "val_by_ent": by_ent, "truth": truth, "holdout_set": holdout,
            "raw_cands": raw_cands, "country": s1["country"], "countries": s1["countries"],
            "model": model, "models": models, "preranker": preranker, "iso": iso, "k_global": k_global, "top_k": top_k,
            "entities": len(rows), "holdout": len(holdout), "pairs": int(X.shape[0]),
            "positive_rate": float(y.mean()), "macro_f05_greedy": raw,
            "macro_f05_one_to_one": one2one, "macro_f05_threshold": thr, "feature_gain": imp,
            "seconds": time.time() - t0}


def main(n_entities: int, max_df: int, max_group: int, out_dir, calibrate=True,
         workers=1, top_k=None) -> None:
    """Fit, report the held-out score, and write it to out_dir."""
    from multiprocessing import get_context
    pool = get_context("spawn").Pool(workers) if workers > 1 else None
    try:
        res = fit(n_entities, max_df, max_group, calibrate, pool, workers, top_k)
    finally:
        if pool is not None:
            pool.close()
    print(f"\nheld-out entities: {res['holdout']:,}")
    print(f"  macro F0.5, greedy admission:      {res['macro_f05_greedy']:.5f}")
    print(f"  macro F0.5, + one-to-one assign:   {res['macro_f05_one_to_one']:.5f}")
    for t, v in res["macro_f05_threshold"].items():
        print(f"  macro F0.5, prob >= {t:.2f}:          {v:.5f}")
    print("\n  top features by gain: " + ", ".join(
        f"{n} {g:.0f}" for n, g in res["feature_gain"][:6]))
    print(f"\ntotal {res['seconds']:.0f}s")

    if out_dir:
        pth = Path(out_dir) / "pair_model.json"
        pth.parent.mkdir(parents=True, exist_ok=True)
        pth.write_text(json.dumps(
            {"entities": res["entities"], "max_df": max_df, "max_group": max_group,
             "calibrated": calibrate, "top_k": top_k, "pairs": res["pairs"],
             "positive_rate": res["positive_rate"],
             "macro_f05_greedy": res["macro_f05_greedy"],
             "macro_f05_one_to_one": res["macro_f05_one_to_one"],
             "macro_f05_threshold": {str(t): v for t, v in res["macro_f05_threshold"].items()},
             "feature_gain": {n: float(g) for n, g in res["feature_gain"]}}, indent=2),
            encoding="utf-8")
        print(f"wrote {pth}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--entities", type=int, default=40000)
    p.add_argument("--max-df", type=int, default=300)
    p.add_argument("--max-group", type=int, default=200)
    p.add_argument("--out", default="results/pair_model")
    p.add_argument("--no-calibrate", action="store_true",
                   help="raw booster output with one global k (the pre-calibration rule)")
    p.add_argument("--workers", type=int, default=1, help="processes for feature building")
    p.add_argument("--top-k", type=int, default=40,
                   help="candidates kept per entity by the pre-ranker (0 keeps all)")
    a = p.parse_args()
    main(a.entities, a.max_df, a.max_group, a.out, not a.no_calibrate, a.workers,
         a.top_k or None)
