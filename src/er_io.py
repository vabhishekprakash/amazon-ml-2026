"""Streaming readers that turn the source TSVs into compact numpy arrays.

The full dataset is 12.5M train records and 11.7M test records. Holding the raw
strings would need far more memory than this machine has, so each file is read
once, line by line, and reduced to fixed-width arrays:

    ids       int64  numeric part of entity_id (the S1-/S2-/S3- prefix is implied)
    country   int8   index into the file's own country label list
    name_key  int64  hash of the normalized name
    sort_key  int64  hash of the token-sorted normalized name

Results are cached under outputs/cache/ (gitignored) so the normalization pass
is paid once. Raw text for a chosen subset of rows is fetched on a second pass
by fetch_fields(), which keeps peak memory proportional to the subset.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.er_norm import hash64, house_numbers, normalize_name, postcodes, token_sort

CACHE_DIR = C.OUT_DIR / "cache"
ID_PREFIX_LEN = 3  # "S1-", "S2-", "S3-"
CACHE_VERSION = 4  # bump when the stored columns or the normalization change

# Address numbers are stored as small fixed-width slots so that agreement is a
# numpy operation on candidate pairs instead of a re-read of the raw text.
# Re-reading would cost one full file scan per batch, which does not survive
# tens of millions of candidate pairs.
MAX_HOUSE = 4
MAX_PC = 2
EMPTY = -1


def _count_rows(path: Path) -> int:
    """Number of data rows, excluding the header."""
    with open(path, "rb") as f:
        return max(sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b"")) - 1, 0)


def parse_source(split: str, source: int, limit=None, rebuild=False) -> dict:
    """Read one source TSV into arrays, caching the result.

    limit reads only the first N data rows (a fast smoke path; never cached).
    Returns a dict with ids, country, name_key, sort_key, countries and
    n_blank_name.
    """
    path = C.source_file(split, source)
    cache = CACHE_DIR / f"{split}_source{source}.v{CACHE_VERSION}.npz"
    if limit is None and cache.exists() and not rebuild:
        blob = np.load(cache, allow_pickle=False)
        return {"ids": blob["ids"], "country": blob["country"],
                "name_key": blob["name_key"], "sort_key": blob["sort_key"],
                "house": blob["house"], "pcode": blob["pcode"],
                "countries": [str(x) for x in blob["countries"]],
                "n_blank_name": int(blob["n_blank_name"][0]),
                "n_malformed": int(blob["n_malformed"][0])}

    n = limit if limit is not None else _count_rows(path)
    ids = np.empty(n, dtype=np.int64)
    country = np.empty(n, dtype=np.int8)
    name_key = np.empty(n, dtype=np.int64)
    sort_key = np.empty(n, dtype=np.int64)
    house = np.full((n, MAX_HOUSE), EMPTY, dtype=np.int32)
    pcode = np.full((n, MAX_PC), EMPTY, dtype=np.int32)
    labels, blank, malformed, i = {}, 0, 0, 0

    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            if i >= n:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                malformed += 1
                continue
            raw_id, name, address, ctry = parts
            ids[i] = int(raw_id[ID_PREFIX_LEN:])
            code = labels.get(ctry)
            if code is None:
                code = labels[ctry] = len(labels)
            country[i] = code
            norm = normalize_name(name)
            if not norm:
                blank += 1
            name_key[i] = hash64(norm)
            sort_key[i] = hash64(token_sort(norm))
            for slot, value in enumerate(sorted(house_numbers(address))[:MAX_HOUSE]):
                house[i, slot] = value
            for slot, value in enumerate(sorted(postcodes(address))[:MAX_PC]):
                pcode[i, slot] = np.int32(hash64(value) & 0x7FFFFFFF)
            i += 1

    if limit is None and i != n:
        raise RuntimeError(f"{path}: parsed {i} rows but counted {n}; "
                           f"{malformed} malformed row(s) were skipped")

    out = {"ids": ids[:i], "country": country[:i], "name_key": name_key[:i],
           "sort_key": sort_key[:i], "house": house[:i], "pcode": pcode[:i],
           "countries": [lab for lab, _ in sorted(labels.items(), key=lambda kv: kv[1])],
           "n_blank_name": blank, "n_malformed": malformed}
    if limit is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache, ids=out["ids"], country=out["country"],
                 name_key=out["name_key"], sort_key=out["sort_key"],
                 house=out["house"], pcode=out["pcode"],
                 countries=np.array(out["countries"], dtype=object).astype(str),
                 n_blank_name=np.array([blank]), n_malformed=np.array([malformed]))
    return out


def slots_intersect(left: np.ndarray, right: np.ndarray, chunk=2_000_000) -> np.ndarray:
    """True where two fixed-slot integer sets share at least one real value.

    Empty slots hold EMPTY and never count as agreement. Comparison is done in
    chunks because the broadcast is rows x slots x slots booleans.
    """
    out = np.zeros(len(left), dtype=bool)
    for start in range(0, len(left), chunk):
        stop = min(start + chunk, len(left))
        a = left[start:stop][:, :, None]
        b = right[start:stop][:, None, :]
        out[start:stop] = ((a == b) & (a != EMPTY)).any(axis=(1, 2))
    return out


def parse_ground_truth(limit=None) -> dict:
    """Read train_ground_truth.tsv into CSR-style arrays.

    Returns s1 (int64 entity numbers), offsets (int64, len s1+1), match_id
    (int64 numeric ids, concatenated) and match_src (int8, 2 or 3).
    """
    path = C.ground_truth_file("train")
    s1, offsets, match_id, match_src = [], [0], [], []
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for row, line in enumerate(f):
            if limit is not None and row >= limit:
                break
            left, _tab, rest = line.rstrip("\n").partition("\t")
            s1.append(int(left[ID_PREFIX_LEN:]))
            if rest:
                for token in rest.split(","):
                    if token:
                        match_id.append(int(token[ID_PREFIX_LEN:]))
                        match_src.append(int(token[1]))
            offsets.append(len(match_id))
    return {"s1": np.array(s1, dtype=np.int64),
            "offsets": np.array(offsets, dtype=np.int64),
            "match_id": np.array(match_id, dtype=np.int64),
            "match_src": np.array(match_src, dtype=np.int8)}


def fetch_fields(split: str, source: int, rows) -> dict:
    """Return {row_index: (name, address)} for the given row indices.

    Second pass over the file so only the requested subset is held in memory.
    """
    wanted = np.sort(np.asarray(rows, dtype=np.int64))
    out, cursor = {}, 0
    with open(C.source_file(split, source), encoding="utf-8") as f:
        next(f, None)
        for i, line in enumerate(f):
            if cursor >= len(wanted):
                break
            if i == wanted[cursor]:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 4:
                    out[i] = (parts[1], parts[2])
                cursor += 1
                while cursor < len(wanted) and wanted[cursor] == wanted[cursor - 1]:
                    cursor += 1
    return out


def country_keys(parsed: dict) -> np.ndarray:
    """Globally comparable int64 country key per row.

    The country column is an index into that file's own label list, and the
    lists are built in first-seen order, so source 1's code 0 need not be
    source 2's code 0. Hashing the label gives a key that compares correctly
    across files and stays valid for a country seen only in test.
    """
    labels = np.array([hash64(c) for c in parsed["countries"]], dtype=np.int64)
    return labels[parsed["country"]]


def combine_keys(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mix two int64 hash columns into one blocking key.

    Wraparound on overflow is intended: this is modular hashing, not counting.
    """
    return a * np.int64(31) + b


def group_sizes(keys: np.ndarray) -> np.ndarray:
    """Size of each key's group, aligned to the input order."""
    uniq, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return counts[inverse]


def index_of(haystack: np.ndarray, needles: np.ndarray) -> np.ndarray:
    """Position of each needle in haystack, or -1 when absent.

    Used to turn entity ids into row indices without building a Python dict
    over millions of keys.
    """
    order = np.argsort(haystack, kind="stable")
    pos = np.searchsorted(haystack[order], needles)
    pos = np.clip(pos, 0, len(order) - 1)
    cand = order[pos]
    ok = haystack[cand] == needles
    return np.where(ok, cand, -1)
