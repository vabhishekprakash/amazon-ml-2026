"""Word-level transliteration table for non-Latin business names.

Source 2 and 3 render some Indian business names in Devanagari, Bengali,
Tamil, Telugu, Kannada or Gujarati, word for word ("शिव फुडस पराइवेट लिमिटेड" for
"Shiva Foods Private Limited"). Those records share no characters with the
Latin Source 1 name, so no name key or name similarity can reach them.

The labelled training pairs are a parallel corpus. For every true pair whose
target name is non-Latin and has the same number of words as the Source 1
name, the words are aligned by position and counted. A non-Latin word maps to
the Latin word it was aligned with most often, provided that happened at least
min_count times and in at least min_share of its alignments. Word-order swaps
in the noise only add a minority of wrong alignments, which the vote absorbs.

Only the provided training data is used. The table is written to
outputs/cache/translit.json and er_norm.normalize_name applies it.

Run:
    python src/translit.py
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import er_io
from src.er_norm import TRANSLIT_PATH, normalize_name

ENCODE = 2_000_000_000


def is_ascii(tok: str) -> bool:
    """True for a token with no non-ASCII character."""
    return all(ord(c) < 128 for c in tok)


def load_names(split: str, source: int, keep_nonlatin_only: bool) -> dict:
    """{numeric id: normalized name (no transliteration)} for one source file."""
    out = {}
    with open(C.source_file(split, source), encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            name = normalize_name(parts[1], translit=False)
            if keep_nonlatin_only and is_ascii(name):
                continue
            out[int(parts[0][er_io.ID_PREFIX_LEN:])] = name
    return out


def build(min_count: int, min_share: float) -> dict:
    """Mine the table from the training ground truth."""
    s1 = load_names("train", 1, keep_nonlatin_only=False)
    tgt = {s: load_names("train", s, keep_nonlatin_only=True) for s in (2, 3)}
    print(f"non-Latin target names: {len(tgt[2]):,} (S2), {len(tgt[3]):,} (S3)", flush=True)
    gt = er_io.parse_ground_truth()
    counts = collections.defaultdict(collections.Counter)
    pairs = aligned = 0
    off, mid, msrc = gt["offsets"], gt["match_id"], gt["match_src"]
    for g, sid in enumerate(gt["s1"].tolist()):
        a = s1.get(sid, "").split()
        for j in range(off[g], off[g + 1]):
            b = tgt[int(msrc[j])].get(int(mid[j]))
            if b is None:
                continue
            pairs += 1
            b = b.split()
            if len(a) != len(b):
                continue
            aligned += 1
            for x, y in zip(b, a):
                if not is_ascii(x) and is_ascii(y):
                    counts[x][y] += 1
    table = {}
    for x, c in counts.items():
        y, n = c.most_common(1)[0]
        if n >= min_count and n / sum(c.values()) >= min_share:
            table[x] = y
    print(f"true pairs with a non-Latin target name: {pairs:,}; same word count: {aligned:,}; "
          f"table: {len(table):,} words of {len(counts):,} seen", flush=True)
    return table


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min-count", type=int, default=3)
    p.add_argument("--min-share", type=float, default=0.5)
    a = p.parse_args()
    table = build(a.min_count, a.min_share)
    TRANSLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRANSLIT_PATH.write_text(json.dumps(table, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(f"wrote {TRANSLIT_PATH}")
