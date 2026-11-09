"""Byte-level format checks on a matching_results.tsv, beyond the official validator.

Checks: exact header; exactly one row per test Source 1 entity and no others;
two tab-separated fields on every line; empty lists written as "S1-x<TAB>";
strict id grammar; UTF-8 without BOM; LF line endings only; exactly one
trailing newline; no quotes or spaces. Prints each check and the file's size
and SHA256.

Run:
    python src/check_submission.py outputs/submissions/<run>/matching_results.tsv
"""
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

HEADER = b"source1_entity_id\tmatched_entity_ids"
ROW = re.compile(rb"^S1-\d+\t(S[23]-\d+(,S[23]-\d+)*)?$")


def test_s1_ids() -> set:
    """Every entity_id in the test Source 1 file."""
    with open(C.source_file("test", 1), "rb") as f:
        next(f)
        return {line.split(b"\t", 1)[0] for line in f if line.strip()}


def main(path: str) -> int:
    """Run every check; return the number that failed."""
    data = Path(path).read_bytes()
    lines = data.split(b"\n")
    body = lines[1:-1]
    ids = [ln.split(b"\t", 1)[0] for ln in body]
    want = test_s1_ids()
    checks = {
        "UTF-8 decodes": _decodes(data),
        "no BOM": not data.startswith(b"\xef\xbb\xbf"),
        "no CR (LF endings only)": b"\r" not in data,
        "ends with exactly one newline": data.endswith(b"\n") and not data.endswith(b"\n\n"),
        "header exact": lines[0] == HEADER,
        "every line has exactly two tab-separated fields":
            all(ln.count(b"\t") == 1 for ln in lines[:-1]),
        "every data row matches the strict id grammar": all(ROW.match(ln) for ln in body),
        "no quotes": b'"' not in data and b"'" not in data,
        "no spaces": b" " not in data,
        "no duplicate Source 1 rows": len(ids) == len(set(ids)),
        "rows == test Source 1 entities exactly": set(ids) == want and len(ids) == len(want),
        "no duplicate id inside a list": all(
            len(v) == len(set(v)) for v in (ln.split(b"\t")[1].split(b",") for ln in body
                                            if ln.endswith(b"\t") is False)),
    }
    empty = sum(1 for ln in body if ln.endswith(b"\t"))
    failed = 0
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        failed += not ok
    print(f"  rows {len(body):,} (test Source 1: {len(want):,}), empty lists {empty:,}")
    print(f"  size {len(data):,} bytes | sha256 {hashlib.sha256(data).hexdigest()}")
    return failed


def _decodes(data: bytes) -> bool:
    """True when the bytes are valid UTF-8."""
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
