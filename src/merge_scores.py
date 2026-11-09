"""Replace the pairs of some countries in a score dump with the pairs of another dump.

A country's matcher can change without rescoring the rest of the split: the base
dump (for example a cross-encoder blend for US and India, written by ce_blend.py
apply) keeps every other country's rows as they are, and the countries named here
take their rows from the replacement dump (a predict_matches.py --countries run).
Blocking is within country, so no target record can appear under two countries and
the one-to-one step in decide.py stays exact.

The replacement rows carry stage2 = False and the output records unseen_keep as the
base did, so decide.py --candidates final-stage lists every replacement pair for
countries that have no training labels, exactly as it would for their own dump.

Run:
    python src/merge_scores.py --base outputs/scores/test_v9_sub13_mpnetAC \\
        --replace outputs/scores/test_v10_france --countries France \\
        --out outputs/scores/test_v10_merged
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import er_io


def main(a) -> None:
    """Write the merged dump: base shards without the named countries, then the replacement shards."""
    s1 = er_io.parse_source(a.split, 1)
    swap = np.isin(np.array(s1["countries"])[s1["country"]], a.countries)
    out = Path(a.out)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    kept = replaced = 0
    for f in sorted(Path(a.base).glob("shard_*.npz")):
        b = np.load(f)
        m = ~swap[b["s1_row"]]
        np.savez(tmp / f.name, s1_row=b["s1_row"][m], enc_id=b["enc_id"][m], prob=b["prob"][m],
                 stage2=(b["stage2"][m] if "stage2" in b.files else np.zeros(int(m.sum()), bool)))
        kept += int(m.sum())
    for f in sorted(Path(a.replace).glob("shard_*.npz")):
        b = np.load(f)
        m = swap[b["s1_row"]]
        if not m.any():
            continue
        np.savez(tmp / f"shard_r{f.stem.split('_')[-1]}.npz", s1_row=b["s1_row"][m],
                 enc_id=b["enc_id"][m], prob=b["prob"][m], stage2=np.zeros(int(m.sum()), bool))
        replaced += int(m.sum())
    meta_path = Path(a.base) / "blend_apply.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.update({"merged_from": {"base": str(a.base), "replace": str(a.replace),
                                 "countries": a.countries},
                 "pairs_kept": kept, "pairs_replaced": replaced})
    (tmp / "blend_apply.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    print(f"kept {kept:,} base pairs, took {replaced:,} pairs for {', '.join(a.countries)} -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="dump whose other countries are kept")
    p.add_argument("--replace", required=True, help="dump supplying the named countries")
    p.add_argument("--countries", nargs="+", required=True)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--out", required=True)
    main(p.parse_args())
