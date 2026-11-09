"""Assemble the submission zip in the layout the organizers specify.

    <team>_submission.zip
    |-- output/matching_results.tsv                        results/<run>/matching_results.tsv
    |-- output/candidate_pairs.tsv                         outputs/submissions/<run>/candidate_pairs.tsv
    |-- code/business_entity_resolution/config.py
    |-- code/business_entity_resolution/src/<PACKAGE_MODULES>
    |-- code/business_entity_resolution/README.md          docs/package_README.md
    |-- code/business_entity_resolution/requirements.txt   docs/package_requirements.txt
    `-- Documentation_template.md                          docs/Documentation_filled.md

Only the files listed above are copied, so data, caches, models, logs and working notes
never enter the archive. Before zipping, the build stops if:
  - results/<run>/matching_results.tsv differs from the sha256 in results/<run>/sha256.txt
    (when that file exists) or from outputs/submissions/<run>/matching_results.tsv, the
    sibling of the candidate file, so both TSVs come from the same run;
  - results/<run>/decide.json does not record --candidates final-stage, i.e. the candidate
    file is not each country's final-stage inference set;
  - a packaged module imports a src module that is not packaged, or does not compile;
  - the methodology document still holds a TODO.
--draft turns the candidate-mode and TODO checks into warnings, for dry runs.
Prints the archive size, SHA256 and members, and writes them next to the zip as JSON.

Run:
    python scripts/build_package.py --run-name final --team-name "Abhishek's Team"
    python scripts/build_package.py --run-name sub12f_mpnetA --team-name "Abhishek's Team" --draft
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

# Source files of the pipeline and of the analyses behind its design, copied into src/.
PACKAGE_MODULES = [
    "er_norm.py", "er_io.py", "translit.py", "candidates.py", "pair_model.py",
    "loss_breakdown.py", "predict_matches.py", "pseudo_labels.py", "ce_pairs.py",
    "ce_more_pairs.py", "ce_rerank.py", "kaggle_ce_runner.py", "ce_blend.py", "ce_ensemble.py",
    "decision_rules.py", "decide.py", "check_submission.py", "metrics.py", "log_experiment.py",
    "eda_er.py", "blocking_ceiling.py", "baseline_exact.py", "dense_eval.py",
    "full_density_eval.py", "stage2.py", "rescue.py", "merge_scores.py", "diff_features.py",
]

SRC_IMPORT = re.compile(r"^\s*from src(?:\.(\w+))? import ([^\n]+)", re.M)


def sha256(path: Path) -> str:
    """Hex SHA256 of a file, read in chunks so large archives stay cheap."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def warn_or_stop(message: str, draft: bool) -> None:
    """Stop the build on a failed release check, or only print it for a draft."""
    if not draft:
        raise SystemExit(message)
    print(f"draft, not a release build: {message}")


def check_outputs(run_name: str, draft: bool) -> tuple:
    """(matching file, candidate file) of a run, refused unless both come from the same staged run."""
    staged = C.RESULTS_DIR / run_name / C.MATCHING_FILE
    sub = C.OUT_DIR / "submissions" / run_name
    for path in (staged, sub / C.MATCHING_FILE, sub / C.CANDIDATE_FILE):
        if not path.exists():
            raise SystemExit(f"missing {path}")
    digest = sha256(staged)
    record = C.RESULTS_DIR / run_name / "sha256.txt"
    if record.exists() and record.read_text(encoding="utf-8").split()[0] != digest:
        raise SystemExit(f"{staged} does not match the sha256 recorded in {record}")
    if sha256(sub / C.MATCHING_FILE) != digest:
        raise SystemExit(f"{staged} differs from {sub / C.MATCHING_FILE}: the candidate file "
                         "may belong to another run")
    decide = C.RESULTS_DIR / run_name / "decide.json"
    mode = json.loads(decide.read_text(encoding="utf-8")).get("candidates") if decide.exists() else None
    if mode != "final-stage":
        warn_or_stop(f"{sub / C.CANDIDATE_FILE} was not written with decide.py --candidates "
                     f"final-stage ({decide.name}: {mode})", draft)
    print(f"matching_results.tsv sha256 {digest}")
    return staged, sub / C.CANDIDATE_FILE


def check_modules(src_dir: Path) -> None:
    """Refuse a module that does not compile or imports a src module left out of the package."""
    names = {m[:-3] for m in PACKAGE_MODULES}
    missing = set()
    for m in PACKAGE_MODULES:
        text = (src_dir / m).read_text(encoding="utf-8")
        compile(text, m, "exec")
        for sub, rest in SRC_IMPORT.findall(text):
            wanted = {sub} if sub else {w.split(" as ")[0].strip(" ()") for w in rest.split(",")}
            missing |= {f"{m}: {w}" for w in wanted if w and w not in names}
    if missing:
        raise SystemExit(f"packaged modules import unpackaged ones: {sorted(missing)}")


def stage(run_name: str, staging: Path, doc: Path, code_only: bool, draft: bool) -> None:
    """Copy the outputs, code and methodology document into the required tree."""
    if staging.exists():
        shutil.rmtree(staging)
    code = staging / "code" / "business_entity_resolution"
    (code / "src").mkdir(parents=True)
    if not code_only:
        out = staging / "output"
        out.mkdir()
        matching, candidates = check_outputs(run_name, draft)
        shutil.copy2(matching, out / C.MATCHING_FILE)
        shutil.copy2(candidates, out / C.CANDIDATE_FILE)
    for module in PACKAGE_MODULES:
        shutil.copy2(C.ROOT / "src" / module, code / "src" / module)
    check_modules(code / "src")
    shutil.copy2(C.ROOT / "config.py", code / "config.py")
    shutil.copy2(C.ROOT / "docs" / "package_README.md", code / "README.md")
    shutil.copy2(C.ROOT / "docs" / "package_requirements.txt", code / "requirements.txt")
    shutil.copy2(doc, staging / "Documentation_template.md")


def main(a) -> None:
    """Stage, check, zip and report."""
    doc = Path(a.doc)
    if "TODO" in doc.read_text(encoding="utf-8"):
        warn_or_stop(f"{doc} still holds a TODO", a.draft)
    out_dir = Path(a.out_dir)
    staging = out_dir / a.run_name
    stage(a.run_name, staging, doc, a.code_only, a.draft)

    zip_path = out_dir / f"{a.team_name}_{'code' if a.code_only else 'submission'}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(staging).as_posix())

    digest, size = sha256(zip_path), zip_path.stat().st_size
    with zipfile.ZipFile(zip_path) as z:
        members = [(i.filename, i.file_size) for i in sorted(z.infolist(), key=lambda i: i.filename)]
    print(f"\nzip:    {zip_path}")
    print(f"size:   {size / (1 << 20):.2f} MB ({size:,} bytes)")
    print(f"sha256: {digest}")
    print(f"members: {len(members)}")
    for name, n in members:
        print(f"  {n:>12,}  {name}")
    zip_path.with_suffix(".json").write_text(json.dumps(
        {"zip": str(zip_path), "run": a.run_name, "bytes": size, "sha256": digest,
         "members": dict(members)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True, help="run whose results/ and outputs/submissions/ files ship")
    p.add_argument("--team-name", required=True, help="exact team name for the zip filename")
    p.add_argument("--doc", default=str(C.ROOT / "docs" / "Documentation_filled.md"),
                   help="filled methodology document, shipped as Documentation_template.md")
    p.add_argument("--out-dir", default=str(C.OUT_DIR / "package"),
                   help="folder for the staging tree, the zip and its JSON record")
    p.add_argument("--draft", action="store_true",
                   help="dry run: a TODO in the document or a candidate file not written with "
                        "--candidates final-stage only warns")
    p.add_argument("--code-only", action="store_true", help="omit output/")
    main(p.parse_args())
