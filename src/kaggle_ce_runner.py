"""Run src/ce_rerank.py as a Kaggle GPU notebook (accelerator: GPU T4 x2).

The cross-encoder steps (training, continued training, scoring) run on GPU.
This script is the notebook: push it as a Kaggle script kernel with GPU and
internet enabled and these inputs attached:
  - a dataset with this folder's code (config.py and src/);
  - a dataset with the challenge data (the student_resource zip, or the
    extracted student_resource folder); only the source TSVs are unpacked;
  - a dataset with one pair folder (src/ce_pairs.py export or
    src/ce_more_pairs.py output: train_pairs.npz, heldout_pairs.npz,
    test_pairs.npz, shards.json, manifest.json);
  - for a continued or scoring-only run, the output of the earlier run whose
    model/ folder (it holds ce_train.json) is the starting point, attached as a
    dataset or as a notebook source.
Code and data are copied under /tmp; the scorer writes /kaggle/working/ce
(ce_heldout.npy, ce_test.npy, ce_log.json, model/), which is the notebook
output to download. Internet access is used to download the pretrained model
weights from the Hugging Face Hub.

The scorer's arguments are the whole ER_CE_ARGS environment variable when it
is set, else PRESETS[RUN]; a script kernel takes no arguments, so RUN is set in
this file for each notebook. {data_root}, {pairs} and {out} are filled in with
the located folders, and {model} with the attached model/ folder. The presets
hold the settings of the runs behind the submission; e5 ran the same command on
one local RTX 4060 (8 GB), hence its smaller batches.
"""
import glob
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import zipfile

INPUT = "/kaggle/input"
WORK = "/tmp/er/repo"
DATA = "/tmp/er/data"
OUT = "/kaggle/working/ce"
IO = "--data-root {data_root} --pairs {pairs} --out {out} "
PRESETS = {
    # paraphrase-multilingual-mpnet-base-v2 fine-tuned on the export's pairs
    "base": IO + ("--model sentence-transformers/paraphrase-multilingual-mpnet-base-v2 "
                  "--max-len 128 --epochs 1 --batch 64 --lr 2e-5 --infer-batch 512"),
    # an attached model continued for one epoch on the attached pair folder
    "continue": IO + "--init-dir {model} --max-len 128 --epochs 1 --batch 64 --lr 2e-5 --infer-batch 512",
    # held-out and test scoring with an attached model, no training
    "score": IO + "--skip-train --load-dir {model} --max-len 128 --infer-batch 512",
    # multilingual-e5-base fine-tuned on the attached pair folder
    "e5": IO + ("--model intfloat/multilingual-e5-base "
                "--max-len 128 --epochs 1 --batch 32 --lr 2e-5 --infer-batch 256"),
}
RUN = os.environ.get("ER_CE_RUN", "continue")  # preset used when ER_CE_ARGS is unset
NEEDED = {"torch": "torch", "transformers": "transformers", "sklearn": "scikit-learn"}


def find_pairs() -> str:
    """The one attached folder holding *_pairs.npz files."""
    dirs = sorted({os.path.dirname(p) for p in glob.glob(f"{INPUT}/**/*_pairs.npz", recursive=True)})
    if len(dirs) != 1:
        raise SystemExit(f"expected one folder with *_pairs.npz under {INPUT}, found {dirs}")
    return dirs[0]


def find_data() -> str:
    """Folder holding train/ and test/ source TSVs, extracting the challenge zip if needed.

    Only the source files are extracted; no ground-truth file is unpacked.
    """
    zips = [z for z in glob.glob(f"{INPUT}/**/student_resource*.zip", recursive=True)
            if "__MACOSX" not in z]
    if zips:
        with zipfile.ZipFile(zips[0]) as z:
            members = [m for m in z.namelist()
                       if "__MACOSX" not in m and "_source" in m and m.endswith(".tsv")]
            z.extractall(DATA, members)
        base = DATA
    else:
        found = [d for d in glob.glob(f"{INPUT}/**/student_resource", recursive=True)
                 if "__MACOSX" not in d and os.path.isdir(os.path.join(d, "dataset"))]
        if not found:
            raise SystemExit(f"no student_resource zip or folder under {INPUT}")
        base = os.path.dirname(found[0])
    root = os.path.join(base, "student_resource", "dataset")
    if not os.path.isdir(os.path.join(root, "train")):
        raise SystemExit(f"no train/ folder in {root}")
    return root


def find_model():
    """The attached model folder (the one holding ce_train.json), or None."""
    models = sorted({os.path.dirname(p) for p in glob.glob(f"{INPUT}/**/ce_train.json", recursive=True)})
    if len(models) > 1:
        raise SystemExit(f"expected at most one attached model folder, found {models}")
    return models[0] if models else None


def setup() -> dict:
    """Copy the code, install missing packages, show the GPUs; return the located folders."""
    code = os.path.dirname(glob.glob(f"{INPUT}/**/config.py", recursive=True)[0])
    shutil.copytree(code, WORK, dirs_exist_ok=True)
    for arc in glob.glob(f"{WORK}/*.tar") + glob.glob(f"{WORK}/*.zip"):
        shutil.unpack_archive(arc, WORK)  # the Kaggle CLI may upload folders as archives
    if not os.path.exists(os.path.join(WORK, "src", "ce_rerank.py")):
        raise SystemExit("src/ce_rerank.py is not in the attached code")
    missing = [pkg for mod, pkg in NEEDED.items() if importlib.util.find_spec(mod) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi"])
    return {"data_root": find_data(), "pairs": find_pairs(), "out": OUT, "model": find_model()}


def main() -> None:
    """Set up, run the scorer once with the chosen arguments, print its log."""
    where = setup()
    print(json.dumps(where, indent=2), flush=True)
    template = os.environ.get("ER_CE_ARGS") or PRESETS[RUN]
    if "{model}" in template and not where["model"]:
        raise SystemExit("the arguments use {model}, but no model folder with ce_train.json is attached")
    cmd = [sys.executable, "src/ce_rerank.py", *shlex.split(template.format(**where))]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=WORK, check=True)
    log = os.path.join(OUT, "ce_log.json")
    if os.path.exists(log):
        print(open(log, encoding="utf-8").read())


if __name__ == "__main__":
    main()
