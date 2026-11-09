"""Run a queue of held-out experiments inside a Kaggle CPU notebook.

Expects two attached datasets: er-data (the challenge zip, or its extracted
student_resource folder) and er-code (src/, config.py, requirements). Copies
the code to a writable folder, unpacks the data, builds the transliteration
table, then runs each experiment with src/loss_breakdown.py and writes one
JSON per experiment to /kaggle/working/.

Edit EXPERIMENTS (or set the ER_EXPERIMENTS environment variable to a
';'-separated list of "name|flags") to choose what runs.
"""
import glob
import os
import shutil
import subprocess
import sys
import zipfile

EXPERIMENTS = [
    ("maxgroup300", "--max-group 300"),
    ("maxdf200", "--max-df 200"),
]

WORK = "/kaggle/working/repo"
DATA = "/kaggle/working/data"


def setup() -> dict:
    """Copy code, unpack data, install extras; return the environment to run with."""
    code = os.path.dirname(glob.glob("/kaggle/input/**/config.py", recursive=True)[0])
    shutil.copytree(code, WORK, dirs_exist_ok=True)
    for arc in glob.glob(f"{WORK}/*.tar") + glob.glob(f"{WORK}/*.zip"):
        shutil.unpack_archive(arc, WORK)  # the CLI may upload folders as archives
    zips = [z for z in glob.glob("/kaggle/input/**/student_resource*.zip", recursive=True)
            if "__MACOSX" not in z]
    if zips:
        os.makedirs(DATA, exist_ok=True)
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall(DATA)
        root = DATA
    else:
        found = [d for d in glob.glob("/kaggle/input/**/student_resource", recursive=True)
                 if "__MACOSX" not in d and os.path.isdir(os.path.join(d, "dataset"))]
        root = os.path.dirname(found[0])
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rapidfuzz==3.14.6",
                    "lightgbm==4.7.0"], check=True)
    env = dict(os.environ, ER_DATA_DIR=root)
    subprocess.run([sys.executable, "src/translit.py"], cwd=WORK, env=env, check=True)
    return env


def main() -> None:
    """Set up once, then run every queued experiment."""
    env = setup()
    queue = EXPERIMENTS
    if os.environ.get("ER_EXPERIMENTS"):
        queue = [tuple(x.split("|", 1)) for x in os.environ["ER_EXPERIMENTS"].split(";")]
    for name, flags in queue:
        out = f"/kaggle/working/{name}.json"
        cmd = [sys.executable, "src/loss_breakdown.py", "--workers", "4", "--out", out,
               *flags.split()]
        print(">>", name, " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=WORK, env=env)


if __name__ == "__main__":
    main()
