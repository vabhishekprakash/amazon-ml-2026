"""Append one row per experiment to experiments.csv — the team's shared memory.
Usage from code:  log_experiment(name="tfidf_lgbm", cv=12.34, notes="char ngrams help")
"""
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

FIELDS = ["timestamp", "name", "cv_mean", "cv_std", "lb_score", "pred_file", "who", "notes"]


def log_experiment(name, cv=None, cv_std=None, lb=None, pred_file="", who="", notes=""):
    new = not C.EXPERIMENT_LOG.exists()
    with open(C.EXPERIMENT_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "name": name,
            "cv_mean": f"{cv:.5f}" if cv is not None else "",
            "cv_std": f"{cv_std:.5f}" if cv_std is not None else "",
            "lb_score": lb or "",
            "pred_file": str(pred_file),
            "who": who,
            "notes": notes,
        })
    print(f"logged '{name}' to {C.EXPERIMENT_LOG}")


if __name__ == "__main__":
    # CLI: python src/log_experiment.py "name" "notes..." [lb_score]
    args = sys.argv[1:]
    log_experiment(args[0] if args else "manual",
                   notes=args[1] if len(args) > 1 else "",
                   lb=args[2] if len(args) > 2 else None)
