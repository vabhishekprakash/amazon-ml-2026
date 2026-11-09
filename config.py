"""Central config: paths, column names and run settings for the pipeline.

Set ER_DATA_DIR to point the pipeline at a dataset outside this folder; it
defaults to ./data next to this file.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("ER_DATA_DIR", ROOT / "data"))
OUT_DIR = ROOT / "outputs"
PRED_DIR = OUT_DIR / "preds"
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figs"
EXPERIMENT_LOG = ROOT / "experiments.csv"

# ---- EDIT THESE ON DAY 1 ----
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
SAMPLE_SUB_FILE = DATA_DIR / "sample_submission.csv"
IMAGE_DIR = DATA_DIR / "images"          # if the problem ships images

ID_COL = "sample_id"                      # id column name in train/test
TARGET_COL = "price"                      # target column name in train
TEXT_COLS = ["title", "description"]      # text columns to concatenate
IMAGE_COL = None                          # column holding image filename/url, or None

TASK = "regression"                       # "regression" | "classification"
METRIC = "smape"                          # key from src/metrics.py REGISTRY
LOG_TARGET = True                         # train on log1p(target)? (good for prices)
N_SPLITS = 5
SEED = 42

# ---- Entity resolution dataset ----
# Three source files per split plus a ground-truth file for train. Tab separated.
# SAMPLE_ROOT mirrors DATASET_ROOT's layout with a small slice for quick runs.
DATASET_ROOT = DATA_DIR / "student_resource" / "dataset"
SAMPLE_ROOT = DATA_DIR / "sample" / "dataset"
RESULTS_DIR = ROOT / "results"

TSV_SEP = "\t"
SOURCES = (1, 2, 3)
SOURCE_PREFIX = {1: "S1-", 2: "S2-", 3: "S3-"}

ENTITY_ID_COL = "entity_id"
NAME_COL = "business_name"
ADDRESS_COL = "business_address"
COUNTRY_COL = "country"

GT_ID_COL = "source1_entity_id"
GT_MATCH_COL = "matched_entity_ids"
CANDIDATE_COL = "candidate_entity_ids"

MATCHING_FILE = "matching_results.tsv"
CANDIDATE_FILE = "candidate_pairs.tsv"


def data_root(sample: bool = False):
    """Return the dataset root: the small slice when sample is set, else the full data."""
    return SAMPLE_ROOT if sample else DATASET_ROOT


def source_file(split: str, source: int, sample: bool = False):
    """Path to one source file, e.g. train/train_source2.tsv."""
    return data_root(sample) / split / f"{split}_source{source}.tsv"


def ground_truth_file(split: str = "train", sample: bool = False):
    """Path to a split's ground-truth file (train only; test has no labels)."""
    return data_root(sample) / split / f"{split}_ground_truth.tsv"


for d in (DATA_DIR, PRED_DIR, MODEL_DIR, FIG_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
