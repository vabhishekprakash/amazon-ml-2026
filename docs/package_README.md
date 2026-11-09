# Business Entity Resolution: pipeline

Regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the
challenge data (data -> blocking -> matching -> output). Every command below runs from
this folder (`code/business_entity_resolution/`). Only the provided training and test
files are read; the only download is the pretrained cross-encoder weights from the
Hugging Face Hub.

## The cascade, per country

| Stage | US, India (countries with training labels) | France (absent from training) |
|---|---|---|
| 1. Blocking within country (name, address and number keys) | ~457 raw candidates per entity (240,000-entity training sample) | same keys |
| 2. Learned pre-ranker keeps up to k = 40: the 24 best by its score, plus the 8 best by address and the 8 best by name trigram overlap | 28.61 per entity | 30.68 per entity |
| 3. LightGBM, 5 bagged models | 48 features (44 pairwise, 4 describing the candidate's place among the entity's candidates); scores every pair from stage 2 | 72 features (adds 6 decoy features, 4 sibling-context features and 14 address-number-role features); scores every pair from stage 2: **final model** |
| 4. LightGBM gate: probability >= 0.02 | 4.26 per entity (US 4.11, India 4.38) | not used |
| 5. Cross-encoders + logistic blend | rescore every gated pair: **final model** | not used |
| 6. Decision: each Source 2/3 record goes to its best-scoring Source 1 entity, then kept at blended (or LightGBM) probability >= 0.80 | India also keeps a blank-address record whose normalized name equals the entity's and is unique among India's Source 1 names | |
| 7. Rescue stage: three further key families (digit-repaired name, domain prefix, cross name/address key) propose records outside the stage-1 candidates and outside every assigned record; the top 3 per entity by trigram overlap are scored by a separate LightGBM and kept at >= 0.30 after target argmax | 2.29 extra candidates per entity; +22,933 matches on test | not used |

`candidate_pairs.tsv` lists, for each Source 1 entity, the exact set its country's final
model runs inference on: the gated pairs for US and India, the whole stage-2 set for France
(`decide.py --candidates final-stage`). On test that is 17,613,589 pairs, 10.17 per entity
(6,275,261 gated US/India pairs + 3,379,594 rescue-scored US/India pairs + 7,958,734 France
pairs), against 50,098,191 pairs (28.92 per entity) scored by the main LightGBM. The 0.02 gate predates this file's definition: it was
chosen as the floor below which pairs are not worth a cross-encoder pass, and was not tuned
for the candidate count. On held-out entities the stage-2 set holds 98.06% of all true
pairs, and the gate keeps 99.69% of the true pairs LightGBM scores (28.97 -> 4.22 pairs per
entity). France's k is the `--top-k` of the test pass (40).

## Layout

```
business_entity_resolution/
├── config.py              paths, column names, seed (single source of truth)
├── requirements.txt       pinned dependencies (CPU and GPU parts)
└── src/
    ├── er_norm.py          name/address normalization, blocking-key and number extraction
    ├── er_io.py            streaming TSV readers -> compact cached numpy arrays
    ├── translit.py         word transliteration table (Indic scripts -> Latin) mined from training pairs
    ├── candidates.py       token postings for the blocking keys
    ├── pair_model.py       blocking, learned pre-ranker, pair features, LightGBM fit
    ├── diff_features.py    optional address-number-role and name-alignment columns (ER_DIFF)
    ├── rescue.py           rescue stage: extra key families, scorer fit and apply
    ├── merge_scores.py     replaces one country's rows of a score dump with another dump's
    ├── loss_breakdown.py   held-out cross-fit: score, loss by cause, prediction dumps
    ├── predict_matches.py  fits the matcher, scores a whole split, dumps every pair score
    ├── pseudo_labels.py    self-labels for countries absent from training
    ├── ce_pairs.py         exports the pair lists the cross-encoders train on and rescore
    ├── ce_more_pairs.py    extra cross-encoder training batches from unused training entities
    ├── ce_rerank.py        cross-encoder fine-tuning and scoring (GPU)
    ├── kaggle_ce_runner.py runs ce_rerank.py as a Kaggle GPU notebook
    ├── ce_blend.py         blend fit/apply for one cross-encoder, dump rewriting and checks
    ├── ce_ensemble.py      logistic blend of LightGBM with one or more cross-encoders
    ├── decision_rules.py   score-dump reader and offline decision-rule comparison
    ├── decide.py           applies the decision rule, writes both TSVs
    ├── check_submission.py byte-level format checks on matching_results.tsv
    ├── metrics.py          F0.5 per entity and macro-averaged
    ├── log_experiment.py   appends a row per run to experiments.csv
    └── eda_er.py, blocking_ceiling.py, baseline_exact.py, dense_eval.py,
        full_density_eval.py, stage2.py   analysis tools behind the design choices
```

## Setup

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the challenge data at `./data/student_resource/` or point `ER_DATA_DIR` at the
folder that contains `student_resource/`:

```
student_resource/dataset/train/train_source{1,2,3}.tsv
student_resource/dataset/train/train_ground_truth.tsv
student_resource/dataset/test/test_source{1,2,3}.tsv
student_resource/utils/validate_submission.py     (organizers' validator)
```

Everything the pipeline writes goes under `outputs/` (caches, models, score dumps,
submissions) and `results/` (metrics and blend models), both created on first use; test
passes also append a row to `experiments.csv`.

## Hardware and runtimes

- **CPU steps**: one 10-core machine with 16 GB RAM (Python 3.14), `--workers 9`. Run the
  heavy steps one at a time; each fits in 16 GB alone.
- **GPU steps** (`ce_rerank.py`): Kaggle notebooks with two NVIDIA T4 GPUs (16 GB each).
  Training uses one GPU; scoring splits every batch across both. One run
  (multilingual-e5-base) used a single RTX 4060 laptop GPU (8 GB) with smaller batches.

| Step | Measured time |
|---|---|
| Held-out cross-fit (`loss_breakdown.py`, 240,000 entities, 4 folds x 5 models) | 41-52 min |
| Test pass (`predict_matches.py`, fit + 1,732,544 test entities) | 2.4-2.5 h |
| One 400,000-entity cross-encoder batch (`ce_more_pairs.py sample`, 8 workers) | 24-26 min |
| Cross-encoder training, one epoch on ~2.9M pairs (mpnet, one T4) | 3.7 h |
| Cross-encoder scoring, 7.36M test pairs (two T4) | 61-69 min |
| multilingual-e5-base: training / test scoring (RTX 4060) | 4.7 h / 1.5 h |
| Blend fit / apply (`ce_ensemble.py`) | under 1 min each |

The transliteration table, self-labels, pair export, decision and validation were not
timed separately; each takes minutes.

## Reproduce

Artifact names below are the ones the submission used; the names only need to agree
between commands.

### 1. Transliteration table (before anything else)

Normalization applies this table, and the parsed-array caches store normalized keys, so it
must exist before any other step builds a cache.

```bash
python src/translit.py                               # -> outputs/cache/translit.json
python src/eda_er.py --out results/eda               # optional data report
```

### 2. LightGBM, first round (feeds the self-labels and the cross-encoder pairs)

```bash
# Held-out cross-fit: 60,000 sampled training entities, each scored by a model that never
# saw it (4 entity folds), plus 180,000 training-only entities; learned pre-ranker trained
# on 30,000 other entities. Dumps the held-out predictions and every scored training pair.
python src/loss_breakdown.py --entities 60000 --extra-train 180000 --preranker 30000 \
    --models 5 --cv 4 --workers 9 \
    --dump outputs/cache/cv4m5_addridf.pkl --dump-pairs outputs/cache/pairs_addridf.npz \
    --out results/loss_breakdown/cv4m5_addridf.json

# Test pass with isotonic calibration; its confident scores become the self-labels.
python src/predict_matches.py --split test --entities 60000 --extra-train 180000 \
    --preranker 30000 --models 5 --top-k 40 --shard 40000 --workers 9 \
    --dump-scores outputs/scores/test_v8 --run-name sub08_addridf
```

### 3. Self-labels for the country absent from training

```bash
# 100,000 France entities rebuilt with the full features; score >= 0.9 and the record's
# best claimant -> positive, <= 0.05 -> negative, the rest left out.
python src/pseudo_labels.py --scores outputs/scores/test_v8 --model outputs/models/sub08_addridf \
    --entities 100000 --out outputs/cache/pseudo_v8.npz
```

### 4. LightGBM, final (127 leaves, self-labels added to training)

```bash
python src/loss_breakdown.py --entities 60000 --extra-train 180000 --preranker 30000 \
    --models 5 --cv 4 --workers 9 --lgb '{"num_leaves": 127, "min_data_in_leaf": 100}' \
    --pseudo outputs/cache/pseudo_v8.npz --dump outputs/cache/cv4m5_g127_pseudo.pkl \
    --out results/loss_breakdown/cv4m5_g127_pseudo.json

# Raw (uncalibrated) scores for every test pair; the blend was fitted on raw scores.
python src/predict_matches.py --split test --no-calibrate --entities 60000 --extra-train 180000 \
    --preranker 30000 --models 5 --top-k 40 --shard 40000 --workers 9 \
    --lgb '{"num_leaves": 127, "min_data_in_leaf": 100}' --pseudo outputs/cache/pseudo_v8.npz \
    --dump-scores outputs/scores/test_v9 --run-name sub09_g127_pseudo
```

This writes the boosters and pre-ranker to `outputs/models/sub09_g127_pseudo/`, every
scored test pair to `outputs/scores/test_v9/`, and `results/sub09_g127_pseudo/metrics.json`.

### 5. Cross-encoder pair lists

```bash
# Held-out pairs with LightGBM >= 0.02 (the gate); training pairs of the 180,000
# training-only entities (all positives plus up to 4 random negatives per positive, and no
# pair whose target is a held-out target); the gated test pairs.
python src/ce_pairs.py export --fit-pairs outputs/cache/pairs_addridf.npz \
    --heldout-dump outputs/cache/cv4m5_addridf.pkl --test-scores outputs/scores/test_v9 \
    --test-metrics results/sub09_g127_pseudo/metrics.json --out outputs/ce/export

# Three more batches of 400,000 unused training entities each (A, B, C), and A+B merged.
python src/ce_more_pairs.py sample --entities 400000 \
    --heldout-dump outputs/cache/cv4m5_addridf.pkl --exclude outputs/cache/pairs_addridf.npz \
    --model-dir outputs/models/sub09_g127_pseudo --out outputs/ce/pairs_A
python src/ce_more_pairs.py sample --entities 400000 \
    --heldout-dump outputs/cache/cv4m5_addridf.pkl \
    --exclude outputs/cache/pairs_addridf.npz,outputs/ce/pairs_A/train_pairs.npz \
    --model-dir outputs/models/sub09_g127_pseudo --seed-offset 606 --out outputs/ce/pairs_B
python src/ce_more_pairs.py sample --entities 400000 \
    --heldout-dump outputs/cache/cv4m5_addridf.pkl \
    --exclude outputs/cache/pairs_addridf.npz,outputs/ce/pairs_A/train_pairs.npz,outputs/ce/pairs_B/train_pairs.npz \
    --model-dir outputs/models/sub09_g127_pseudo --seed-offset 707 --out outputs/ce/pairs_C
python src/ce_more_pairs.py merge --parts outputs/ce/pairs_A,outputs/ce/pairs_B --out outputs/ce/pairs_AB
```

Each pair folder holds `train_pairs.npz`, the export's `heldout_pairs.npz`, `test_pairs.npz`
and `shards.json`, and a manifest. `ce_rerank.py` refuses to train on a folder whose
training pairs share a Source 1 entity or a target record with the held-out pairs.

### 6. Cross-encoder training and scoring (GPU)

Each run writes `ce_heldout.npy` and `ce_test.npy` (one probability per exported held-out
and test pair), `ce_log.json` (settings, timings, versions, held-out AUC) and `model/`.

```bash
DATA=data/student_resource/dataset
MPNET="--max-len 128 --epochs 1 --batch 64 --lr 2e-5 --infer-batch 512"

# Base: paraphrase-multilingual-mpnet-base-v2 fine-tuned on the export's pairs.
python src/ce_rerank.py --data-root $DATA --pairs outputs/ce/export --out outputs/ce/mpnet \
    --model sentence-transformers/paraphrase-multilingual-mpnet-base-v2 $MPNET

# The base model continued for one epoch on each further batch.
python src/ce_rerank.py --data-root $DATA --pairs outputs/ce/pairs_A --out outputs/ce/mpnet_A \
    --init-dir outputs/ce/mpnet/model $MPNET
python src/ce_rerank.py --data-root $DATA --pairs outputs/ce/pairs_B --out outputs/ce/mpnet_B \
    --init-dir outputs/ce/mpnet/model $MPNET
python src/ce_rerank.py --data-root $DATA --pairs outputs/ce/pairs_C --out outputs/ce/mpnet_C \
    --init-dir outputs/ce/mpnet/model $MPNET
python src/ce_rerank.py --data-root $DATA --pairs outputs/ce/pairs_AB --out outputs/ce/mpnet_AB \
    --init-dir outputs/ce/mpnet/model $MPNET

# Sequential continuations: A then C, A+B then C.
python src/ce_rerank.py --data-root $DATA --pairs outputs/ce/pairs_C --out outputs/ce/mpnet_AC \
    --init-dir outputs/ce/mpnet_A/model $MPNET
python src/ce_rerank.py --data-root $DATA --pairs outputs/ce/pairs_C --out outputs/ce/mpnet_ABC \
    --init-dir outputs/ce/mpnet_AB/model $MPNET

# multilingual-e5-base fine-tuned on batch A (8 GB GPU).
python src/ce_rerank.py --data-root $DATA --pairs outputs/ce/pairs_A --out outputs/ce/e5_A \
    --model intfloat/multilingual-e5-base --max-len 128 --epochs 1 --batch 32 --lr 2e-5 --infer-batch 256
```

**On Kaggle** (how these runs were made): push `src/kaggle_ce_runner.py` as a script
kernel with GPU T4 x2 and internet on; attach this folder (code), the challenge data and
one pair folder as datasets, plus, for a continued run, the notebook whose `model/` is the
starting point. Set `RUN` in the script to `base`, `continue`, `score` or `e5`; its presets
are the settings above. The notebook's `/kaggle/working/ce` folder is the `--out` folder:
download it to `outputs/ce/<name>/`. Scoring can also run as its own notebook with
`RUN = "score"` and the trained notebook attached.

### 7. Blend and apply (US, India)

```bash
python src/ce_ensemble.py fit --heldout-dump outputs/cache/cv4m5_g127_pseudo.pkl \
    --ce outputs/ce/mpnet_AC --out results/ce_blend/blend_g127_mpnetAC.json
python src/ce_ensemble.py apply --model results/ce_blend/blend_g127_mpnetAC.json \
    --scores outputs/scores/test_v9 --out outputs/scores/test_v9_sub13_mpnetAC
```

`fit` prints the out-of-fold held-out macro F0.5 (weighted to the test country mix) at
thresholds 0.70-0.85 and stores the best. `apply` writes a copy of the test dump in which
the gated US/India pairs carry the blended probability and are flagged as the final stage's
input; France pairs keep their LightGBM probability. Final coefficients: intercept -0.026,
LightGBM logit 0.584, cross-encoder logit 0.573, product 0.026.

### 8. France, final LightGBM (address-number-role columns)

```bash
# Self-labels rebuilt with the 72-column features, then a France-only fit and test pass.
ER_DIFF=address python src/pseudo_labels.py --scores outputs/scores/test_v8 \
    --model outputs/models/sub08_addridf --entities 100000 --hi 0.9 --lo 0.05 --top-k 40 \
    --out outputs/cache/pseudo_v12addr.npz
ER_DIFF=address python src/predict_matches.py --split test --countries France --entities 60000 \
    --extra-train 180000 --models 5 --preranker 30000 --top-k 40 \
    --lgb '{"num_leaves":127,"min_data_in_leaf":100}' --pseudo outputs/cache/pseudo_v12addr.npz \
    --workers 5 --no-calibrate --dump-scores outputs/scores/test_v12_sub18fr --run-name sub18fr
python src/merge_scores.py --base outputs/scores/test_v9_sub13_mpnetAC \
    --replace outputs/scores/test_v12_sub18fr --countries France --out outputs/scores/test_v12m_sub18fr
```

`ER_DIFF` is set on these two commands only; every other command runs without it.

### 9. Decide and write both files

```bash
python src/decide.py --scores outputs/scores/test_v12m_sub18fr --split test --rule argmax_threshold \
    --threshold 0.80 --unseen-threshold 0.80 --candidates final-stage --blank-name-rule India \
    --run-name sub18fr_final
```

### 10. Rescue stage (US, India)

```bash
# Scorer fit on 40,000 training entities disjoint from the held-out entities and their targets.
python src/rescue.py fit --out outputs/models/rescue --countries US India \
    --exclude outputs/cache/cv4m5_g127_pseudo.pkl
python src/rescue.py evaluate --model outputs/models/rescue --dump outputs/cache/cv4m5_g127_pseudo.pkl
python src/rescue.py apply --model outputs/models/rescue --base outputs/submissions/sub18fr_final \
    --out outputs/submissions/sub18_final --countries US India
```

`apply` keeps every match and candidate of the base files and adds the accepted rescue
matches and the rescue-scored candidates. `outputs/submissions/sub18_final/` holds the two
files in `output/`.

### 11. Validate

```bash
python data/student_resource/utils/validate_submission.py \
    --matching outputs/submissions/sub18_final/matching_results.tsv \
    --candidate outputs/submissions/sub18_final/candidate_pairs.tsv \
    --test-dir data/student_resource/dataset/test --check-ids
python src/check_submission.py outputs/submissions/sub18_final/matching_results.tsv
```

## Models and licenses

| Model | License | Parameters | Role |
|---|---|---|---|
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | Apache-2.0 | 278,044,417 as a cross-encoder (counted from the saved weights) | cross-encoder |
| intfloat/multilingual-e5-base | MIT | ~278M (same XLM-RoBERTa-base architecture) | cross-encoder |
| LightGBM (library: MIT) | trained here | 5 x 300 trees; pre-ranker 150 trees | pair model, pre-ranker |

An earlier submission used sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
(Apache-2.0, 117,654,145 parameters as a cross-encoder); the final pipeline does not. Every
model is far below the 8B limit. The cross-encoders are fine-tuned only on pairs of the
provided training data; LightGBM also trains on the France self-labels derived from the
provided test records.

## Reproducibility notes

- Code revisions: the US/India LightGBM (steps 2 and 4) ran before the decoy, sibling-context
  and address-number-role columns were added; its 48 features are the first 44 pairwise
  columns of `pair_model.FEATURES` plus the 4 group columns. Rerunning step 4 with this
  folder's code builds the 50-column model instead, whose scores differ slightly from the
  ones blended in the submission. Steps 8-10 ran with this folder's code.
- The rescue step was reproduced on a second machine (Windows, Python 3.12) with byte-identical
  output files. Copying `outputs/models/rescue/` on Windows with line-ending conversion changes
  the files; copy them byte for byte.

- Sampling, folds and LightGBM seeds derive from `config.SEED` (42) with fixed offsets.
- Cross-encoder training uses fp16 on GPU, so a rerun can differ in the last digits of its
  probabilities; the blend and threshold absorb such differences.
- `ce_rerank.py` records in `ce_train.json` the digest of the held-out pairs its training
  pairs were checked against; `ce_blend.py` and `ce_ensemble.py` refuse a cross-encoder
  whose digest differs from the held-out pairs being blended.
