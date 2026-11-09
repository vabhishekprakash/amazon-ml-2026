# Business entity resolution for Amazon ML Challenge 2026

This repository links business listings from two secondary sources (Source 2 and Source 3)
to the Source 1 entity each one describes, across the US, India and France. It writes
`matching_results.tsv` and `candidate_pairs.tsv` in the format the organizers specify.

The final submission scored **0.97902 on the public leaderboard**. On held-out US and India
entities it reaches a macro F0.5 of 0.98499 (US 0.98648, India 0.98378), measured out of
fold. France has no labels, so the public score is the only measurement for it.

## The problem

Every Source 1 entity is scored separately: F0.5 between the predicted and true matched ids,
then averaged over all entities. Precision counts more than recall, and an entity with no
true match scores 0 as soon as it gets a single prediction.

The training set has 2,206,821 Source 1 entities from the US and India, 10,320,219 Source 2
and Source 3 records, and 7,638,365 labelled matches. Most entities (89%) have two or more
matches, so finding only one of three true matches scores 0.714 and recall still matters.
Matches never cross a country, which lets every stage run within one country at a time.
France makes up 15% of test entities and does not appear in training at all.

## How it works

Each country runs through the same cascade:

1. Blocking on ten families of name, address and number keys proposes candidate records.
2. A learned pre-ranker keeps up to 40 candidates per entity. On held-out entities that set
   still contains 98.06% of all true pairs.
3. Five bagged LightGBM models score every remaining pair on 48 features (72 for France).
4. For the US and India, pairs below a 0.02 LightGBM probability are dropped. The gate keeps
   99.69% of the true pairs LightGBM scores.
5. A multilingual cross-encoder fine-tuned on training pairs rescores the survivors, and a
   logistic blend combines it with LightGBM.
6. Each Source 2/3 record goes to the Source 1 entity that scores it highest, and the match
   is kept at a probability of 0.80 or more.
7. A rescue stage searches three more key families (digit-repaired name, domain prefix,
   cross name/address key) for records the first pass missed. On test it adds 22,933
   matches.

France has no training labels. Its final model is a 72-feature LightGBM that adds decoy,
sibling-context and address-number-role features and trains with self-labels taken from
confident test predictions. Steps 4, 5 and 7 apply to the US and India only.

## Results

Selected submissions, in order. Held-out scores are US/India macro F0.5 (sub04 on a single
split, later rows on the cross-fit).

| Submission | Change | Held-out | Public |
|---|---|---|---|
| sub04 | 44 features, pre-ranker, one-to-one assignment | 0.96787 | 0.955044 |
| sub06 | new keys and features, 5 models, 180,000 more training entities | 0.9747 | 0.961932 |
| sub07 | learned pre-ranker | 0.97604 | 0.963182 |
| sub08ce | address token weights, MiniLM cross-encoder blend | 0.98102 | 0.974133 |
| sub12f | 127 leaves, France self-labels, mpnet trained on 400,000 more entities | 0.98324 | 0.977362 |
| sub17 | India blank-address rule, France decoy and sibling features, rescue stage | 0.98499 | 0.97897 |
| sub18 | final package, regenerated from code | 0.98499 | 0.97902 |

The full breakdown of what moved the metric, and what did not, is in
[`docs/Documentation_filled.md`](docs/Documentation_filled.md).

## Repository layout

```
config.py          paths, column names and seed, the single source of truth
src/               pipeline modules: normalization, blocking, pair model, cross-encoder, rescue, decision, metrics
scripts/           build_package.py (submission zip), kaggle_heldout.py, make_synthetic_data.py
tests/             unit tests for normalization, tokenization and the metric
docs/              problem statement, methodology write-up, package README
results/           per-run metrics and the SHA256 of each staged submission
experiments.csv    experiment log with held-out and leaderboard scores
aws/               GPU instance helper
data/, outputs/    local only (gitignored): datasets, caches, models, predictions
```

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests
python src/translit.py      # build the transliteration table before anything else
```

The complete reproduction, with every command and its runtime, is in
[`docs/package_README.md`](docs/package_README.md). The CPU stages ran on a 10-core machine
with 16 GB of RAM, where the full test pass takes about two and a half hours. Cross-encoder
training and scoring ran on Kaggle notebooks with two T4 GPUs.

To build the submission archive from a finished run:

```bash
python scripts/build_package.py --run-name final --team-name "<team name>"
```

The build checks that the packaged results match their recorded SHA256 and that every
packaged module compiles before it writes the zip.


## Team

Built by a team of two for the Amazon ML Challenge 2026. Abhishek designed
and built the pipeline: data analysis, blocking, features, models, the
cross-encoder, evaluation and the submission package. Akhil ran training
jobs on his GPU and MacBook.
