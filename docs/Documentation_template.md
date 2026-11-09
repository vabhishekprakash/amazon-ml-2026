# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [TEAM NAME]
**Team Members:** Abhishek, Akhil
**Submission Date:** 2026-09-25

---

## 1. Executive Summary

Records are blocked on (country, normalized business name) and accepted when the name is
unique within the target source for that country, or when the two addresses share a house
number or postal code. The normalization layer and the F0.5 scorer are the reusable parts;
this first submission is a deterministic rule with no fitted parameters, scored on the full
training set at macro F0.5 **0.46315**.

---

## 2. Methodology

### 2.1 Problem Analysis

Measured on the full training set (2,206,821 Source 1 entities, 7,638,365 labelled matches):

- **Most entities have several matches.** 89.02% have two or more, mean 3.461, max 11. Only
  5.58% are singletons. Because scoring is per entity, recovering one of three true matches
  scores 0.714, so recall matters far more than the "precision-heavy" framing suggests.
- **Matches never cross a country boundary.** All 7,638,365 true pairs share a country label,
  so blocking within country is lossless and cuts the comparison space about threefold.
- **Matched records are entity-unique.** No Source 2 or Source 3 record is claimed by two
  Source 1 entities, which makes a global one-to-one assignment a legitimate constraint.
  About 1.34M records per target source match nothing at all and act as distractors.
- **Names agree exactly far less often than expected.** After normalization, only 28.6-29.2%
  of true pairs share an identical name; sorting tokens raises this to 34.5-35.1%, showing
  word-order transposition accounts for roughly six points on its own.
- **Postal codes are nearly absent.** A 5- or 6-digit code appears on either side of a true
  pair only 8.4-8.9% of the time. House numbers are the usable address signal.
- **Names are noisy in specific ways.** Legal-suffix variants (Corp/Corporation, Pvt Ltd/
  Private Limited), injected punctuation ("-- Name", "<< Name", "[LLC]"), character-level
  typos (a zero inside a word, an l for an i), Devanagari and Tamil script in Sources 2 and 3
  against Latin script in Source 1, and about 3.5% of Source 2 names that are bare domains.

### 2.2 Solution Strategy

**Approach Type:** Blocking + deterministic acceptance rule (first submission)
**Core Innovation:** Address digit runs as a precision filter. A shared house number appears
on 81.89% of true pairs but only 2.95% of non-matching candidates drawn from the same name
group, lifting precision on ambiguous names from 8.32% to 71.57% while keeping 82% of the
true matches.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** (country, normalized name). Normalization lowercases, strips
  accents and punctuation, canonicalizes abbreviations, and recovers domain-style names.
- **Candidate pairs generated:** 1,196,948 of 1,732,544 test entities receive at least one
  candidate; 2,242,376 candidate ids in total.
- **Relationship to the final matches in this submission:** `candidate_pairs.tsv` and
  `matching_results.tsv` have identical contents here, and that is deliberate rather than an
  omission. The acceptance rule is applied *during* candidate generation: a record only
  becomes a candidate if it already satisfies the name and address conditions, so there is no
  later narrowing stage for the file to record. `candidate_pairs.tsv` is defined as the exact
  set the final matcher scores, which for this rule is the match set itself. Later submissions
  separate the two stages, and the file will then be strictly larger than the match set.
- **How true matches were kept:** country equality was verified lossless on all 7.6M training
  pairs before being used as a key. Blocking schemes are ranked by the macro F0.5 a *perfect*
  matcher would achieve from their candidates, rather than by pair recall, because an entity
  with no candidates contributes a hard zero to a macro average. Measured ceilings on a fixed
  20,000-entity sample: exact name 0.5209, token-sorted name 0.5897, house number 0.1741,
  name or token-sorted 0.5913, all three combined 0.6380 at 23.9 candidates per entity.

---

## 4. Matching Model

**Features used:**
- Name features: exact match of the normalized name; uniqueness of that name within the
  target source for the country.
- Address features: intersection of house numbers (digit runs of 1-6 characters, leading
  zeros removed); intersection of postal codes (runs of exactly 5 or 6 digits).
- Other: none in this submission.

**Model type:** Deterministic rule, no fitted parameters.
**Threshold selection method:** None. Because nothing is fitted, the training-set score is an
unbiased estimate rather than an optimistic one.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.46315 on all 2,206,821 training entities.
- Mean F0.5 on the 2,083,574 entities that have at least one true match: 0.44280.
- Singletons: 99,469 of 123,247 correctly predicted empty; 23,778 received a wrong match.
- **Common false positives (wrong merges):** ambiguous names where an address number collides
  by coincidence. Short digit runs such as "1" or "100" are weak evidence, and the acceptance
  rule does not currently weight a house number by how common it is.
- **Common false negatives (missed matches):** the dominant failure. 30.9% of test entities
  receive no candidate at all, because the true match differs by word order, a typo, or a
  change of script. Every one of those entities scores zero regardless of the matcher.

---

## 6. Conclusion

A deterministic name-and-address rule reaches macro F0.5 0.46315 without fitting anything,
and the measurement work shows precisely where the remaining score is: coverage, not
classification. A perfect matcher over the current candidates would score 0.638, so the next
gains come from candidate generation (character n-gram similarity and cross-script matching),
followed by a learned pairwise model whose calibrated probabilities feed a per-entity
selection rule derived from the metric itself.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` is a self-contained, runnable copy of the pipeline.

```
config.py          paths, column names, run settings
requirements.txt   pinned dependencies
src/er_norm.py         name normalization and address feature extraction
src/er_io.py           streaming readers: TSV to compact numpy arrays
src/eda_er.py          data report
src/blocking_ceiling.py  entity-level ceiling of each blocking scheme
src/metrics.py         F0.5 scorer, per entity and macro-averaged
src/baseline_exact.py  matcher; writes both output TSVs
src/log_experiment.py  run log
```

Entry points, in order:

```
python src/eda_er.py --out results/eda
python src/baseline_exact.py --split train --score
python src/baseline_exact.py --split test --run-name sub01_exact_name
```

The third command regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
See `code/business_entity_resolution/README.md` for setup and the dataset path variable.

### B. Additional Results

The scoring function used throughout is the identity form of F0.5. With c correct ids out of
n predicted against k true, F0.5 = 1.25c / (n + 0.25k), verified against the specification's
worked example and against 20,000 randomized cases. Two consequences guided the design: a
candidate is worth adding only when its probability exceeds the current entity score divided
by 1.25, so the bar rises with each id added; and for equal proportional loss, precision is
worth about 3.5 times recall, because a miss shrinks numerator and denominator together while
a false positive inflates only the denominator.
