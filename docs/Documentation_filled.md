# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Abhishek's Team  
**Team Members:** Abhishek, Akhil  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

Final submission (sub18): public leaderboard 0.97902; held-out macro F0.5 on US/India
0.98499 (US 0.98648, India 0.98378; out of fold, weighted to the test country mix). The
held-out figure covers only US and India, the countries with labels. The public score also
includes France (15.0% of test entities), which has no labels to measure on.

Candidates come from blocking within each country on ten families of name, address and
number keys. A learned pre-ranker keeps up to 40 per Source 1 entity (98.06% of all true
pairs on held-out entities), and a 5-model LightGBM scores them. The two countries with training labels (US, India) get two further stages.
Pairs that clear a 0.02 LightGBM gate (it keeps 99.69% of the true pairs LightGBM
scores) are rescored by a multilingual cross-encoder
fine-tuned on training pairs and combined with LightGBM in a logistic blend. A rescue stage
then adds matches from three further key families outside the main candidates, without
changing any decision already made. France has no training labels, so LightGBM stays its
final model. It is trained with self-labels from its own confident test predictions and
with features that describe how two records differ (legal-form and added-word decoys,
sibling context, the role of each address number). These target the failure measured on
an unseen country: when India is scored by a US-only model, 97.2% of the confident wrong
matches are records that belong to no Source 1 entity, mostly near-duplicates with the
legal form swapped or a word added, which no threshold removes. Finally, every Source 2/3 record goes
to the Source 1 entity that scores it highest, and the match is kept if its probability is
0.80 or more.

---

## 2. Methodology

### 2.1 Problem Analysis

Scoring: for each Source 1 entity, F0.5 between the predicted and the true matched ids
(precision weighted above recall), macro-averaged over all Source 1 entities. An entity with
no true match scores 1 for an empty list and 0 for any prediction (identity in Appendix B).

Measured on the full training set: 2,206,821 Source 1 entities (US 1,323,633, India
883,188), 5,034,616 Source 2 and 5,285,603 Source 3 records, 7,638,365 labelled matches
(48.36% in Source 2).

- **Most entities have several matches.** 89.02% have two or more (mean 3.461, median 3,
  maximum 11); 5.58% have none. Scoring is per entity, so finding one of three true matches
  scores only 0.714. Recall therefore matters more than the precision-weighted metric
  suggests, while a single wrong id on an entity without matches still scores 0.
- **Matches never cross a country.** 100% of true pairs share the country label, so all
  blocking is within country. The label is used as a partition, never as a feature value.
- **Each Source 2/3 record belongs to at most one Source 1 entity** (no exception in 7.6M
  matches). This allows a one-to-one step: a record goes to the entity that scores it
  highest.
- **Exact names are rare.** Only 28.56% (Source 2) and 29.23% (Source 3) of true pairs
  share the normalized name; sorting the tokens raises this to 34.68% and 35.09%. Exact-name
  blocking leaves 37-39% of entities without a single candidate.
- **Names are ambiguous.** 40.01% of Source 1 records share their (country, name) with
  another record. Such groups reach 253 records in Source 1 and 2,413 in Source 3 (chains,
  branches, generic names), so name agreement alone cannot decide a match.
- **Postal codes are nearly absent** (on either side of 8.39-8.86% of true pairs). House
  numbers agree on 81.55-82.75% of true pairs and are the usable address signal.
- **Noise seen in the data**: legal-suffix and abbreviation variants, injected punctuation,
  character typos (a zero inside a word), word-order swaps, names replaced by a one-off
  nonsense word, whole names written word by word in Devanagari, Bengali, Tamil, Telugu,
  Kannada or Gujarati against Latin Source 1 names, domain-style names
  ("supritaindustries"), truncated or perturbed house numbers ("414" vs "14", "2251" vs
  "2256"), and blanked addresses.
- **Test** has 1,732,544 Source 1 entities: US 663,106, India 809,986, and France 259,452
  (15.0%), a country with no training record.

### 2.2 Solution Strategy

**Approach Type:** Hybrid cascade: blocking, learned pre-ranker, gradient-boosted pair
classifier, cross-encoder second stage with a logistic blend, and a one-to-one threshold
decision.  
**Core Innovation:** Every signal the pair model sees is computed from the split's own
files (word rarity, name and address multiplicity, per-country address token weights), so a
feature means the same thing in a country the model never saw. Candidate generation uses
address number-word and word-pair keys that survive reordering, truncation and a change of
script. A cross-encoder second stage reads both records jointly. It improved when fed
400,000 training entities that no other stage had used (0.98222 to 0.98324), and digest
checks ensure that no held-out entity or held-out target record is ever trained on.

The pipeline, in order:

1. **Normalize** names and addresses: lowercase, accents and punctuation removed, a small
   generic abbreviation table that also covers French street words, domain-style names
   recovered, and Indic-script words transliterated with a table mined from training pairs.
2. **Block** within country on ten key families (Section 3), then keep up to 40 candidates
   per entity with a learned pre-ranker.
3. **Score** every kept pair with LightGBM (48 features for US/India, 72 for France; Section 4).
4. **Rescore** (US, India) every pair with LightGBM probability >= 0.02 with the
   cross-encoders, and blend.
5. **Decide**: each Source 2/3 record goes to its highest-scoring Source 1 entity, and the
   entity keeps it if the (blended) probability is >= 0.80.
6. **Add** (India) a gated candidate kept by target-argmax but below 0.80 when its record
   has a blank address and its normalized name equals the entity's name, and that name is
   unique among the country's Source 1 names. Then (US, India) add the rescue-stage matches
   (Section 3) without changing any earlier decision.

### 2.3 Validation Design

- **Held-out sample.** 60,000 training Source 1 entities drawn with a fixed seed. Entities
  that receive no candidate stay in it (they score 0, or 1 when they truly have no match);
  leaving them out would flatter every blocking change.
- **4-fold entity cross-fit.** Each of the 60,000 is scored by a model fitted without it.
  180,000 further training-only entities join every fold's training set, and the
  pre-ranker is trained on 30,000 other entities. We moved to this after a noise check:
  adding one pure-noise feature to the model on the earlier single 15,000-entity split moved
  the score to 0.97289 and 0.97210 (from 0.97364), about as much as the feature gains being
  measured. Under cross-fit the fold standard deviation is 0.0003-0.001.
- **Cross-encoder leakage guards.** Cross-encoder training pairs exclude every held-out
  entity and every target record of a held-out pair (on test every target record is new).
  The training script records a digest of the held-out pairs it was checked against, and the
  blend refuses a cross-encoder whose digest differs.
- **Blend evaluation.** The logistic blend is cross-fitted over 5 entity folds on the
  held-out gated pairs, so every held-out pair gets an out-of-fold probability. Held-out
  entities are re-weighted to the test country mix (US 45%, India 55% of the labelled-country
  test entities, against about 60/40 in the held-out sample).
- **Leave-one-country-out.** A model trained on US only and scored on India stands in for
  France. It measured what an unseen country costs and whether self-training, the
  cross-encoder and the threshold carry over. For the threshold it was run twice: on the
  60,000-entity sample, and on all 883,188 India training entities scored like a test pass,
  where every same-name entity competes in the one-to-one step (Section 2.4).
- **Leaderboard transfer.** LightGBM changes moved the public score about one for one with
  held-out (sub06: +0.0068 held-out, measured across the move from a single split to
  cross-fit, and +0.0069 public; sub07: +0.0013, +0.00125). Cross-encoder transfer varied.
  sub08ce (+0.011 public against about +0.005 forecast) and sub12f (+0.00222 held-out,
  +0.00323 public) also changed France; sub13, which changed only the US/India
  cross-encoder, moved held-out by +0.00044 and public by +0.0003. Decisions were made on
  held-out numbers; the public leaderboard was used only to confirm transfer.
- **Selection on the held-out sample.** The same 60,000 held-out entities also chose the
  thresholds and which changes to adopt, so their absolute scores carry some selection
  optimism. The public leaderboard is the only independent check.

### 2.4 The country absent from training (France)

- **Nothing keys on a country name.** Blocking partitions by the label, frequency features
  are computed per label from the split's own files, and the only routing rule is whether a
  country has training labels.
- **Generic normalization** includes French street words (rue/r, boulevard/bd,
  impasse/imp, ...). It did not move the public score measurably (0.955 against 0.955044),
  so France's gap is not about abbreviations.
- **Self-training.** A first LightGBM scored every test pair. 100,000 France entities were
  sampled and their features rebuilt. A pair scoring >= 0.9 whose entity is the record's
  best claimant became a positive, a pair scoring <= 0.05 a negative, and the middle band was
  left out. The 2,906,935 labelled pairs (10.5% positive) join LightGBM training only (never
  held out, never in a fold). On India as the unseen country, one self-training round moved
  the score from 0.92584 to 0.92750. In the full recipe with self-labels, US and India did
  not lose (0.97696 to 0.97749 together with 127-leaf trees, every fold up).
- **France keeps LightGBM.** A cross-encoder trained on US pairs only reached 0.874 on India
  alone, against 0.928 for the US-only LightGBM; blended onto India it scored 0.9636 against
  0.97153 for LightGBM. The blend is therefore applied only to countries with training
  labels.
- **Threshold.** On the 60,000-entity sample the unseen country peaked near 0.95 (0.93229
  against 0.92759 at 0.75). Scoring all 883,188 India entities like a test pass reversed
  this: 0.75 0.93130, 0.80 0.93129, 0.95 0.92475. Same-name siblings are mostly absent from a
  sample, so their records remain false matches that only a high bar removes; on the full
  split they compete in the one-to-one step. France uses 0.80, the same as US and India.
- **Where the unseen-country loss comes from.** Scoring all 883,188 India entities with a
  US-only model gives precision 0.969 and recall 0.881 at the best threshold, against 0.999
  and 0.953 for a country with training labels. 97.2% of its confident wrong matches
  (p >= 0.9, 61 per 1,000 entities) are records that belong to no Source 1 entity:
  near-duplicates of a real business with the legal form swapped ("Private Limited" to
  "Limited" or "LLP") or a real word added ("Exports", "Ventures", "Textiles"), usually at
  the same address with a unit number added. The model learns what these decoys look like
  from each country's own training pairs, and that knowledge does not carry over to a new
  country's legal forms and vocabulary. No threshold removes them, because they score
  >= 0.9. Rules applied after scoring do not separate them either: rejecting legal-form
  conflicts cost 0.005 (2.75% of true copies also switch form), rejecting added words cost
  up to 0.026 (91% of added-word pairs are true), and the best relative rule gained 0.0007.
- **Features that describe how two records differ.** Since the unseen-country loss is
  mostly precision against near-duplicates, France's final LightGBM adds the features below.
  All are computed from the two records' text and within-country counts, and each was
  measured on India as the unseen country (LightGBM trained on US with self-training,
  threshold 0.80):
  - six decoy features: legal-form conflict, legal form on one side only, identifying words
    added or dropped, a bracket tag added, filler words added (0.92925 -> 0.93428);
  - four sibling-context features: each decoy signal relative to its minimum over the
    entity's candidates (0.93428 -> 0.93481);
  - fourteen address-number-role features: agreement or conflict of the probable postal
    code, house number and unit number (missing is not conflict), a letter or "bis" suffix
    on the same house number, compound numbers, the house-number gap, shared rare-token
    weight over each side's weight, unmatched rare weight on each side, rare tokens matched
    within a spelling edit, and the longest shared run of street words (0.93481 -> 0.93783,
    all four folds up). On the leaderboard this France-only change moved the public score
    from 0.97897 to 0.97902.

  Name-token alignment features (typo, abbreviation, reordering, replaced or added words)
  lost 0.00083 on the same check and were not used.
- **Measured directly on the leaderboard.** Scoring France with the US/India cross-encoder
  blend instead of LightGBM changed 14.7% of France entities and moved the public score
  from 0.977664 to 0.97755, about -0.0008 on France, so France stays on LightGBM. A 0.5B
  language model (Qwen2.5-0.5B, Apache-2.0) fine-tuned on US pairs reached 0.841 on India
  as the unseen country (0.926 blended) against 0.928 for LightGBM.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** all within country. A Source 2/3 record is a candidate for a
  Source 1 record when they share any of:
  1. the whole normalized name;
  2. the token-sorted name;
  3. a house number (digit run of 1-6 characters, leading zeros dropped);
  4. any name token;
  5. any address token;
  6. a number-word key: each address number with the next two address words of 4+
     characters ("319 High Street, Roxboro" gives "319 high" and "319 roxboro"; "36 Rue de
     Ségur" gives "36 segur");
  7. the core name (legal forms, honorifics and corporate filler removed, tokens sorted);
  8. an unordered pair of adjacent address words of 4+ characters (survives truncated
     numbers and reordering);
  9. an unordered pair of core-name words (survives an edit to any one word);
  10. for one-word, domain-style target names, the concatenation of an ordered pair of
      Source 1 name words ("supritaindustries").

  Names are transliterated before keying with a 1,308-word table mined from aligned
  training pairs, which covers 93.4% of the Indic words in test. A token found in more than
  300 records of a source is not indexed (max_df 300), and a key whose group holds more than
  200 records is skipped (max_group 200).
- **Candidate pairs generated:** about 457 raw pairs per entity (109,778,575 for the
  240,000-entity training sample).
  - The learned pre-ranker is a small LightGBM on 7 cheap features (address and name
    trigram Jaccard, address and name token Jaccard, blank target address, shared house
    number, raw candidate count), trained on 30,000 other training entities. It keeps up to
    40 per entity: the 24 best by its score, plus the 8 best by address trigram overlap and
    the 8 best by name trigram overlap. These "solo slots" stop a copy whose name was
    replaced, or whose address was blanked, from being crowded out.
  - On test, LightGBM scores 50,098,191 pairs (28.92 per entity: US 27.09, India 29.85,
    France 30.68).
  - `candidate_pairs.tsv` holds each country's input to its final stage, 17,613,589 pairs in
    total (10.17 per entity). For US and India these are the 6,275,261 pairs with LightGBM
    >= 0.02 that the cross-encoder blend rescores (4.11 and 4.38 per entity), plus the
    3,379,594 pairs the rescue stage scores. For France they are all 7,958,734 pairs
    LightGBM scores, since LightGBM is France's final model.
  - For US and India the cross-encoder blend is the final matching model and the 0.02
    LightGBM gate is the last filtering stage, so the file holds the blend's input. The gate
    was set as the floor below which a pair is not worth a cross-encoder pass, not to shrink
    this file, and it keeps 99.69% of the true pairs LightGBM scores.
  - France's set is the pre-ranker's top k = 40. Smaller k was checked with India as the
    unseen country (US-only training): top-20 cost 0.0048 there and top-12 cost 0.028,
    mostly true matches that the pre-ranker ranks below the cut, so France keeps 40.
- **Rescue stage (US, India).** Three further key families (the name with digits read back
  as letters, the domain-name prefix, and cross name/address word keys, at most 20 records
  per key) propose records that are neither among the entity's stage-1 candidates nor
  assigned to any entity. The 3 best per entity by combined trigram overlap (2.5 per held-out
  entity, 2.29 per test entity) are scored by a separate LightGBM on the main pair-model
  features (58 columns, including the decoy and sibling-context features), trained on 40,000 training
  entities disjoint from the held-out entities and their targets. Its threshold (0.30) was
  frozen on its own inner split at precision >= 0.85 with at least 30 accepted pairs. After
  each record goes to its highest-scoring entity (target-argmax), the accepted pairs are
  added to the matches, and nothing already decided changes. Held-out: 0.98402 -> 0.98499, every fold up, +817 true and +77 false matches.
  Mixing the same keys into the stage-1 top-40 instead evicted true matches and lost score.
- **How you ensured true matches were not lost:**
  - Country equality was verified lossless on all 7.6M training pairs before it was used as
    a partition.
  - Keys were added by measuring, on held-out entities, the macro F0.5 lost to true matches
    that no key reached. Number-word keys raised the share of reachable true pairs from 80.3%
    to 91.9% (25,000-entity sample). The transliteration table cut India's never-blocked loss
    from 0.0189 to 0.0093; the core-name key cut the total from 0.0251 to 0.0173, address
    word pairs from 0.0171 to 0.0105, core-name word pairs from 0.0101 to 0.0064, and
    concatenated names from 0.0064 to 0.0057 (points of macro F0.5). Appendix B gives the
    full loss by cause and country.
  - In the final LightGBM cross-fit, true matches never blocked cost 0.00573 and those cut by
    the pre-ranker 0.00104. On held-out entities the pre-ranker set holds 98.06% of all true
    pairs, and the 0.02 gate keeps 99.69% of the true pairs LightGBM scores (28.97 -> 4.22
    pairs per entity).
  - The candidate count was set by held-out F0.5, not by recall alone: keeping 70 or 80
    candidates scored lower than 50 (more near-duplicates compete), solo slots cut the
    pre-ranker loss from 0.0067 to 0.0045, and the learned pre-ranker replaced trigram-sum
    ranking (0.97473 to 0.97604).

---

## 4. Matching Model

**Features used:** 48 per pair for the US/India model: 44 pairwise and 4 describing the
pair's place among the entity's candidates. France's final model adds 24 (Section 2.4): six
decoy features, four sibling-context features and fourteen address-number-role features.
All come from the raw text and from counts over the split's own files; no label enters a
feature.

- Name features (19):
  - token Jaccard and containment; exact normalized equality; token-sorted equality; length
    ratio; character-trigram Jaccard;
  - rapidfuzz token_set_ratio and partial_ratio (typo-tolerant token matching);
  - count of tokens unique to each side; best fuzzy ratio between the unmatched words of the
    two sides (typo against substitution);
  - non-Latin script flag for each side (a name in another script is not disagreement);
  - word rarity over the split's Source 2/3 names: rarest target-name word, most common
    word unique to each side (a true copy keeps the name up to typos or swaps in a one-off
    word; a different business at the same site brings a real, common word);
  - core-name multiplicity: Source 1 entities sharing the Source 1 core name, Source 1
    entities sharing the target's core name, Source 2/3 records sharing the target's core
    name (a unique name is strong evidence, a chain name is not).
- Address features (23):
  - token Jaccard and containment; exact equality; trigram Jaccard; rapidfuzz
    token_set_ratio and partial_ratio; blank-address flag for each side;
  - numbers: shared house number; shared postal code; Jaccard, containment and
    both-present of the two number sets; truncation flag ("414" vs "14") and same-length
    sibling flag ("2251" vs "2256"); signed offset of the closest numbers when none is shared
    (decoys sit at small positive offsets, while noise on true copies goes both ways); count
    of shared number-word keys;
  - per-country token weights (log10 of records over records with the token, within the
    country, over the split's three sources, capped at 5): weighted containment, weight of
    the rarest shared token, weight of the rarest Source 1 token the target lacks (a dropped
    city weighs about 1, a different street 2-5);
  - address multiplicity: Source 1 entities at the target's exact address, other Source 1
    entities co-located with the Source 1 record, smallest Source 1 multiplicity among the
    shared number-word keys.
- Other (6): the entity's raw candidate count; a rank slot that is constant 0 in this
  version; for name and address trigram Jaccard, the gap to the entity's best candidate and
  the rank among its pre-ranked candidates (being the closest of about 30 candidates says
  more than an absolute similarity).

**Model type:**

- *LightGBM pair classifier.* Binary objective, learning rate 0.1, 300 rounds, 127 leaves,
  at least 100 samples per leaf; the mean of 5 bagged models (seeds 42-46, bagging and
  feature fractions 0.8). The test model trains on the candidate pairs of 225,000 training
  entities (the 240,000 sampled minus a 15,000-entity held-out quarter) plus the 2,906,935
  France self-labelled pairs; its raw mean probability is used.
- *Cross-encoders (US, India).* Sequence-pair classifiers with one output logit over
  "name | address" of the Source 1 record and of the candidate, at most 128 tokens; binary
  cross-entropy, AdamW at 2e-5 with 5% warmup and linear decay, fp16, batch 64,
  length-bucketed batches, one epoch per stage. sentence-transformers/
  paraphrase-multilingual-mpnet-base-v2 (Apache-2.0, 278M parameters) is first fine-tuned
  on the 2,945,359 pairs of the 180,000 LightGBM training-only entities (all positives, up
  to 4 random negatives per positive, no pair with a held-out target). It is then continued
  for one epoch on each of three batches (A, B, C) of 400,000 further unused training entities (about 2.92M pairs
  each: all positives, every negative LightGBM scores >= 0.02, one random easy negative per
  positive). intfloat/multilingual-e5-base (MIT, same architecture) is fine-tuned from its
  pretrained weights on batch A (batch size 32 on an 8 GB GPU). The final blend uses one
  cross-encoder: mpnet continued on batch A and then on batch C (held-out AUC 0.99669).
  Other cross-encoder choices (adding e5 or further continuations as a second member, or
  another continuation order) moved held-out F0.5 by at most +0.00017 (a variant at its best
  threshold, 0.75, against the final blend at 0.80), below the +0.0003 required for adoption.
- *Logistic blend.* Inputs per gated pair: logit of the LightGBM probability and, per
  cross-encoder, its logit and the product of the two logits (logits clipped at 1e-4). The
  shipped coefficients are refitted on all 256,812 gated held-out pairs (58,391 entities).
  For sub12f: intercept 0.014, LightGBM 0.600, cross-encoder 0.521, product 0.018. Final
  (mpnet continued on batch A, then C): intercept -0.026, LightGBM 0.584, cross-encoder 0.573, product 0.026.
- *Decision.* Target-argmax (each Source 2/3 record to the entity with its highest
  probability), then a flat threshold.

Pair ranking on held-out, as AUC over the 256,812 gated held-out pairs:

| Model | Training pairs | AUC |
|---|---|---|
| LightGBM (127 leaves, France self-labels) | | 0.99327 |
| mpnet, first stage | 2.95M | 0.99481 |
| e5-base on batch A | 2.92M | 0.99582 |
| mpnet continued on batch A | +2.92M | 0.99637 |
| mpnet continued on batch B | +2.92M | 0.99641 |
| mpnet continued on batch C | +2.92M | 0.99636 |
| mpnet continued on batches A+B | +5.84M | 0.99671 |

Each continuation in this table starts from the first stage. More pairs keep helping, in
shrinking steps. At threshold 0.75, a second member next to A+B moved held-out F0.5 by at
most 0.00004 (e5: 0.98372 against 0.98368 for A+B alone; B and C together: 0.98367).

**Threshold selection method:** held-out macro F0.5, out of fold and weighted to the test
country mix, over a grid of flat thresholds after target-argmax. LightGBM alone peaked at
0.75 (0.70 0.96203, 0.75 0.96263, 0.80 0.96221, 0.85 0.96134); the sub12f blend peaked at 0.80
(0.98324), and later blends kept 0.80 so that leaderboard changes isolate the model,
although some cross-encoder variants peaked at 0.75. The one-to-one step is worth about +0.0007 on a dense held-out where every
entity of a city competes (0.97179 against 0.97115). The rule derived from the metric
(add a candidate while its probability exceeds the entity's current expected score divided
by 1.25) needs calibrated probabilities, and it lost to the flat threshold on the same model
(0.87425 against 0.91102). France's 0.80 comes from the full leave-one-country-out check
(Section 2.4).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** final submission sub18: 0.98499 held-out on US/India (US 0.98648,
  India 0.98378), including the India blank-address rule and the rescue stage; public
  leaderboard 0.97902. For comparison, sub12f scored 0.98324 held-out (out-of-fold blend over
  the 60,000 held-out US/India entities, weighted to the test mix; India 0.98168, US 0.98515)
  and 0.977362 public. The LightGBM stage alone scores 0.97749 +/- 0.00056 (4-fold
  cross-fit, unweighted).
- **Common false positives (wrong merges):** in the final LightGBM cross-fit, extra wrong
  ids on entities with matches cost 0.00270, and matches given to entities without any cost
  0.00128. The typical wrong merge is a different business at the same site: same street, a
  house number one or two digits away ("2251" vs "2256", "25-112" vs "25-133"), or a decoy
  record whose number sits a small positive offset from the real one. Next come branches of
  a chain in one city, where only the address separates them.
- **Common false negatives (missed matches):** true pairs scored under the threshold are the
  largest loss (0.01251), then true matches never blocked (0.00573) and those cut by the
  pre-ranker (0.00104). Misses concentrate on copies with a blanked address that compete with
  same-name siblings, and on copies whose distinctive name word was dropped or replaced by a
  generic one while the house number was also perturbed; both are ambiguous by construction.
- **Where the final blend's misses sit** (held-out US/India, final blend at 0.80: precision
  0.9989, recall 0.953, 3.8 wrong ids per 1,000 entities). 1.94% of true matches never reach
  the scored candidates and 2.78% are scored below the threshold. By record type:

  | Record type | Share of true pairs | Recall | Share of scored misses | Share of never-blocked |
  |---|---|---|---|---|
  | name overlaps, address present | 0.803 | 0.986 | 0.083 | 0.463 |
  | Indic-script name | 0.071 | 0.978 | 0.011 | 0.065 |
  | domain-style name ("name.com") | 0.045 | 0.962 | 0.001 | 0.088 |
  | name overlaps, blank address | 0.044 | 0.411 | 0.775 | 0.234 |
  | unrelated brand name, address present | 0.036 | 0.823 | 0.123 | 0.148 |

  Blank-address copies are the largest miss. When the copy's name is unique among the
  country's Source 1 names they are already found (97.5% of such candidates are true, 94%
  recalled). The rest are ambiguous by construction: about 1.3% of all true pairs are
  blank-address copies whose exact name is shared by several Source 1 entities.

### 5.1 What moved the metric

Held-out macro F0.5 by evaluation setup. Numbers are comparable within a setup, not across
setups.

| Change | Before | After |
|---|---|---|
| *25,000-entity sample, about 6,200 held out* | | |
| Address number-word blocking keys (pair recall 80.3% -> 91.9%) | 0.77617 | 0.82097 |
| Shared number-word count feature; keys skip words under 4 characters | 0.82097 | 0.82941 |
| Pre-ranker keeps the top 50 by name + address trigram overlap | 0.82941 | 0.83808 |
| 9 features for missing text, scripts, number sets, unique tokens; flat threshold instead of greedy admission | 0.83808 | 0.91102 |
| *60,000-entity sample, 14,979 held out* | | |
| Gap-to-best and rank of name and address trigram Jaccard | 0.91308 | 0.91756 |
| *15,000 held out, entities without candidates included* | | |
| Transliteration table and fixed Indic tokenization | 0.91655 | 0.93488 |
| Core-name blocking key | 0.93488 | 0.93921 |
| Number truncation and sibling features | 0.93921 | 0.94054 |
| Adjacent address word-pair key | 0.94054 | 0.94593 |
| rapidfuzz name and address features | 0.94593 | 0.94986 |
| Word-rarity features; threshold 0.75 | 0.94986 | 0.96263 |
| Core-name multiplicity features (top 35) | 0.96283 | 0.96528 |
| Solo slots in the pre-ranker (8 per field, cap 40) | 0.96528 | 0.96787 |
| Core-name word-pair key | 0.96758 | 0.96920 |
| Concatenated-name key | 0.96920 | 0.97012 |
| Address multiplicity features | 0.97012 | 0.97191 |
| Signed number offset | 0.97217 | 0.97364 |
| *4-fold cross-fit over 60,000 held-out entities* | | |
| 5 bagged models instead of one | 0.97252 | 0.97349 |
| Drop the shared-sibling-number feature (its single-split gain was noise) | 0.97349 | 0.97382 |
| +60,000 / +180,000 training-only entities | 0.97382 | 0.97473 |
| Learned pre-ranker | 0.97473 | 0.97604 |
| Per-country address token weights | 0.97604 | 0.97696 |
| 127-leaf trees and France self-labels | 0.97696 | 0.97749 |
| *Blend, weighted to the test country mix (US/India); in the first two rows "Before" is the LightGBM of the time alone, under this weighting* | | |
| MiniLM cross-encoder blended with LightGBM | 0.97559 | 0.98102 |
| mpnet cross-encoder instead (127-leaf LightGBM with France self-labels) | 0.97618 | 0.98222 |
| mpnet continued on 400,000 more training entities | 0.98222 | 0.98324 |
| mpnet continued on batch A, then on batch C | 0.98324 | 0.98368 |
| India: blank-address record kept when its name equals the entity's unique name | 0.98368 | 0.98402 |
| Rescue stage (three further key families, separate scorer, 2.5 pairs per held-out entity) | 0.98402 | 0.98499 |
| *India as the unseen country (LightGBM trained on US, self-training, threshold 0.80)* | | |
| Legal-form and added-word decoy features | 0.92925 | 0.93428 |
| Sibling-context group features (decoy signal relative to the entity's other candidates) | 0.93428 | 0.93481 |
| Address-number roles and directional address agreement | 0.93481 | 0.93783 |

### 5.2 Leaderboard

| Submission | Change | Held-out | Public |
|---|---|---|---|
| sub04 | 44 features, pre-ranker with solo slots, one-to-one + 0.75 | 0.96787 (single split) | 0.955044 |
| sub05 | French street and legal normalization | 0.96758 (single split) | 0.955 |
| sub06 | new keys and features, 5 models, +180,000 training entities | 0.9747 (cross-fit) | 0.961932 |
| sub07 | learned pre-ranker | 0.97604 (cross-fit) | 0.963182 |
| sub08ce | address token weights; MiniLM cross-encoder blend for US/India | 0.98102 (blend) | 0.974133 |
| sub12f | 127 leaves and France self-labels; mpnet continued on 400,000 more entities; France at 0.80 | 0.98324 (blend) | 0.977362 |
| sub13 | mpnet continued on batch A then C | 0.98368 (blend) | 0.977664 |
| sub13fr | test: France scored by the cross-encoder blend | 0.98368 (US/India unchanged) | 0.97755 |
| sub17 | India blank-address rule; France with decoy and sibling-context features; US/India rescue stage | 0.98499 (blend) | 0.97897 |
| sub18 | France with address-number-role features | 0.98499 (US/India unchanged) | 0.97902 |

### 5.3 What did not work

- **Calibrated probabilities with the metric's own admission rule.** Isotonic calibration
  with a per-entity expected match count scored 0.76137 against 0.77627 uncalibrated, and
  greedy admission lost to a flat threshold (0.87425 against 0.91102).
- **More candidates.** Keeping 80 or 70 candidates scored 0.94178 and 0.96131 against 0.94593
  and 0.96263. Larger key groups (max_group 300) gave +0.00045 for 47% more raw pairs and
  were not adopted; max_df 200 changed nothing.
- **Features:** core-name equality (neutral), counts of numbers present on one side only
  (0.97364 to 0.97217), truncation direction (to 0.97128), shared sibling numbers (removed,
  see 5.1).
- **Model variants at the time:** 127 leaves with learning rate 0.05 on a single model
  (0.96220 against 0.96283), single-model subsampling (0.96584 against 0.96787), learning
  rate 0.05 with 600 rounds (no change). 127 leaves only paid off once combined with the
  5-model bag and self-labels.
- **A second stage with seed features** (similarity to the entity's confident candidates):
  0.96797 against 0.96787.
- **Propagating predictions across content-identical duplicate records:** 0.96920 to 0.96918.
- **More data on a single split:** +90,000 training entities looked harmful (0.94986 to
  0.94926), while under cross-fit the same kind of change helped (0.97382 to 0.97473). The
  single split was too noisy to judge it.
- **Two cross-encoders of different sizes:** a learned blend gave MiniLM about zero weight
  next to mpnet (0.98226 against 0.98222).
- **For France:** the cross-encoder (Section 2.4), a high threshold taken from a sample check
  (reversed by the full check), and further attempts that did not beat the self-trained
  LightGBM on India as the unseen country: synthetic France-style pairs built from Source 1
  records, dropping the dataset-statistics features, covariate-shift weights, a second
  self-training round, stricter self-labels and other tree sizes.
- **Late experiments that did not clear their adoption bar** (US/India held-out through the
  full decision procedure, or India as the unseen country): fine-tuning the cross-encoder on
  mined hard examples (-0.0019); a context-aware second-stage blend (+0.00017); scoring pairs
  below the 0.02 gate with the cross-encoder (+0.00005); a lower France threshold matched to
  the labelled countries' match count (-0.0036 on India as the unseen country); record-to-record agreement with
  the entity's confident matches (+0.00023); the India blank-address rule applied to an
  unseen country (+0.00062); hard-example weighting in LightGBM (+0.0009 / +0.0003, mixed
  folds).

---

## 6. Conclusion

A cascade of within-country blocking on address number-word and name-pair keys, a learned
pre-ranker, a LightGBM pair model and a cross-encoder blend reaches 0.98499 held-out macro
F0.5 on US and India, with France self-trained and kept on LightGBM. The final submission
(sub18) adds an India blank-address rule, a rescue stage for US/India and features for
France that describe how two records differ. It scores 0.97902 on the public leaderboard.

Three lessons carried the work:

- Blocking is measured by the F0.5 it loses, not by pair recall.
- A single held-out split is too noisy to rank changes, while cross-fitting over all
  sampled entities makes 0.001 effects visible.
- Decisions whose outcome depends on competition between entities, such as the one-to-one
  step and its threshold, must be checked on complete splits rather than samples.

The remaining gap is France. With no labels, our best proxy for it (India scored by a
US-only model) sits about 0.05 below a labelled country. The address-number-role columns
closed 0.003 of that gap on the proxy, yet moved the public score by only +0.00005.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` is a self-contained, runnable copy of the pipeline;
its `README.md` gives every command in order, with runtimes and hardware.

```
config.py                paths, column names, seed
requirements.txt         pinned dependencies (CPU and GPU parts)
src/er_norm.py           normalization, blocking-key and number extraction
src/er_io.py             streaming TSV readers, cached arrays
src/translit.py          transliteration table from training pairs
src/candidates.py        token postings for blocking
src/pair_model.py        blocking, pre-ranker, pair features, LightGBM fit
src/diff_features.py     address-number-role columns (France model); name-alignment columns, not used
src/rescue.py            rescue stage: extra key families, scorer fit and apply
src/merge_scores.py      replaces one country's rows of a score dump
src/loss_breakdown.py    held-out cross-fit, loss by cause, dumps
src/predict_matches.py   fit and score a whole split, dump every pair score
src/pseudo_labels.py     self-labels for countries absent from training
src/ce_pairs.py          cross-encoder pair export
src/ce_more_pairs.py     extra cross-encoder training batches
src/ce_rerank.py         cross-encoder training and scoring (GPU)
src/kaggle_ce_runner.py  the Kaggle GPU notebook wrapper for ce_rerank.py
src/ce_blend.py          single cross-encoder blend, dump rewriting and checks
src/ce_ensemble.py       blend of LightGBM with several cross-encoders
src/decide.py            decision rule; writes both TSVs
src/check_submission.py  byte-level format checks
src/metrics.py           F0.5 per entity and macro
src/decision_rules.py, src/log_experiment.py, and the analysis tools eda_er.py,
blocking_ceiling.py, baseline_exact.py, dense_eval.py, full_density_eval.py, stage2.py
```

Entry points, in order: `translit.py`; `loss_breakdown.py` and `predict_matches.py` (first
round); `pseudo_labels.py`; `loss_breakdown.py` and `predict_matches.py` (final LightGBM);
`ce_pairs.py export`; `ce_more_pairs.py sample` / `merge`; `ce_rerank.py` (GPU);
`ce_ensemble.py fit` / `apply`; France's final LightGBM (`pseudo_labels.py` and
`predict_matches.py --countries France` with `ER_DIFF=address`) and `merge_scores.py`;
`decide.py --candidates final-stage --blank-name-rule India`; `rescue.py fit` / `apply`, which
writes `matching_results.tsv` and `candidate_pairs.tsv` to its `--out` folder; then the
organizers' `utils/validate_submission.py` and `check_submission.py`. The US/India LightGBM
was fitted before the France-only columns existed; the README's reproducibility notes explain
why rerunning that step with this code gives slightly different scores.

### B. Additional Results

Where the final LightGBM cross-fit loses F0.5 (points of the overall macro score, threshold
0.75, 60,000 held-out entities):

| Cause | India | US | Total |
|---|---|---|---|
| true pairs scored below the threshold | 0.00568 | 0.00683 | 0.01251 |
| true matches never blocked | 0.00303 | 0.00270 | 0.00573 |
| wrong extra ids | 0.00139 | 0.00131 | 0.00270 |
| matches given to entities without any | 0.00057 | 0.00072 | 0.00128 |
| true matches cut by the pre-ranker | 0.00084 | 0.00020 | 0.00104 |

Cross-encoder held-out AUC on gated pairs: mpnet 0.99481, mpnet continued on batch A
0.99637. One epoch on about 2.9M pairs takes 3.7 hours on one T4; scoring the 7.36M gated
test pairs of all three countries takes about an hour on two.

Scoring identity used throughout: with c correct ids among n predicted and k true, F0.5 =
1.25c / (n + 0.25k). Adding a candidate of probability p raises the expected score only
while p exceeds the current score divided by 1.25, so the bar rises with every id admitted.
