# 4060 Ti lane — hypothesis: unordered pairs of core name words as a blocking key

Baseline reproduced on this machine: macro F0.5 **0.96758**, 15,000 held-out entities
(India 5,983, US 9,017), fingerprint `15000 16440644396`.

## 1. The rule

The generator builds a true copy by editing **whole words one at a time** — appending or
dropping a legal or filler word, reordering the words, duplicating one, or corrupting the
characters of a single word — so two of the original words almost always survive intact
together, while every existing name key needs either the whole token set (`name_key`,
`sort_key`, `core`) or one individually rare token (`tok1`) and is therefore destroyed by any
one of those edits.

## 2. Target bucket and country

`not_blocked` (0.01012 of the overall score; US 0.00576, India 0.00436) — true matches that no
blocking key reached, so they cannot be scored at all and are a guaranteed loss. The effect is
much stronger for **US** (56.6% of its missed ids are reachable, against 23.6% for India).

Two reinforcing observations on this bucket: 448 of 1,457 missed ids (30.7%) have an address
that normalizes to empty, so the name is the only route to them at all; and the remainder have
addresses the generator rewrote heavily (reordered components, dropped sub-unit numbers).

## 3. Evidence

Measured over **every** `not_blocked` true id in the held-out baseline (1,457 ids, not a sample):

| shared core name tokens with Source 1 | ids | share |
|---|---|---|
| >= 2 (reachable by a word-pair key) | 614 | 42.1% |
| exactly 1 | 533 | 36.6% |
| 0 | 310 | 21.3% |

- All 614 are reached by an **unordered all-pairs** key; sorted-**adjacent** pairs reach only
  446 (30.6%), so the key has to be all pairs, not the adjacent form `ww` uses for addresses.
- By country: US 463/818 (56.6%), India 151/639 (23.6%).
- Of the 34 `not_blocked` examples printed in full and read by hand (23 US, 11 India), 20 share
  an intact core-word pair. The India misses are dominated by two other mechanisms this rule
  does not address: a two-word distinctive name with one word corrupted (only one word
  survives, so no pair does), and the name replaced by a concatenated domain
  ("Elite Projects Limited" -> "Eliteprojects.Com"), which normalizes to a single token.

A pair of words is rare even when both words are common. Document frequency of the 876 winning
(pair, country) groups across all 10,320,219 target records: median 75, p75 286, p90 1,664.
**67.2%** are at or below `max_group` 200 and 76.4% at or below `max_df` 300, so about two
thirds of the reachable ids survive the existing caps.

Cost is small, because business names are short — 87.7% of target records have at most 3 core
tokens and 96.1% at most 4. Capping at 8 tokens per record, all pairs come to **2.25 postings
per record**, 23.2M in total, which is well under the address-token family already built.

## 4. Upper bound

Ceiling if every reachable id were recovered *and* then scored above the threshold:
0.01012 x 0.421 = **+0.0043**. Applying the measured df caps: 0.01012 x 0.421 x 0.672 =
**+0.0029**. Both are ceilings, not forecasts: a recovered candidate must also survive the
top-40 pre-ranker and clear 0.75, so the realistic figure is lower, and this sits close to the
+0.002 bar rather than comfortably above it.

## 5. Identical computation on test

The key is a new `TOKEN_FIELDS` family built by `candidates.build_token_postings`, which
streams the split's own three source files and reads nothing else — the same path `nw4`, `core`
and `ww` already take. No labels, no ground truth, no train-side table: `predict_matches`
builds the index for `test` with `build_index(split, max_df)` and gets the test postings.
The Source 1 query side goes through `_query_tokens`, which is given the split's own raw text.

## 6. French records

The rule keys on nothing language-specific: no country field, no word list, no script
assumption. It needs only a name with at least two core words, which French business names have
("Boulangerie Dupont Freres"), and `FILLER` already strips the French legal forms
(`sarl`, `sas`, `sci`, `ste`, `eurl`, `snc`, `cie`, `groupe`, `fils`, `freres`) so the pair is
built from the distinctive words as it is for US and India. France should therefore behave like
the US case rather than the India one, since its misses come from name edits rather than from
Indic-script or domain-concatenated names. **Unverified**: train has no French records, and
measuring this would need a test pass, which this lane does not run.

## 7. Risks to watch

- More candidates per entity push harder against the top-40 pre-ranker: the `prefiltered`
  bucket (0.00388) could grow while `not_blocked` shrinks.
- Runtime rises with a sixth postings family and a larger raw pair count.
