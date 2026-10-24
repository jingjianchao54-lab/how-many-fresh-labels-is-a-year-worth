# Analysis plan, version 1.0

Working title: *How Many Fresh Labels Is a Year Worth? Measuring the Exchange Rate between Label Recency and Label Quantity for Lightweight Text Classifiers*

Target journal: Algorithms (MDPI). Author decision of 2026-10-09: candidate A; no time-weighting of training samples; second dataset HuffPost News Category; frozen MiniLM baseline, accuracy only (no speed or cost reporting).

Status: FROZEN before any count, split size, class distribution or score defined below was computed.
Later changes go to `exp/AMENDMENTS.md` with date and reason; this file is not edited after freezing.

## 1. Prior exposure (declared before freezing)

- The ten Reddit corpora were used in the author's doctoral thesis and in a separate manuscript on user-level next-community prediction. Known from that work: totals after cleaning (16,765,540 utterances read, 14,784,735 retained), the overall time range (2008-02 to 2018-10-31), user-level results, and that a user-level TF-IDF classifier over a whole posting history reached Macro-F1 0.75. No utterance-level classifier, no per-year count and no cross-year score has been computed.
- HuffPost file `data.json`: byte size (87,295,572), line count (209,527), SHA-256, field names and the first two records were inspected. No per-year or per-category count has been computed.
- Literature read in full before freezing: Luu et al. 2022, Agarwal and Nenkova 2022, Röttger and Pierrehumbert 2021. Alkhalifa et al. 2023 was read at summary level only.

## 2. Research questions and hypotheses

- RQ1. How much does a classifier trained on year t lose on test data from year u ≠ t, relative to a classifier trained on year u with the same number of labels?
- RQ2. How many labels from the test year are needed to match a classifier trained on N labels per class from g years earlier (break-even size n*(g), label-equivalence ratio ρ(g) = n*(g) / N)?
- RQ3. When only n fresh labels per class are available, does adding the N stale labels still help?

Primary model for hypothesis tests: TF-IDF + logistic regression. Primary direction: forward (training year earlier than test year). All tests are run per dataset.

- H1 (staleness cost exists). C(1) > 0: the lower bound of the 95% interval of C(1) is above 0.
- H2 (cost grows with the gap). C(g_max) − C(1) > 0 with the 95% interval above 0.
- H3 (labels expire). ρ(g_max) < 0.5. Supported if the upper 95% bound of n*(g_max) is below N/2; not supported if the lower bound is at or above N/2; otherwise inconclusive.
- A staleness cost below 0.010 Macro-F1 is described as practically small whatever its interval.
- Every outcome is reported, including "stale labels do not expire on this task".

## 3. Data

### 3.1 Reddit-10 (D1)
- Source: ten ConvoKit subreddit corpora (chess, Coffee, Cooking, gardening, history, investing, languagelearning, philosophy, running, woodworking); `utterances.jsonl` of each.
- Unit: one utterance (post or comment). Label: its subreddit (10 classes). Time: calendar year (UTC) of its timestamp.
- Exclusions, in order: missing id, author, text or timestamp; text equal to `[deleted]` or `[removed]` after stripping, or empty; author `[deleted]`; author equal to `AutoModerator` or matching `bot\d*$` or `^auto[-_]?mod` (case-insensitive); exact duplicates of (author, text) within a subreddit (first kept).
- Author split (global, all years and subreddits): u = first 8 bytes of SHA-256("paper5|" + author) / 2^64; train if u < 0.60, validation if 0.60 ≤ u < 0.70, test if u ≥ 0.70. Training, validation and test authors are disjoint in every comparison.
- Per-author cap: at most 5 eligible utterances per author within one (subreddit, year) cell, the earliest five by timestamp then id.

### 3.2 HuffPost (D2)
- Source: News Category Dataset (Misra), file `data.json` from the Hugging Face mirror `heegyu/news-category-dataset`, licence CC BY 4.0, SHA-256 2cfc51fa4a82a96ecbe251fa383737aa1dea2e0a517f61dece633fcc25a4a065.
- Unit: one record; text = headline + " " + short_description. Label: `category` as given (no merging of categories). Time: calendar year of `date`.
- Exclusions: missing headline or date; exact duplicate text (first kept).
- Split by record: u from SHA-256("paper5|" + link) as above, same thresholds. No per-author cap (author field is incomplete).

### 3.3 Text processing (both datasets)
- Lower-case; tokens = matches of `[a-z0-9]+(?:'[a-z]+)?`; URLs replaced by a space before tokenising.
- D1 primary analysis: tokens equal to a lower-cased subreddit name are removed (name masking). Unmasked text is a sensitivity analysis.
- Eligibility by length: at least 10 tokens (D1) or 5 tokens (D2) after processing. Texts are truncated to the first 256 tokens for every model.

### 3.4 Cell sizes, classes and year window (rule fixed here, applied to counts only)
A cell is (class, year). A cell is eligible at size s = (N, V, T) if, after exclusions, caps and the split, it has at least 2N training-pool items, V validation items and T test items.

- Size ladders. D1: (2000, 500, 1000), (1000, 250, 500), (500, 150, 300). D2: (1000, 200, 500), (500, 100, 300), (300, 100, 200).
- D1: all ten classes are required. Take the largest size for which at least 6 consecutive years have all ten classes eligible; at that size take the longest such run of years (ties: the later run). If no size gives 6 years with ten classes, repeat with at least 8 classes (dropping the ineligible ones) and report it.
- D2: take the largest size for which at least 5 consecutive years have at least 5 eligible classes; at that size take the longest run (ties: more classes, then the later run). Classes = those eligible in every year of the run; if more than 10, the 10 with the largest minimum yearly training-pool count.
- If a dataset fails the smallest size, work on it stops and the author is consulted. The rule is not changed after counts are seen.

### 3.5 Sampling
- Test set: T items per cell, drawn once (seed 0). Validation set: V items per cell, drawn once (seed 0).
- Training pool: 2N items per cell, drawn once (seed 0). Replicate r = 1..5 draws N items per cell from the pool (seed r). Smaller training sizes are nested prefixes of the replicate's shuffled N.
- Training sizes per class: n ∈ {25, 50, 100, 200, 500, 1000, 2000} up to N.

## 4. Models

| Id | Representation | Classifier | Grid |
|---|---|---|---|
| M1 (primary) | TF-IDF, word 1–2-grams, min_df 2, sublinear tf, L2 norm, max 200,000 features, fitted on the training set only | multinomial logistic regression, lbfgs, max_iter 1000 | C ∈ {1, 10, 100} |
| M2 | as M1 | linear SVM (LinearSVC) | C ∈ {0.1, 1, 10} |
| M3 | word 1–2-gram counts, min_df 2 | Complement naive Bayes | alpha ∈ {0.1, 0.5, 1.0} |
| M4 | mean of GloVe 6B 100d vectors of in-vocabulary tokens | logistic regression on standardised features | C ∈ {0.1, 1, 10} |
| M5 | frozen all-MiniLM-L6-v2, mean pooling, L2-normalised, 256 word pieces, NumPy implementation checked against the published reference sentences | logistic regression | C ∈ {0.1, 1, 10} |

- Hyper-parameters are selected per (dataset, model, training size n) by mean validation Macro-F1 over training years, with validation data from the same year as the training data, on replicate 1. Ties: the smaller C or alpha. The selected value is then fixed for all years and replicates.
- No sample weighting by time and no temporal adaptation method is used.
- M5 reports accuracy measures only; no encoding time, memory or cost.

## 5. Experiments

- E1 Aging grid. Train each model on year t with N labels per class; score on the test set of every year u in the window. S(t→u) is Macro-F1.
  - Staleness cost C(g) = mean over u of [S(u→u) − S(u−g→u)], g ≥ 1 (forward). The backward counterpart is reported separately.
  - Temporal degradation score of Luu et al. (2022) is also reported for comparison with that paper.
- E2 Exchange rate. For each test year u, train on n fresh labels per class from year u for every n; F(n, u) is Macro-F1 on the test set of u. For each gap g, average F(n, u) and S(u−g→u) over the test years that have gap g and over replicates. n*(g) is the smallest n at which the averaged fresh curve reaches the averaged stale score, by linear interpolation in log n between grid points. If the fresh curve at N is still below the stale score, n*(g) is reported as "> N"; if the smallest n already reaches it, as "< 25". ρ(g) = n*(g) / N.
- E3 Stale plus fresh. Train on N stale labels (year u−g) plus n fresh labels (year u), concatenated without weights, for g ∈ {1, g_max} and every n < N; compare with fresh-only n and stale-only N.
- E4 Descriptive. Per-class staleness cost; Jensen–Shannon divergence between unigram distributions of training pools of two years, per class; Spearman correlation between divergence and per-class F1 loss over (class, year-pair) cells. No causal claim.
- E5 Sensitivity. D1 without name masking; D2 headline only; M1 at half the cell size.

Metrics: Macro-F1 (primary), accuracy.

## 6. Inference

- 95% percentile bootstrap, 2,000 resamples of test items within each (class, year) test cell, paired across models and conditions; replicate scores are averaged inside each resample. Seed 20261009.
- n*(g) intervals come from the same resamples (the interpolation is repeated per resample; censored values are kept as censored when computing percentiles).
- Replicate-to-replicate standard deviations are reported alongside.
- Holm correction across H1–H3 within a dataset.

## 7. Locking and test use

1. Counts-only data gate (Section 3.4); results written to `results/gate.json`.
2. Hyper-parameter selection on validation data only; `results/selection.json`.
3. `results/LOCK.json` records the selected values and SHA-256 of this plan, the sample index files and the evaluation scripts.
4. Test sets are scored once per experiment after locking.

## 8. Not done in this paper

No time-weighted training, no temporal adaptation or continual learning method, no user-level history, no label prior, no next-community prediction, no first-visit analysis, no cold-start grouping, no cost or speed comparison. Fine-tuned transformers are not run (no GPU); this is stated as a limitation.
