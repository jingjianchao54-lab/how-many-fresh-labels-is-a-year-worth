# Amendments and execution records (analysis plan v1.0, SHA-256 283db2c2fffdc81c320dd113edae05693043d07dbbe7413819b8aab1a093003e)

## R-01 Data gate outcome (2026-10-09 20:10 +08:00; counts only, no model fitted)
- D1 Reddit-10: passed at the largest size (2000, 500, 1000); all ten classes; years 2012-2018 (7 years).
- D2 HuffPost: FAILED at every size. At the smallest size only POLITICS and ENTERTAINMENT are eligible in five consecutive years (the category scheme changed in 2014 and the volume is small after 2018). As the plan requires, work on D2 stopped and the author was consulted (message sent shortly after the gate ran). The rule was not changed.

## N-01 Implementation notes written before any validation or test score was computed (2026-10-09 20:15 +08:00)
- TF-IDF is computed from a global 1-2-gram count matrix restricted, for each training set, to terms with document frequency >= 2 in that training set (at most 200,000 by training term count, stable order for ties), with IDF and L2 normalisation from the training rows only. A unit test shows equality with scikit-learn's TfidfVectorizer fitted on the training rows (max difference < 1e-10).
- M3 (Complement naive Bayes) uses all terms with training document frequency >= 2 (the plan states no cap for M3).
- E3 (stale N plus fresh n): the plan fixes hyper-parameters per training size but does not name the value for the combined set of N + n items per class; the value selected for size N is used.
- Replicate r draws its N training items per cell with a generator seeded by (r, class, year) from the fixed pool of 2N; smaller sizes are prefixes of that draw.
- Logistic-regression fits that stop at max_iter are flagged in the run files and reported.

## A-01 Replacement of the second dataset (registered 2026-10-09 20:38 +08:00, before any Stack Exchange file was downloaded or counted)
Reason: D2 HuffPost failed the pre-specified gate (R-01). The author chose (20:37) to replace it with the public Stack Exchange data dump. HuffPost is not used further; its failure is reported in the paper.

- Source: Stack Exchange data dump hosted on the Internet Archive (item `stackexchange`, files dated April 2024), licence CC BY-SA.
- Sites (10 classes), chosen before download as the ten largest archives among the topical-interest sites whose archive size is between 25 MB and 125 MB in the listing consulted on 2026-10-09 (chess, cooking, gardening, history, fitness, boardgames, pets, outdoors, bicycles, movies, music, photo, philosophy): philosophy, photo, music, history, cooking, movies, bicycles, boardgames, gardening, chess. No count of posts per site or per year was known when this list was fixed.
- Unit: one question body, answer body or comment (`Posts.xml` rows with PostTypeId 1 or 2, and `Comments.xml` rows). Label: the site. Time: calendar year (UTC) of CreationDate.
- Text: HTML tags removed and entities unescaped; then the processing of plan Section 3.3. No name masking (site names are not used as forum references inside posts). Minimum 10 tokens.
- Exclusions, in order: missing text, date or user id (OwnerUserId for posts, UserId for comments); user id -1 (the Community account); exact duplicates of (user id, text) within a site.
- Author split and cap: as plan Section 3.1, with the key "<site>:<user id>"; at most 5 items per user within one (site, year) cell.
- Cell sizes and year window: the D1 ladder and the D1 rule of plan Section 3.4 are applied unchanged (ten classes required over at least 6 consecutive years; fallback to at least 8 classes). Because the dump ends in early 2024, calendar year 2024 is excluded before the rule is applied.
- Everything else (sampling, models, selection, experiments, inference, locking) is as in the plan. Sensitivity "D2 headline only" is dropped; D2 has no name-masking sensitivity.
- If D2 fails the gate again, the paper proceeds with D1 only and reports both failures; no third dataset is tried without a new registered amendment.

## R-02 Data gate outcome for Stack Exchange (2026-10-09 22:02 +08:00; counts only, no model fitted)
- All ten archives were obtained from the Internet Archive item `stackexchange` (April 2024 files), as registered in A-01; byte sizes agree with the listing. The mirror considered at 21:30 was not used.
- The D1 rule applied unchanged: passed at size (500, 150, 300) with all ten classes; years 2015-2021 (7 years). Larger sizes fail because of the validation cells of the smaller sites (for example chess 2015: 184 validation items).
- Training sizes for D2 are therefore n in {25, 50, 100, 200, 500}, N = 500, training pool 1000 per cell.
- Note: LOCK_reddit.json (20:36) recorded the hash of this file before A-01 and R-02 were appended; nothing above the A-01 heading has been changed since.

## X-01 Execution record, Reddit-10 (D1)
- Locked 2026-10-09 20:36:26 +08:00 (results/LOCK_reddit.json). Test predictions written 20:36-23:50; the cloud container restarted several times and the resumable runner was relaunched each time; all 1,820 run files load and no logistic-regression fit stopped at max_iter.
- Test sets scored once, 2026-10-09 23:50-23:54 +08:00: `evaluate.py reddit test M1 M2 M3 M4`; results/results_reddit.json SHA-256 a3f933ff011d479b1ddf5eb1e74aaaa57e284ed3f90ed07724fbdc18eba3e8c2. `drift.py reddit` run once afterwards (descriptive).
- Before locking, the pipeline was dry-run on validation sets only (replicate 1, M2-M4); that file is kept as results_reddit_dry_validation_M2M3M4.json and is not a result.
- M5 (MiniLM) for D1 not yet run at this point; it will be selected on validation data and scored separately, and reported as such.

## X-02 Execution record, Stack Exchange (D2), and one reporting fix
- Selection on validation data and locking ran automatically (src/pipeline_se.sh): locked 2026-10-09 22:36:59 +08:00 (results/LOCK_stackexchange.json). 1,260 test run files; no fit stopped at max_iter.
- Test sets scored once, 2026-10-09 23:55 +08:00. First output kept as results/results_stackexchange_before_X02.json.
- Reporting fix after scoring (2026-10-10 00:00 +08:00): when the 97.5th percentile of the break-even size fell between a finite resample and a censored one, numpy's interpolation with the censoring sentinel printed a meaningless number (M4, gap 5, upper bound). Such bounds are now reported as censored ("> N"). The scoring was re-run with the same seed; the only two fields that changed are that upper bound and its ratio (verified field by field). D1 had no such case. Final results/results_stackexchange.json SHA-256 3fc2de9069eee3e70818445197e411c1905200ce81439625c61d57228b9c725f.

## X-03 Code changes after the main locks (2026-10-10 00:10 +08:00; no M5 score of any kind existed yet)
- run.py: the test stage reads M5 hyper-parameters from a separate lock file (results/LOCK_<dataset>_M5.json) because the M5 vectors were not available when the main locks were written; two environment switches were added for the pre-registered sensitivity runs (E1 and E2 only; stale model at half the cell size).
- select_lock.py: `lock_m5` selects M5 on validation data (replicate 1, same rule as the other models) and writes that separate lock; main lock files are not modified.
- evaluate.py: E3 files that were deliberately not produced (sensitivity runs) are skipped instead of raising an error. Scores of M1-M4 are unaffected (deterministic re-run check to follow when M5 is added).
- M5 input text: natural text with URLs removed (D1: subreddit names removed by a word-boundary pattern), truncated at 256 word pieces by the encoder.

## A-02 Reduced design for M5 (registered 2026-10-10 00:13 +08:00; no M5 selection or test score existed; 2 of 26 Stack Exchange encoding chunks had been written and are discarded)
Reason: on the available machine (2 CPU cores, no GPU) the NumPy encoder processed about 9-16 texts per second, so encoding every sampled text of both datasets at 256 word pieces would take more than ten hours.
- M5 is run on replicate 1 only. Texts encoded: all validation and test items, and the N training items per cell of replicate 1 (not the full 2N pool).
- Encoder input is truncated at 128 word pieces instead of 256.
- Everything else for M5 is unchanged: validation-only selection, separate lock, one scoring. M5 results are reported with test-set bootstrap intervals and without a replicate standard deviation, and this reduction is stated in the paper.
- M1-M4 are not affected.

## X-04 Execution record: M5 on Stack Exchange, and sensitivity runs (2026-10-10 00:55-01:15 +08:00)
- M5 (A-02 design) on D2: validation selection, lock 00:55:32 (results/LOCK_stackexchange_M5.json), 63 test run files, scored once with `evaluate.py stackexchange test M5 M1 M2 M3 M4 --reps1` (output results_stackexchange_rep1.json; M1-M4 there are the replicate-1 subset of the already scored runs, shown for a like-for-like comparison). evaluate.py gained the `--reps1` switch for this purpose.
- E5 half-size sensitivity (primary model): D1 with N/2 = 1000; D2 with 200 labels per class, because 250 is not on the pre-specified size grid (the nearest grid size below N/2 is used). Outputs results/sens_half_<dataset>.json.
- E5 name-masking sensitivity (D1, primary model, unmasked text, E1 and E2 only, hyper-parameters as locked for the masked text): results/results_reddit_tok_raw.json.

## X-05 Execution record: M5 on Reddit-10 (2026-10-10 02:10-02:30 +08:00)
- Encoding under A-02 (validation, test and replicate-1 training items; 245,000 texts; 128 word pieces) finished about 02:10. Validation-only selection, then the separate lock results/LOCK_reddit_M5.json at 02:26:07 (C = 1 at every size), then 91 test run files.
- Scored once with `evaluate.py reddit test M5 M1 M2 M3 M4 --reps1`; output results/results_reddit_rep1.json, SHA-256 858bbe723808fd616b0d9bbeae318ce74e8bafe1b6341355e1f089f118826192. The main file results_reddit.json is unchanged (SHA-256 a3f933ff...).
- The runtime warning "Mean of empty slice" printed by evaluate.py comes from averaging the E3 array over gap values for which E3 is not defined; no reported number is affected.
- Figures are drawn by src/make_figures.py from the result files only; M5 is taken from the replicate-1 files and M1-M4 from the five-replicate files, and the captions say so.

## X-06 Post-hoc computations for presentation only (2026-10-13 +08:00; no model fitted, no hypothesis changed)
- An independent review of the draft found that Figure 3 plotted the all-year mean learning curve while n*(g) is computed on the curve averaged over the test years U_g. src/fresh_by_gap.py recomputes the U_g curves (point estimates only) from the stored test predictions; with them the break-even sizes are reproduced to within 1.0 label per class (rounding of the stored curves). Figure 3 now uses these curves. Output: results/fresh_by_gap_<dataset>.json.
- The manuscript states the identity rho(g) = exp(-C(g)/s_g), which holds exactly under the log-linear interpolation when n*(g) lies in the top grid segment. Values of s_g and exp(-C/s_g) are computed in ms/values.py from the files above; they are explanatory, not estimates of new quantities.
- Replicate-1 values of M1-M4 (results_<dataset>_rep1.json, produced under X-04 for the like-for-like comparison with M5) are now quoted in the text.
- src/aggregate.py writes results/summary.csv from the result files; results/env.json records software versions (recorded 2026-10-13 in the unchanged container image).

## A-03 Added GPU models on D1 and D2 (registered 2026-10-13 14:22 +08:00, before any M5F or M6 selection or test score; M1-M5 results are final and are not changed)
Reason: the CPU-only machine forced a reduced MiniLM design (A-02) and no fine-tuned transformer. A GPU is now available. These models are added after the main results were known; the paper reports them as an added analysis.
- M5F: frozen all-MiniLM-L6-v2 exactly as in plan Section 4 (mean pooling, L2-normalised, 256 word pieces), transformers implementation checked on the published reference sentences; every sampled item encoded; logistic regression with C in {0.1, 1, 10}; five replicates; E1, E2 and E3 as for M1-M4. M5 (A-02) is kept and reported as before.
- M6: bert-base-uncased fine-tuned end to end (classification head on [CLS]); input = natural text (field nat), 128 word pieces; AdamW, weight decay 0.01, batch 32, linear warm-up over 10% of steps then linear decay; epochs = max(3, ceil(150 / steps per epoch)), at most 30; last epoch used, no early stopping; bf16 (fp16 if bf16 unsupported); seed from SHA-256 of (dataset, job kind, replicate, year, size, extra). Learning rate in {2e-5, 5e-5}, selected per (dataset, training size) on same-year validation data, replicate 1, ties to the smaller value. E1 and E2 with five replicates; E3 with replicate 1 only (cost).
- Selection and locking: validation-only; separate lock files results/LOCK_<dataset>_M5F.json and results/LOCK_<dataset>_M6.json written by select_lock.py lock_extra before any test prediction of that model.
- Scoring: evaluate.py with the unchanged bootstrap (2,000 resamples, seed 20261009), output files with suffix _A03 (five replicates; E3 summary omitted for M6 because its E3 has one replicate) and _rep1_A03e3 (M6 E3 and replicate-1 comparisons). Existing result files are not overwritten (evaluate.py refuses).
- Code changes made for this amendment (before any M5F/M6 score): run.py (M5F), select_lock.py (lock_extra), evaluate.py (--out suffix; E3 summary skipped when E3 replicates are incomplete; verified to reproduce results_stackexchange.json exactly), new src/gpu_models.py.
- Hypotheses H1-H3 remain defined for the primary model M1 only; M5F and M6 are reported descriptively with the same quantities.

## X-07 Execution note: bert-base-uncased obtained via read-only mirror (2026-10-13 14:57 +08:00)
Reason: the GPU server has no route to huggingface.co (huggingface_hub raised LocalEntryNotFoundError: ConnectError [Errno 101] Network is unreachable). The author approved the use of the read-only Hugging Face mirror hf-mirror.com. A second failure mode was met and fixed: the installed huggingface_hub enables the Xet content-addressed backend, whose LFS reconstruction request is sent to cas-server.xethub.hf.co (not mirrored, unauthenticated) and returned HTTP 401. Setting HF_HUB_DISABLE_XET=1 forces the ordinary <endpoint>/resolve/<rev>/<file> path, which the mirror serves.
- Source: repo google-bert/bert-base-uncased, revision 86b5e0934494bd15c9632b12f734a8a67f723594, fetched with HF_ENDPOINT=https://hf-mirror.com and HF_HUB_DISABLE_XET=1.
- Files: config.json, vocab.txt, tokenizer.json, tokenizer_config.json, model.safetensors (Apache-2.0).
- Integrity: model.safetensors SHA-256 = 68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3, identical to the official repository LFS hash; the other four files match the official sizes.
- No setting in Section 3 is changed. No M5F or M6 selection or test score existed when this note was written.

## X-08 Execution incident during A-03 (2026-10-13, Stack Exchange-10; entered in this log 2026-10-14)
Three consecutive steps of the A-03 run plan were pasted together on the server, so run.py select M5F started before encode had finished and select_lock.py lock_extra ran before select had finished. Both stopped with FileNotFoundError (exit 1). lock_extra reads every select run file before writing, so no lock file was created (verified absent); the steps were rerun in order and produced both locks with the registered counts. Recorded in the run notes of 2026-10-13; entered here so that the X numbering has no gap. No setting changed and no score existed when it happened.

## X-09 Execution note: A-03 results checked and two derived files added (2026-10-13 18:18 +08:00)
- The A-03 results package (paper5_gpu_results_20261013.zip, SHA-256 1da04c734dec8e94d1f6e5df07029dca3d4d8cc78fcb84383937f5cc07056782) was checked independently: the 32 result files that existed before A-03 are byte-identical to the earlier copies; every src/*.py file is identical to the code prepared for A-03; each A-03 lock reproduces from its select run files; each lock time precedes the first test run file of its model; re-running evaluate.py on the delivered test predictions reproduces results_*_A03.json and results_*_rep1_A03e3.json key by key and the four summary csv files byte for byte.
- results/summary_stackexchange_verify.csv is kept. It is byte-identical to summary_stackexchange.csv and is the evidence of the deterministic reproduction of the Stack Exchange scoring run before A-03; deleting it afterwards would be an unrecorded action.
- fresh_by_gap.py gained an A03 mode that writes results/fresh_by_gap_<dataset>_A03.json (M5F and M6, point estimates from the stored test predictions, no refitting; the M6 Reddit gap-6 curve reproduces n*(6) to within 1 label). aggregate.py gained an A03 mode that writes results/summary_with_A03.csv; results/summary.csv is unchanged.
- exp/requirements_gpu.txt (the environment used for A-03) is added to the repository.

## X-10 Presentation changes after pre-submission review (2026-10-14 +08:00)
No experiment was run and no result file changed. make_figures.py: Figure 1 uses a diverging colour scale centred at zero (negative losses in orange); Figures 2 and 4 label models with their codes (M1-M6); Figure 4 draws upper bounds censored at N as arrows instead of caps; Figure 3 shows the whole size grid from 25 labels per class; Figure 5 uses "old"/"current" instead of "stale"/"fresh". The source-data CSV files of Figures 1, 3 and 5 are byte-identical to the previous versions. The statement that every averaged current-year learning curve rises between adjacent grid sizes (Section 3.1) was checked on results/fresh_by_gap_<dataset>.json and results/fresh_by_gap_<dataset>_A03.json for all models and gaps.

## X-11 Added descriptive analysis: ratio for each single pair of years (2026-10-14 +08:00, after all results were known)
Reason: pre-submission reviews pointed out that the bootstrap intervals exclude variation between calendar years and that the six-year gap rests on one pair. src/year_pairs.py applies the registered interpolation (evaluate.breakeven) to each pair (u - g -> u) separately, using the stored E1/E2 test predictions of M1-M4, M5F and M6 averaged over five replicates. No model was refitted, no bootstrap was drawn and no existing result file changed. Outputs: results/year_pairs_<dataset>.json. Consistency check: the single gap-6 pair reproduces the reported rho(6) of every model (e.g. M1 0.6587 on Reddit-10, 0.6954 on Stack Exchange-10). The analysis is reported as descriptive and post hoc; it does not enter H1-H3.

## X-12 Added descriptive checks: replicate spread of the break-even size and hypotheses with replicate variation (2026-10-14 +08:00, after all results were known)
Reason: an independent pre-submission review noted that the intervals cover test sampling only. src/rep_spread.py recomputes n*(1) and n*(gmax) for each training replicate separately (same interpolation, U_g-averaged curves, stored test predictions, no refitting, no bootstrap); replicate 1 reproduces results_<dataset>_rep1.json. src/combined_check.py adds the replicate variance (divided by five) to the bootstrap variance for the H1 and H2 lower bounds (normal approximation; for H2 the replicate SD of the difference is bounded by the sum of the two SDs) and lists the smallest replicate n*(gmax) for H3. Outputs: results/rep_spread_<dataset>.json, results/combined_check.json. Descriptive only; the registered H1-H3 decisions are unchanged.
