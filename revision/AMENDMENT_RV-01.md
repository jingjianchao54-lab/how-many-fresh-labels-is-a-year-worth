# Amendment RV-01 — analyses added in response to peer review (Round 1, algorithms-4699623)

Recorded: 2026-11-18, after all registered results were known. Status: added analysis, reported descriptively.
The registered hypotheses (H1–H3), splits, replicate draws, bootstrap seed (20261009) and result files are unchanged.

## Scope

| Id | Analysis | Reviewer comment | Data used | Script | Output |
|:--|:--|:--|:--|:--|:--|
| RV-01a | Author-clustered stratified bootstrap (authors resampled within class–year test cells), all models | R2-4 | stored test predictions + author hash | `b1_cluster.py` | `out/cluster_<ds>.json` |
| RV-01b | Class-level staleness cost and break-even size (M1, M6) | R1-8 | stored test predictions | `a_analyses.py classwise` | `out/classwise_<ds>.json` |
| RV-01c | Five estimators of the break-even size (registered log-linear interpolation, linear interpolation, inverse power law on all sizes, single log-linear fit, inverse power law on sizes ≤ N/4 extrapolated), all models | R1-6, R1-7 | stored test predictions | `a_analyses.py curves` | `out/curves_<ds>.json` |
| RV-01d | Confusion-matrix change, stale vs same-year model (M1, M6) | R1-4 | stored test predictions | `a_analyses.py confusion` | `out/confusion_<ds>.json` |
| RV-01e | Stack Exchange scores by test-item type (posts vs comments) | R2-6 | stored test predictions | `a_analyses.py posttype` | `out/posttype_stackexchange.json` |
| RV-01f | Language identification (langid.py 1.1.6) and English-only scoring | R2-2 | stored test predictions + text | `lang_id.py`, `b_eval.py <ds> english english` | `out/lang_<ds>.json`, `out/variant_english_english_<ds>.json` |
| RV-01g | Refits of M1: personal identifiers scrubbed; emoji/emoticons kept; Stack Exchange site names masked | R2-1, R2-3, R2-7 | registered splits and replicate draws | `b_runs.py`, `b_eval.py` | `out/variant_{pii,emoji,sitemask}_<ds>.json` |
| RV-01h | Refits of M1: C fixed at the value for N (E2); six-value C grid (selection + test if selection changes) | R1-9, R2-7 | registered splits | `b_runs.py`, `b_select.py`, `b_eval.py` | `out/variant_fixedC_<ds>.json`, `rwork/widegrid/selection_<ds>.json` |
| RV-01i | E3 at g = 6 for M1: equal total weight of old and current labels; C re-selected for the combined set on same-year validation data | R2-8 | registered splits | `b_runs.py`, `b_select.py`, `b_eval.py` | `out/variant_{mixw,mixtune}_<ds>.json` |
| RV-01j | Corpus descriptives: class shares, lengths, truncation shares, post-type shares | R2-6, R2-8 | sample text | `a5_descriptives.py` | `out/descriptives.json` |

## Integrity checks

- Rebuilding the sample from the public index and the raw corpora reproduced the registered sample file; refitting M1 on the registered text reproduced the stored test predictions exactly in all 24 runs checked (`out/check_runs.json`).
- Rescoring the stored predictions with the revision scorer reproduced the locked results of M1 exactly (Stack Exchange-10, all fields).
- Author keys are stored only as salted SHA-256 hashes (16 hex characters); no text is released.

## Effect on reported results

None of the registered numbers changed. Section 4.7, Tables 1, 8–10 and A3, Algorithm 1 and Figures 6–8 of the revised manuscript are new.
