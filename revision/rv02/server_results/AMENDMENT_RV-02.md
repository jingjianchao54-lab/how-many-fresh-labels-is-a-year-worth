## RV-02 Added experiments requested in peer review: third domain (arXiv-10) and wider BERT grid (registered before any data of this amendment was downloaded or counted and before any M6G/M6L run)

Registered: 2026-11-19 13:18:34 +0800 (filled by the operator at registration; the SHA-256 of this file is recorded in RV02_registration.txt).
Reason: Reviewer 1 (comments 3 and 9) and Reviewer 2 (comment 6) of algorithms-4699623, Round 1. The registered results of Reddit-10 and Stack Exchange-10 are final and are not changed. H1–H3 keep their definitions; on arXiv-10 they are tested for the primary model M1 with the same rules, and reported as a replication, not as a test of the original hypotheses on new data chosen after the fact.

### 1. Part A — wider learning-rate grid for fine-tuned BERT (M6G) and longer input (M6L)
- M6G: the A-03 recipe unchanged except the learning-rate grid G6 = {1e-5, 2e-5, 3e-5, 5e-5, 8e-5, 1e-4}; selection on same-year validation data, replicate 1, mean Macro-F1 over training years, ties to the smaller value; lock file LOCK_<dataset>_M6G.json written before any M6G test run.
- One-step extension: if 1e-4 is selected at any size, the values {2e-4, 3e-4} are added once and the lock is computed over the extended grid. No further extension.
- If the M6G lock equals the A-03 M6 lock at every size, no M6G test run is made and the M6 results stand for M6G ("identical selection"). Otherwise E1 and E2 (five replicates) and E3 (replicate 1) are run as in A-03.
- M6L: the M6G recipe with input truncated at 256 word pieces instead of 128, learning rates from LOCK_<dataset>_M6G.json, E1 and E2 (five replicates), no E3, no separate selection.
- Seeds: SHA-256 of (dataset, job kind, replicate, year, size, extra, model) as in gpu_models.seed_of; select runs use the A-03 formula, so the values shared with M6 serve as a reproducibility check (target |ΔF1| ≤ 0.01 per run; a larger difference is recorded as a deviation and does not change the M6G selection, which uses only M6G runs).
- Datasets: Reddit-10 and Stack Exchange-10 (registered samples, unchanged).

### 2. Part B — third domain: arXiv-10
- Source: Hugging Face dataset librarian-bots/arxiv-metadata-snapshot (mirror of the arXiv OAI metadata snapshot, CC0 1.0), Parquet files under data/, fetched at one pinned commit through hf-mirror.com; the commit hash and the SHA-256 of every file are recorded in an execution note before any count is computed.
- Prior exposure: approximate public arXiv submission volumes by archive were known; no count or score from this snapshot existed at registration.
- Unit: one paper. Text: title + ". " + abstract, whitespace collapsed. Label: primary category, taken as the first entry of `categories` (checked against the arXiv API on 300 sampled papers; acceptance ≥ 95% agreement, otherwise STOP). Year: calendar year (UTC) of the v1 `created` date; years after 2025 excluded (incomplete).
- Author key: `submitter`, whitespace-normalised and case-folded. Split: registered hash (common.split_of) on "arxiv:" + key, thresholds 0.60/0.70. Cap: at most 5 papers per (submitter, category, year), the earliest by v1 date, then id.
- Exclusions, in order: missing id, submitter, title, abstract, categories or v1 date; primary category not a candidate; withdrawn (the word "withdrawn" in the first 300 characters of the abstract or in the comments); cross-listed to another candidate category; exact duplicate of (submitter, normalised title), first kept; fewer than 10 tokens; year > 2025.
- Candidate classes, list A (gated first): astro-ph.GA, cond-mat.mtrl-sci, cs.CL, cs.CV, hep-ph, hep-th, math.AP, math.CO, quant-ph, physics.optics. List B (used only if list A gives no decision): astro-ph.SR, cond-mat.str-el, cond-mat.stat-mech, gr-qc, math.PR, math.NA, math.OC, math.NT, cs.LG, stat.ME.
- Gate: the D1 rule of plan Section 3.4 applied unchanged (gate.reddit_gate): ladder (2000, 500, 1000), (1000, 250, 500), (500, 150, 300); all ten classes in at least 6 consecutive years, longest run at the largest size (ties: later run); fallback to at least 8 classes. If both lists fail, work on arXiv-10 stops.
- Text processing as plan 3.3 (lower-case, URLs removed, token pattern, 256-token truncation); no name masking.
- Sampling, replicates, sizes, models M1–M4, M5F and M6 (A-03 recipe), experiments E1–E4, selection, locking, bootstrap (2,000 resamples, seed 20261009) and Holm adjustment exactly as for Reddit-10. M5F and M6 are descriptive; the MiniLM pre-training data include scientific text and may overlap the study period.
- Environment: the project .venv on the GPU server (Python 3.10, versions in results/env_gpu.json); pyarrow added to read Parquet. Package versions are recorded in results/env_rv02.json.

### 3. Code added for this amendment (before any RV-02 data or run existed)
src/build_eligible_arxiv.py, src/gate_arxiv.py, src/build_sample_arxiv.py, src/gpu_models_rv.py; tools/download_arxiv.py, tools/env_rv02.py, tools/export_ids.py, tools/check_primary.py, tools/slim_sample.py, tools/selftest_rv02.py, tools/verify_rv02.py; the driver rv02.sh (all phases idempotent; no file is deleted). Registered code (common.py, gate.py, build_sample.py, run.py, features.py, evaluate.py, select_lock.py, gpu_models.py) is imported, not edited. A synthetic self-test (tools/selftest_rv02.py) was run before registration; it reads no real data.

### 4. Reporting
All RV-02 results are reported, including failures of the gate or of the acceptance checks. Deviations are written to results/RV02_DEVIATIONS.md with date, content and reason.
