# How Many Fresh Labels Is a Year Worth?

Code, analysis plan, amendment log, locked configurations, sample indices, test predictions and results for the paper

> *How Many Fresh Labels Is a Year Worth? Measuring the Exchange Rate between Label Recency and Label Quantity in Text Classification* (submitted to *Algorithms*, MDPI).

The study estimates the break-even size *n\**(*g*), the number of current labels per class that matches a classifier trained on *N* labels per class collected *g* years earlier, and the label-equivalence ratio *ρ*(*g*) = *n\**(*g*)/*N*. It uses two ten-class community classification tasks (Reddit, 2012–2018; Stack Exchange, 2015–2021) and five lightweight classifiers (M1–M5). Two models were added under amendment A-03 after the main results were known: MiniLM with the full planned design (M5F) and a fine-tuned bert-base-uncased (M6).

## Repository layout

| Path | Content |
|---|---|
| `analysis_plan.md` | Analysis plan v1.0, frozen before any per-year count or score was computed. SHA-256 `283db2c2fffdc81c320dd113edae05693043d07dbbe7413819b8aab1a093003e` |
| `PLAN_FREEZE.json` | Freeze record of the plan |
| `AMENDMENTS.md` | Time-stamped amendments (A-), data-gate outcomes (R-), implementation notes (N-) and execution records (X-) |
| `src/` | Original study code (Python standard library, NumPy, SciPy, scikit-learn) |
| `tests/` | Unit tests for the TF-IDF construction and the evaluation |
| `results/` | Data gate, validation selections, lock files, result files, `summary.csv`, `env.json`, sample indices |
| `figures/` | Figures 1–5 (PDF, PNG) and the source data of each figure (`*_data.csv`) |
| `predictions/test_predictions.zip` | Original community prediction archive, including test runs (`work/runs/<dataset>/test/<model>/*.npz`) and stored selection runs (`work/runs/<dataset>/select/<model>/*.npz`) |
| `logs/a03/` | Logs and checks of the A-03 GPU runs (server paths replaced by `<server>`); `m6_jobs.csv` gives the time and epochs of every M6 job |
| `requirements_gpu.txt` | Package versions used for M5F and M6 |
| `manuscript_build/` | Numeric formatting helpers for `results/` |
| `data_manifest/` | SHA-256 and byte sizes of the raw data and embedding files |

## Data

No text is redistributed. Place the raw files in a folder `data/` next to this repository folder (the code looks for `../data`).

- **Reddit-10**: ConvoKit subreddit corpora (`subreddit-<name>`) for chess, Coffee, Cooking, gardening, history, investing, languagelearning, philosophy, running and woodworking, saved as `data/reddit/<name>.corpus.zip`. The files used are listed with SHA-256 in `data_manifest/reddit_corpora_sha256.txt`.
- **Stack Exchange-10**: Stack Exchange data dump of April 2024 on the Internet Archive (item `stackexchange`), files `<site>.stackexchange.com.7z` for philosophy, photo, music, history, cooking, movies, bicycles, boardgames, gardening and chess, saved as `data/stackexchange/`. Licence CC BY-SA. Hashes in `data_manifest/stackexchange_sha256.txt`.
- **GloVe** 6B 100d vectors as `data/glove.6B.100d.txt`; **all-MiniLM-L6-v2** (`model.safetensors`, `vocab.txt`, `config.json`) as `data/minilm/`. Hashes in `data_manifest/embeddings_sha256.txt`.
- For step 6, the Hugging Face model folders `all-MiniLM-L6-v2` and `bert-base-uncased` (revision `86b5e0934494bd15c9632b12f734a8a67f723594`, `model.safetensors` SHA-256 `68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`) in `../models/`.

`results/sample_index_<dataset>.csv.gz` lists every sampled item by identifier (Reddit utterance id and line number; Stack Exchange post or comment id), year and part (0 training pool, 1 validation, 2 test).

## Reproducing the results

Requirements: Python 3.13 with the versions in `requirements.txt` (recorded in `results/env.json`). Steps 1–5 need no GPU; M5 is run with the NumPy implementation in `src/minilm_np.py`. Step 6 (M5F and M6) needs a CUDA GPU and the versions in `requirements_gpu.txt` (recorded in `results/env_gpu.json`; four RTX 4090 were used).

```sh
cd src
# 1. data gate (counts only) and samples
python build_eligible.py && python build_eligible_se.py
python gate.py && python gate.py stackexchange
python build_sample.py reddit && python build_sample.py stackexchange
# 2. validation-only selection, then the lock
python run.py reddit select M1   # repeat for M2-M4 and for stackexchange
python select_lock.py reddit select && python select_lock.py reddit lock
# 3. test predictions and one scoring
python run.py reddit test M1     # repeat for M2-M4
python evaluate.py reddit test M1 M2 M3 M4
python drift.py reddit
# 4. MiniLM (reduced design, amendment A-02)
python encode_minilm.py check && python encode_minilm.py reddit
python run.py reddit select M5 && python select_lock.py reddit lock_m5 && python run.py reddit test M5
python evaluate.py reddit test M5 M1 M2 M3 M4 --reps1
# 5. sensitivity analyses, per-gap curves, figures, summary
HALF=1 SKIP_E3=1 python run.py reddit test M1 && python sens_half.py reddit
python fresh_by_gap.py reddit && python make_figures.py && python aggregate.py
# 6. GPU models (amendment A-03); model folders in ../models/ (or set MODELS_DIR)
python gpu_models.py check && python gpu_models.py encode reddit
python run.py reddit select M5F && python gpu_models.py bert reddit select
python select_lock.py reddit lock_extra M5F && python select_lock.py reddit lock_extra M6
python run.py reddit test M5F && python gpu_models.py bert reddit test      # optional: add <worker> <n_workers>
python evaluate.py reddit test M5F M6 M1 --out A03
python evaluate.py reddit test M6 M5F M1 --reps1 --out A03e3
python fresh_by_gap.py reddit A03 && python aggregate.py A03
# 7. ratio for each single pair of years (X-11, descriptive; uses stored predictions only)
python year_pairs.py reddit && python year_pairs.py stackexchange
# 8. replicate spread of n* and hypotheses with replicate variation (X-12, descriptive; stored predictions only)
python rep_spread.py reddit && python rep_spread.py stackexchange && python combined_check.py && python aggregate.py A03
```

The same steps apply to `stackexchange` (half-size sensitivity: `HALF_N=200`). `src/pipeline_se.sh`, `src/chain_m5.sh` and `src/chain_sens.sh` are the scripts actually used to chain these steps. `src/apply_a02.py` is the one-off patch that applied amendment A-02 to `run.py` and `encode_minilm.py`; it is kept for the record. To check the stored results without rerunning the models, unzip `predictions/test_predictions.zip` at the repository root and run only `evaluate.py`.

## R1 release inventory

The returned RV-02 archive supplies 4,224 per-run NPZ files in `revision/rv02/work/runs/`. The arXiv task contributes 680 selection and 2,032 test outputs. Reddit contributes 392 selection and 532 test outputs. Stack Exchange contributes 210 selection and 378 test outputs. Selection scores are distinct from test predictions. Individual arrays reflect the run type and schedule. The pre-existing `predictions/test_predictions.zip` is a separate community archive (4,809 test, 1,680 selection and 273 dry-run NPZ files). File-level hashes appear in `SHA256SUMS.txt`.

RV-01 outputs and selections are in `revision/out/` and `revision/selections/`. RV-02 server records and locally scored summaries are in `revision/rv02/server_results/` and `revision/rv02/scored/`. The text-free arXiv index contains the original row order and numeric class, year and split codes. Consult the frozen gate for category order. Splits are disjoint by submitting author, and the snapshot is dated 2026-11-19. The two community indices are in `results/`. Full manuscripts, responses and raw post/abstract text are excluded.

All nine revised figures and their CSVs are in `revision/figures/`. The `figures/` directory preserves the five original figures. Run the two plotting scripts in `revision/` to draw the revised figures from the stored outputs. Figure 5 CSV gives the plotted curves; the shaded band uses the frozen paired interval for the gain of combined over old labels alone in the six-year M1 mixing results, added to the old-label score. Figure 8 CSV retains finite lower bounds for censored point estimates, drawn as open triangles in the ratio panels. Figure 9 CSV panel b records the break-even size in labels per class; divide by N = 500 to obtain the plotted ratio. A right-censored estimate is indicated by an open triangle.

For the five-draw variation in Section 4.7, `revision/rv02/replicate_spread.py` reads the returned community predictions and the sample indices. It computes sample standard deviations with denominator four. No model or bootstrap is executed by this script.

The historical RV-02 kit digest records its original freeze. Machine-specific launch instructions and private execution diaries are excluded from this distribution. Use the package manifest to verify distributed files.

## Licence

Code: MIT (see `LICENSE`). Results, figures and documentation: CC BY 4.0. The raw data remain under their own licences.

## Citation

Please cite the paper once published. Citation details will be added here.
