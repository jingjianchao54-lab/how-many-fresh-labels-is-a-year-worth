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
| `src/` | All code (Python standard library, NumPy, SciPy, scikit-learn) |
| `tests/` | Unit tests for the TF-IDF construction and the evaluation |
| `results/` | Data gate, validation selections, lock files, result files, `summary.csv`, `env.json`, sample indices |
| `figures/` | Figures 1–5 (PDF, PNG) and the source data of each figure (`*_data.csv`) |
| `predictions/test_predictions.zip` | Predictions of every test run (`work/runs/<dataset>/test/<model>/*.npz`) and the validation scores of every selection run (`work/runs/<dataset>/select/<model>/*.npz`) |
| `logs/a03/` | Logs and checks of the A-03 GPU runs (server paths replaced by `<server>`); `m6_jobs.csv` gives the time and epochs of every M6 job |
| `requirements_gpu.txt` | Package versions used for M5F and M6 |
| `manuscript_build/` | Template and scripts that insert every number of the manuscript from `results/` |
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

The same steps apply to `stackexchange` (half-size sensitivity: `HALF_N=200`). `src/pipeline_se.sh`, `src/chain_m5.sh` and `src/chain_sens.sh` are the scripts actually used to chain these steps. `src/apply_a02.py` is the one-off patch that applied amendment A-02 to `run.py` and `encode_minilm.py`; it is kept for the record. To check the stored results without rerunning the models, unzip `predictions/test_predictions.zip` into `work/` and run only `evaluate.py`.

## Manuscript numbers

`manuscript_build/values.py` reads `results/` and formats every number quoted in the paper; `build.py` fills `manuscript_template.md` with them.

## Licence

Code: MIT (see `LICENSE`). Results, figures and documentation: CC BY 4.0. The raw data remain under their own licences.

## Citation

If you use this code or these results, please cite the archived release:

Jing, J.; Lim, C.S.; Rana, M.E.; Wang, C.; Shang, X.; Liu, M.; Wang, M. *How Many Fresh Labels Is a Year Worth? Code, data indices and results*. Zenodo, 2026. https://doi.org/10.5281/zenodo.23621118

The DOI above always resolves to the latest version. The reference to the accompanying article will be added once it is published.
