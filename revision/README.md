# revision/ — analyses added in Round 1 of peer review (amendment RV-01)

See `AMENDMENT_RV-01.md` for the list of analyses, the reviewer comments they answer and the files they write.

Order of execution (Python 3.13, NumPy, SciPy, scikit-learn as in `results/env.json`):

1. `build_lite_sample.py` — labels, years and parts from the public sample index (enough for all analyses on stored predictions).
2. `a_analyses.py <dataset> {curves,classwise,confusion,posttype} [models]`, `b1_cluster.py <dataset>` and `a5_descriptives.py` need the stored predictions (`predictions/test_predictions.zip` unpacked to `work/runs/`); `b1_cluster.py`, `a5_descriptives.py` and `lang_id.py` also need `full/sample_<dataset>.pkl`, rebuilt from the raw corpora with `extract_from_index.py <data_dir> full`. That file contains text and is not released.
3. `b_runs.py <dataset> <variant> {select,test}`, `b_select.py`, then `b_eval.py <dataset> <variant> [mask]` for the refits of M1.
4. `make_figures_rev.py` draws Figures 1–8 of the revised manuscript (PDF with TrueType fonts, SVG, 600-dpi PNG, source data CSV).

`lang_id.py` needs langid.py 1.1.6 (set `LANGID_DIR` to the unpacked source folder). All outputs are in `out/`; the selections of the wide grid and of the re-tuned mixing are in `selections/`.
