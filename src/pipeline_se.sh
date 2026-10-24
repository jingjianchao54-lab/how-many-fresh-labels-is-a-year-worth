#!/bin/sh
# Idempotent Stack Exchange pipeline: validation-only selection -> lock (never overwritten) -> test predictions.
cd "$(dirname "$0")"
python3 -c "from features import global_counts, glove_means; x=global_counts('stackexchange'); g=glove_means('stackexchange'); print(x.shape, g.shape)"
for m in M3 M2 M4 M1; do python3 run.py stackexchange select $m; done
[ -f ../results/selection_stackexchange.json ] || python3 select_lock.py stackexchange select M1 M2 M3 M4
[ -f ../results/LOCK_stackexchange.json ] || python3 select_lock.py stackexchange lock
for m in M3 M2 M4 M1; do python3 run.py stackexchange test $m; done
echo PIPELINE_SE_DONE
