#!/bin/sh
# Pre-registered sensitivity runs (plan E5), primary model only. Idempotent.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p ../work/runs/reddit/half ../work/runs/stackexchange/half
HALF=1 python3 run.py reddit test M1 0 1 tok
HALF=1 python3 run.py stackexchange test M1 0 1 tok
SKIP_E3=1 python3 run.py reddit test M1 0 1 tok_raw
echo CHAIN_SENS_DONE
