#!/bin/sh
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
[ -f ../work/minilm_stackexchange.npy ] || python3 encode_minilm.py stackexchange
[ -f ../work/minilm_reddit.npy ] || python3 encode_minilm.py reddit
echo ENCODE_DONE
