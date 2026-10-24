#!/bin/sh
# After encoding: M5 validation selection -> separate lock -> M5 test predictions, for each dataset. Idempotent.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for ds in stackexchange reddit; do
  [ -f ../work/minilm_$ds.npy ] || python3 encode_minilm.py $ds
  python3 run.py $ds select M5
  [ -f ../results/LOCK_${ds}_M5.json ] || python3 select_lock.py $ds lock_m5
  python3 run.py $ds test M5
done
echo CHAIN_M5_DONE
