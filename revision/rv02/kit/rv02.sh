#!/usr/bin/env bash
# RV-02 driver for aiserver2 (Paper 5, algorithms-4699623, Round 1).
# Every phase is idempotent: finished run files are skipped, locks and registrations are never overwritten,
# nothing is deleted. Usage (on the server):
#   bash rv02.sh <phase> [args]          run a short phase in the foreground
#   bash rv02.sh bg <phase> [args]       run a long phase inside tmux session rv02_<phase>; log in results/RV02_logs/
#   bash rv02.sh status                  progress of all RV-02 phases
# Phases, in order: discover backup install selftest deps register env smoke
#   Part A: a_select a_lock [a_extend a_lock] a_test
#   Part B: b_download b_eligible b_gate b_sample b_ids b_primary_accept
#           b_cpu_select b_lock b_cpu_test b_gpu_select b_gpu_lock b_gpu_test b_eval
#   end:    verify pack        (any time: deviation <text>)
set -uo pipefail
P="${P5_ROOT:-/home/aiadmin/jingjianchao/Paper5_LabelRecency_Algorithms}"
case "$P" in /home/aiadmin/jingjianchao/*|/tmp/*) ;; *) echo "STOP: project root outside jingjianchao: $P"; exit 2 ;; esac
KIT="$P/rv02_kit"
AXDIR="$P/rv02_data/arxiv"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
die() { echo "STOP: $*"; exit 2; }
ts() { date +%Y%m%d_%H%M%S; }

find_exp() {
  if [ -n "${EXP:-}" ]; then :; else
    mapfile -t c < <(find "$P" -maxdepth 4 -path '*/src/gpu_models.py' -not -path '*/.venv/*' -not -path "$P/rv02_*" \
                     -not -path "$P/tmp/*" -not -path '*backup*' 2>/dev/null)
    [ "${#c[@]}" -eq 1 ] || die "expected exactly one repository with src/gpu_models.py under $P, found ${#c[@]}: ${c[*]:-none}. Set EXP=<path> and rerun."
    EXP="$(dirname "$(dirname "${c[0]}")")"
  fi
  [ -f "$EXP/results/LOCK_reddit_M6.json" ] || die "$EXP/results/LOCK_reddit_M6.json missing: not the registered repository"
  SRC="$EXP/src"; RES="$EXP/results"; WORK="$EXP/work"; LOGS="$RES/RV02_logs"; mkdir -p "$LOGS"
  for v in "${VENV:-}" "$P/.venv" "$EXP/.venv" "$(dirname "$EXP")/.venv"; do
    [ -n "$v" ] && [ -f "$v/bin/activate" ] && { VENV="$v"; break; }
  done
  [ -n "${VENV:-}" ] && [ -f "$VENV/bin/activate" ] || die "no project .venv found (looked in $P/.venv, $EXP/.venv)"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  PY="$VENV/bin/python"
  cd "$SRC" || die "cannot cd $SRC"
}

idle_gpus() {   # GPUs with < 1000 MiB in use; override with GPUS="0 1"
  if [ -n "${GPUS:-}" ]; then echo "$GPUS"; return; fi
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2 < 1000 {printf "%s ", $1}'
}

cpu_workers() { # bounded by free memory (6 GB per worker) and by 32
  local free_gb; free_gb=$(free -g | awk '/^Mem:/ {print $7}')
  local n=$(( free_gb / 6 )); [ "$n" -gt "${NW_MAX:-32}" ] && n="${NW_MAX:-32}"; [ "$n" -lt 1 ] && n=1; echo "$n"
}

gpu_pool() {    # gpu_pool <command...> : one worker per idle GPU, worker i of n gets "i n" appended
  local gs; read -r -a gs <<< "$(idle_gpus)"
  [ "${#gs[@]}" -ge 1 ] || die "no idle GPU (all have >= 1000 MiB in use); wait and retry"
  echo "GPU workers on: ${gs[*]}"
  local i=0 pids=()
  for g in "${gs[@]}"; do OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="$g" "$@" "$i" "${#gs[@]}" & pids+=($!); i=$((i + 1)); done
  local rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ "$rc" -eq 0 ] || die "a GPU worker failed (see log); rerun the same phase: finished runs are skipped"
}

cpu_pool() {    # cpu_pool <command...> : n single-thread workers
  local n; n=$(cpu_workers); echo "CPU workers: $n"
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  local pids=() rc=0
  for ((i = 0; i < n; i++)); do "$@" "$i" "$n" >> "$LOGS/cpu_worker_$i.log" 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ "$rc" -eq 0 ] || { tail -n 20 "$LOGS"/cpu_worker_*.log; die "a CPU worker failed; rerun the same phase"; }
}

need_registered() { [ -f "$RES/RV02_registration.txt" ] || die "RV-02 not registered yet (run: bash rv02.sh register)"; }
need_gate() { "$PY" -c "import json,sys;d=json.load(open('$RES/gate.json')).get('arxiv',{}).get('decision');sys.exit(0 if d else 1)" \
              || die "arXiv gate has no decision: Part B stops here (report this)"; }
need_primary_ok() { [ -f "$RES/RV02_primary_check_result.tsv" ] && grep -q "PASS" "$RES/RV02_primary_check_status.txt" 2>/dev/null \
                    || die "primary-category check not accepted yet (results/RV02_primary_check_status.txt must contain PASS)"; }

phase="${1:-}"; shift || true
case "$phase" in
bg)
  find_exp; sub="${1:?phase}"; shift
  tmux has-session -t "rv02_$sub" 2>/dev/null && die "tmux session rv02_$sub already running"
  log="$LOGS/${sub}_$(ts).log"
  tmux new -d -s "rv02_$sub" "P5_ROOT='$P' EXP='$EXP' VENV='$VENV' GPUS='${GPUS:-}' NW_MAX='${NW_MAX:-32}' bash '$KIT/rv02.sh' $sub $* > '$log' 2>&1; echo \"EXIT_CODE=\$?\" >> '$log'"
  echo "started tmux session rv02_$sub; log: $log" ;;

status)
  find_exp
  echo "== tmux"; tmux ls 2>/dev/null | grep rv02_ || echo "(no rv02 session running)"
  echo "== last log lines"; for f in $(ls -t "$LOGS"/*.log 2>/dev/null | grep -v cpu_worker | head -6); do echo "--- $(basename "$f")"; tail -n 3 "$f"; done
  echo "== run files"
  for d in "$WORK"/runs/{reddit,stackexchange}/{select,test}/{M6G,M6L} "$WORK"/runs/arxiv/*/*; do
    [ -d "$d" ] && printf "%-70s %6s\n" "${d#$WORK/}" "$(find "$d" -name '*.npz' | wc -l)"
  done
  echo "== GPUs"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader ;;

discover)
  find_exp
  echo "P=$P"; echo "EXP=$EXP"; echo "VENV=$VENV"; "$PY" --version
  "$PY" - <<'PYEOF'
import importlib
for m in ("numpy", "scipy", "sklearn", "torch", "transformers", "huggingface_hub", "pyarrow"):
    try:
        print(f"{m:16s}", importlib.import_module(m).__version__)
    except Exception as e:
        print(f"{m:16s} MISSING ({e.__class__.__name__})")
try:
    import torch
    print("cuda devices", torch.cuda.device_count())
except Exception:
    pass
PYEOF
  echo "== locks"; ls -l "$RES"/LOCK_*.json
  echo "== samples"; ls -l "$WORK"/sample_*.pkl 2>/dev/null || echo "NO SAMPLES in $WORK"
  echo "== A-03 M6 select runs (needed for the reproduction check)"
  for ds in reddit stackexchange; do printf "%-14s %s\n" "$ds" "$(find "$WORK/runs/$ds/select/M6" -name '*.npz' 2>/dev/null | wc -l)"; done
  echo "== models"; ls -d "${MODELS_DIR:-$(dirname "$EXP")/models}"/* 2>/dev/null || echo "NO MODELS DIR"
  g="$(dirname "$EXP")/data/glove.6B.100d.txt"; [ -f "$g" ] && echo "GloVe present: $g ($(stat -c %s "$g") bytes)" || echo "GloVe MISSING: $g"
  [ -f "$EXP/AMENDMENTS.md" ] && echo "AMENDMENTS.md present" || echo "AMENDMENTS.md MISSING"
  [ -f "$RES/RV02_registration.txt" ] && echo "RV-02 ALREADY REGISTERED" || echo "RV-02 not registered"
  echo "== disk / memory"; df -h "$P" | tail -1; free -g | head -2
  echo "== GPUs"; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
  echo "idle GPUs: $(idle_gpus)"; echo "== tmux"; tmux ls 2>/dev/null || echo "(none)" ;;

backup)
  find_exp; mkdir -p "$P/rv02_backup"; b="$P/rv02_backup/results_before_rv02_$(ts).tar.gz"
  tar -czf "$b" -C "$EXP" results AMENDMENTS.md src && sha256sum "$b" | tee "$b.sha256"; ls -l "$b" ;;

install)
  find_exp
  for f in "$KIT"/src/*.py; do
    t="$SRC/$(basename "$f")"
    if [ -f "$t" ]; then cmp -s "$f" "$t" && echo "same: $(basename "$f")" || die "$(basename "$f") already in src/ with different content; not overwritten"
    else cp "$f" "$t" && echo "installed: $(basename "$f")"; fi
  done ;;

selftest)
  find_exp; mkdir -p "$P/tmp"; "$PY" "$KIT/tools/selftest_rv02.py" "$EXP" "$KIT" "$P/tmp" 2>&1 | tail -n 5 ;;

deps)
  find_exp
  "$PY" -c "import pyarrow" 2>/dev/null && { echo "pyarrow present: $("$PY" -c 'import pyarrow;print(pyarrow.__version__)')"; exit 0; }
  "$PY" -m pip install "pyarrow>=14" && "$PY" -c "import pyarrow;print('pyarrow', pyarrow.__version__)" ;;

register)
  find_exp
  [ -f "$RES/RV02_registration.txt" ] && die "already registered: $RES/RV02_registration.txt"
  now="$(date '+%Y-%m-%d %H:%M:%S %z')"
  sed "s/<REGISTRATION_TIMESTAMP>/$now/" "$KIT/AMENDMENT_RV-02.md" > "$RES/AMENDMENT_RV-02.md"
  { echo; echo; cat "$RES/AMENDMENT_RV-02.md"; } >> "$EXP/AMENDMENTS.md"
  { echo "RV-02 registered at $now on $(hostname)"; echo "== amendment, kit and registered code (sha256)"
    sha256sum "$RES/AMENDMENT_RV-02.md" "$KIT"/src/*.py "$KIT"/tools/*.py "$KIT/rv02.sh" \
              "$SRC"/{common,gate,build_sample,run,features,evaluate,select_lock,gpu_models}.py
    echo "== state at registration: arXiv data present? $(ls "$AXDIR"/data 2>/dev/null | wc -l) files; M6G runs: $(find "$WORK/runs" -path '*M6G*' -name '*.npz' 2>/dev/null | wc -l)"
  } > "$RES/RV02_registration.txt"
  cat "$RES/RV02_registration.txt" ;;

env)
  find_exp; "$PY" "$KIT/tools/env_rv02.py" "$RES" | head -n 30 ;;

smoke)
  find_exp; need_registered
  g="$(idle_gpus | awk '{print $1}')"; [ -n "$g" ] || die "no idle GPU"
  CUDA_VISIBLE_DEVICES="$g" "$PY" gpu_models_rv.py smoke reddit && echo "SMOKE OK (files in work/smoke_rv02; not used in any result)" ;;

# ---------------- Part A: M6G (wider grid) and M6L (256 word pieces) ----------------
a_select)
  find_exp; need_registered
  for ds in reddit stackexchange; do echo "== $ds"; gpu_pool "$PY" gpu_models_rv.py select M6G "$ds"; done
  echo "A_SELECT DONE" ;;

a_extend)
  find_exp; need_registered
  for ds in reddit stackexchange; do
    [ -f "$WORK/runs/$ds/select/M6G/EXTENDED" ] && { echo "== $ds (extension)"; gpu_pool "$PY" gpu_models_rv.py extend M6G "$ds"; }
  done; echo "A_EXTEND DONE" ;;

a_lock)
  find_exp; need_registered
  for ds in reddit stackexchange; do
    echo "== $ds"
    if [ -f "$RES/LOCK_${ds}_M6G.json" ]; then echo "lock exists"; "$PY" -c "import json;l=json.load(open('$RES/LOCK_${ds}_M6G.json'));print(l['selected'], 'same_as_M6_lock =', l['same_as_M6_lock'])"
    else "$PY" gpu_models_rv.py lock M6G "$ds" || exit 2; fi
  done ;;

a_test)
  find_exp; need_registered
  for ds in reddit stackexchange; do
    [ -f "$RES/LOCK_${ds}_M6G.json" ] || die "LOCK_${ds}_M6G.json missing (an EDGE needs a_extend then a_lock)"
    same=$("$PY" -c "import json;print(json.load(open('$RES/LOCK_${ds}_M6G.json'))['same_as_M6_lock'])")
    if [ "$same" = "True" ]; then echo "== $ds: M6G lock identical to M6 lock -> no M6G test runs (registered rule)"
    else echo "== $ds: M6G test"; gpu_pool "$PY" gpu_models_rv.py test M6G "$ds"; fi
    echo "== $ds: M6L test"; gpu_pool "$PY" gpu_models_rv.py test M6L "$ds"
  done; echo "A_TEST DONE" ;;

# ---------------- Part B: arXiv-10 ----------------
b_download)
  find_exp; need_registered; mkdir -p "$AXDIR"
  if ls "$AXDIR"/data/*.parquet >/dev/null 2>&1; then echo "parquet already present"; cat "$RES/RV02_arxiv_snapshot.json" | head -5
  else "$PY" "$KIT/tools/download_arxiv.py" "$AXDIR" "$RES"; fi
  du -sh "$AXDIR"; ls "$AXDIR"/data | head; echo "B_DOWNLOAD DONE" ;;

b_eligible)
  find_exp; need_registered
  [ -f "$RES/RV02_arxiv_snapshot.json" ] || die "snapshot manifest missing: run b_download first"
  "$PY" build_eligible_arxiv.py "$AXDIR/data" && cat "$WORK/eligible/ax_counts_total.json" | head -c 1500; echo; echo "B_ELIGIBLE DONE" ;;

b_gate)
  find_exp; need_registered; "$PY" gate_arxiv.py; need_gate
  "$PY" -c "import json;print(json.dumps(json.load(open('$RES/gate.json'))['arxiv']['decision']))" ;;

b_sample)
  find_exp; need_registered; need_gate
  [ -f "$WORK/sample_arxiv.pkl" ] && { echo "sample exists"; exit 0; }
  "$PY" build_sample_arxiv.py "$AXDIR/data" && echo "B_SAMPLE DONE" ;;

b_ids)
  find_exp; need_registered; "$PY" "$KIT/tools/export_ids.py" && wc -l "$RES/RV02_primary_check_ids.tsv" ;;

b_primary_accept)   # after results/RV02_primary_check_result.tsv was uploaded from the local check
  find_exp; need_registered
  f="$RES/RV02_primary_check_result.tsv"; [ -f "$f" ] || die "upload RV02_primary_check_result.tsv to $RES first"
  "$PY" - "$f" "$RES/RV02_primary_check_status.txt" <<'PYEOF'
import sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
nf = sum(r[2] == "NOT_FOUND" for r in rows)
found = [r for r in rows if r[2] != "NOT_FOUND"]
agree = sum(r[2] == r[1] for r in found)
rate = agree / len(found) if found else 0.0
ok = len(rows) == 300 and len(found) >= 285 and rate >= 0.95
msg = f"{'PASS' if ok else 'FAIL'}: agreement {agree}/{len(found)} = {rate:.3f} (not found by the API: {nf}); acceptance >= 0.95 with >= 285 found"
open(sys.argv[2], "w").write(msg + "\n")
print(msg)
PYEOF
  ;;

b_cpu_select)
  find_exp; need_registered; need_gate; need_primary_ok
  [ -f "$(dirname "$EXP")/data/glove.6B.100d.txt" ] || die "GloVe missing: upload glove.6B.100d.txt first"
  echo "warm-up: building the shared count matrix and GloVe means once"
  "$PY" -c "from run import Data; d = Data('arxiv'); print(d.x.shape, d.g.shape)" || die "warm-up failed"
  for m in M1 M2 M3 M4; do echo "== select $m"; cpu_pool "$PY" run.py arxiv select "$m"; done; echo "B_CPU_SELECT DONE" ;;

b_lock)
  find_exp; need_registered
  [ -f "$RES/LOCK_arxiv.json" ] && { echo "LOCK_arxiv.json exists"; head -c 600 "$RES/LOCK_arxiv.json"; exit 0; }
  "$PY" select_lock.py arxiv select && "$PY" select_lock.py arxiv lock && head -c 800 "$RES/LOCK_arxiv.json" ;;

b_cpu_test)
  find_exp; need_registered; [ -f "$RES/LOCK_arxiv.json" ] || die "LOCK_arxiv.json missing"
  for m in M1 M2 M3 M4; do echo "== test $m"; cpu_pool "$PY" run.py arxiv test "$m"; done; echo "B_CPU_TEST DONE" ;;

b_gpu_select)
  find_exp; need_registered; [ -f "$RES/LOCK_arxiv.json" ] || die "run b_lock first"
  if [ ! -f "$WORK/minilm_arxiv_full.npy" ]; then
    g="$(idle_gpus | awk '{print $1}')"; [ -n "$g" ] || die "no idle GPU"
    CUDA_VISIBLE_DEVICES="$g" "$PY" gpu_models.py encode arxiv || die "MiniLM encoding failed"
  fi
  echo "== M5F select (CPU)"; cpu_pool "$PY" run.py arxiv select M5F
  echo "== M6 select (GPU)"; gpu_pool "$PY" gpu_models.py bert arxiv select; echo "B_GPU_SELECT DONE" ;;

b_gpu_lock)
  find_exp; need_registered
  for m in M5F M6; do
    if [ -f "$RES/LOCK_arxiv_$m.json" ]; then echo "LOCK_arxiv_$m.json exists"; else "$PY" select_lock.py arxiv lock_extra "$m" || exit 2; fi
    head -c 400 "$RES/LOCK_arxiv_$m.json"; echo
  done ;;

b_gpu_test)
  find_exp; need_registered
  [ -f "$RES/LOCK_arxiv_M5F.json" ] && [ -f "$RES/LOCK_arxiv_M6.json" ] || die "run b_gpu_lock first"
  echo "== M5F test (CPU)"; cpu_pool "$PY" run.py arxiv test M5F
  echo "== M6 test (GPU)"; gpu_pool "$PY" gpu_models.py bert arxiv test; echo "B_GPU_TEST DONE" ;;

b_eval)
  find_exp; need_registered
  "$PY" evaluate.py arxiv test M1 M2 M3 M4 || die "evaluate M1-M4 failed"
  "$PY" evaluate.py arxiv test M5F M6 M1 --out A03 || echo "WARN: A03 scoring failed (scored locally instead)"
  "$PY" evaluate.py arxiv test M6 M5F M1 --reps1 --out A03e3 || echo "WARN: A03e3 scoring failed (scored locally instead)"
  for s in "drift.py arxiv" "fresh_by_gap.py arxiv" "fresh_by_gap.py arxiv A03" "year_pairs.py arxiv" "rep_spread.py arxiv"; do
    # shellcheck disable=SC2086
    "$PY" $s > "$LOGS/eval_${s// /_}.log" 2>&1 && echo "ok: $s" || echo "WARN: $s failed (descriptive; scored locally instead)"
  done
  "$PY" "$KIT/tools/slim_sample.py"; echo "B_EVAL DONE" ;;

deviation)    # bash rv02.sh deviation <plain text without quotes>
  find_exp; [ -f "$RES/RV02_DEVIATIONS.md" ] || printf "# RV-02 deviations (date | content | reason)\n\n" > "$RES/RV02_DEVIATIONS.md"
  echo "- $(date '+%Y-%m-%d %H:%M %z') | $*" >> "$RES/RV02_DEVIATIONS.md"; tail -n 3 "$RES/RV02_DEVIATIONS.md" ;;

# ---------------- end ----------------
verify)
  find_exp; "$PY" "$KIT/tools/verify_rv02.py" | tee "$RES/RV02_verify.txt" ;;

pack)
  find_exp; out="$P/rv02_return"; mkdir -p "$out"; f="$out/RV02_return_$(ts).tar.gz"
  [ -f "$WORK/sample_arxiv.pkl" ] && [ ! -f "$WORK/sample_arxiv_slim.pkl" ] && "$PY" "$KIT/tools/slim_sample.py"
  cd "$EXP" || exit 2
  list=$(ls -d results/RV02_* results/AMENDMENT_RV-02.md results/LOCK_*_M6G.json results/LOCK_arxiv*.json results/gate.json \
           results/env_rv02.json results/*arxiv* AMENDMENTS.md work/eligible/ax_counts_total.json work/eligible/ax_index.json \
           work/eligible/ax_*.counts.json work/sample_arxiv_slim.pkl \
           work/runs/reddit/*/M6G work/runs/reddit/*/M6L work/runs/stackexchange/*/M6G work/runs/stackexchange/*/M6L \
           work/runs/arxiv 2>/dev/null | sort -u)
  # shellcheck disable=SC2086
  tar -czf "$f" $list && sha256sum "$f" | tee "$f.sha256" && ls -l "$f" && echo "PACK: $f" ;;

*) sed -n '2,13p' "$0"; exit 1 ;;
esac
