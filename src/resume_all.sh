#!/bin/sh
# Relaunch every resumable background job (safe to run repeatedly; finished work is skipped).
cd "$(dirname "$0")"
pgrep -f "run.py reddit test M1 0 2" >/dev/null || nohup python3 run.py reddit test M1 0 2 >> ../logs/test_a.log 2>&1 &
pgrep -f "run.py reddit test M1 1 2" >/dev/null || nohup python3 run.py reddit test M1 1 2 >> ../logs/test_b.log 2>&1 &
pgrep -f "sh pipeline_se\.sh" >/dev/null || nohup sh pipeline_se.sh >> ../logs/pipeline_se.log 2>&1 &
sleep 1; pgrep -fa "run.py|pipeline_se" | cut -c1-80
