"""RV-02 acceptance checks; run on the server from exp/src:  bash rv02.sh verify

Checks (each prints PASS / FAIL / SKIP):
  1 run maps     every job the code defines exists on disk, no extra files, for each (dataset, stage, model) present
  2 lock order   each lock file was written before the first test run file of that model (file mtimes)
  3 reproduce    M6G select runs at learning rates shared with M6 reproduce the A-03 validation F1 (|diff| <= 0.01; WARN only)
  4 finite       every run file has ok == True
  5 arxiv sample the sample size equals classes x years x (2N + V + T)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from common import RESULTS, WORK  # noqa: E402
from run import GRID, Data, select_jobs, test_jobs  # noqa: E402
import gpu_models_rv as rv  # noqa: E402

fails = 0


def report(name: str, ok: bool | None, msg: str, warn_only: bool = False) -> None:
    global fails
    tag = "SKIP" if ok is None else ("PASS" if ok else ("WARN" if warn_only else "FAIL"))
    if warn_only:
        ok = None if ok is False else ok
    fails += ok is False
    print(f"[{tag}] {name}: {msg}", flush=True)


def expected(ds: str, stage: str, model: str, d: Data) -> set[str]:
    if stage == "select":
        grid = GRID.get(model) or (rv.grid_for(ds, model) if model in ("M6G",) else __import__("gpu_models").GRID_M6)
        return {"_".join(map(str, j)) for j in [("sel", 1, t, n, h) for t in d.years for n in d.sizes for h in grid]}
    jobs = test_jobs(d)
    if model in ("M6", "M6G"):
        jobs = [j for j in jobs if not (j[0] == "e3" and j[1] != 1)]
    if model == "M6L":
        jobs = [j for j in jobs if j[0] != "e3"]
    return {"_".join(map(str, j)) for j in jobs}


for ds in ("reddit", "stackexchange", "arxiv"):
    if not (WORK / f"sample_{ds}.pkl").exists():
        report(f"{ds}", None, "no sample file"); continue
    d = Data(ds)
    models = ["M6G", "M6L"] + (["M1", "M2", "M3", "M4", "M5F", "M6"] if ds == "arxiv" else [])
    for stage in ("select", "test"):
        for m in models:
            base = WORK / "runs" / ds / stage / m
            if not base.exists():
                report(f"run map {ds}/{stage}/{m}", None, "not run"); continue
            if stage == "select" and m == "M6L":
                continue
            have = {p.stem for p in base.glob("*.npz")}
            exp = expected(ds, stage, m, d)
            miss, extra = exp - have, have - exp
            report(f"run map {ds}/{stage}/{m}", not miss and not extra, f"expected={len(exp)} on_disk={len(have)} missing={len(miss)} extra={len(extra)}")
            bad, broken = [], []
            for p in base.glob("*.npz"):
                try:
                    if not bool(np.load(p)["ok"]):
                        bad.append(p.name)
                except Exception:  # noqa: BLE001  truncated file from an interrupted job
                    broken.append(p.name)
            report(f"finite {ds}/{stage}/{m}", not bad, f"{len(bad)} runs with ok=False {bad[:5]}")
            report(f"readable {ds}/{stage}/{m}", not broken, f"{len(broken)} unreadable files {broken[:5]}")
    for m in models:
        tdir = WORK / "runs" / ds / "test" / m
        lk = RESULTS / (f"LOCK_{ds}.json" if (ds == "arxiv" and m in ("M1", "M2", "M3", "M4")) else f"LOCK_{ds}_{'M6G' if m == 'M6L' else m}.json")
        if not tdir.exists() or not lk.exists():
            continue
        first = min(p.stat().st_mtime for p in tdir.glob("*.npz"))
        report(f"lock order {ds}/{m}", lk.stat().st_mtime < first, f"lock {lk.name} before first test file")
    s6 = WORK / "runs" / ds / "select" / "M6G"
    s0 = WORK / "runs" / ds / "select" / "M6"
    if s6.exists() and s0.exists():
        diffs = []
        for p in s6.glob("*.npz"):
            q = s0 / p.name
            if q.exists():
                diffs.append(abs(float(np.load(p)["f1"]) - float(np.load(q)["f1"])))
        report(f"reproduce M6 select {ds}", bool(diffs) and max(diffs) <= 0.01,
               f"{len(diffs)} shared runs, max |dF1| = {max(diffs) if diffs else float('nan'):.4f}, mean = {np.mean(diffs) if diffs else float('nan'):.4f}"
               " (WARN = record in RV02_DEVIATIONS.md; does not stop the work)", warn_only=True)
    if ds == "arxiv":
        n, v, t = d.N, d.V, d.T
        report("arxiv sample size", len(d.cls) == d.K * len(d.years) * (2 * n + v + t), f"{len(d.cls)} items, K={d.K}, years={d.years}, size={(n, v, t)}")
print("SUMMARY:", "ALL PASS" if fails == 0 else f"{fails} FAIL")
