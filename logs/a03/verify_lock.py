import json
from pathlib import Path
import numpy as np

RESULTS = Path("results")
WORK = Path("work")
GRID = {"M5F": [0.1, 1, 10], "M6": [2e-5, 5e-5]}


def verify(dataset, model):
    p = RESULTS / ("LOCK_" + dataset + "_" + model + ".json")
    if not p.exists():
        print("  missing " + str(p))
        return False
    lock = json.load(open(p))
    locked = lock["selected"][model]
    table = lock["validation_macro_f1"]
    base = WORK / "runs" / dataset / "select" / model
    grid = GRID[model]
    ok = True
    print("  locked_at = " + str(lock["locked_at"]))
    for key in sorted(locked, key=int):
        n = int(key)
        means = {}
        nfiles = {}
        missing = []
        for h in grid:
            pat = "sel_1_*_" + str(n) + "_" + str(h) + ".npz"
            fs = sorted(base.glob(pat))
            nfiles[str(h)] = len(fs)
            if len(fs) == 0:
                missing.append(str(h))
                continue
            vals = [float(np.load(f)["f1"]) for f in fs]
            means[h] = float(np.mean(vals))
        if len(missing) > 0:
            print("  n=%5d missing hyperparams %s" % (n, missing))
            ok = False
            continue
        best = max(means.values())
        recomputed = min(h for h, v in means.items() if v >= best - 1e-12)
        locked_v = locked[key]
        sel_ok = abs(float(recomputed) - float(locked_v)) < 1e-15
        tab_ok = True
        for h in grid:
            if abs(round(means[h], 6) - table[key][str(h)]) > 1e-9:
                tab_ok = False
        sel_label = "OK" if sel_ok else "MISMATCH"
        tab_label = "OK" if tab_ok else "MISMATCH"
        print("  n=%5d years=%2d recomputed=%s locked=%s sel=%s table=%s"
              % (n, nfiles[str(grid[0])], recomputed, locked_v, sel_label, tab_label))
        ok = ok and sel_ok and tab_ok
    return ok


allok = True
for ds in ("stackexchange", "reddit"):
    for m in ("M5F", "M6"):
        print("[" + ds + " / " + m + "]")
        r = verify(ds, m)
        allok = allok and r

if allok:
    print(">>> RESULT: lock files fully consistent with select run files")
else:
    print(">>> RESULT: MISMATCH - stop and report to the author")
