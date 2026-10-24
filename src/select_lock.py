"""Validation-only selection and locking (plan Sections 4 and 7).

python select_lock.py <dataset> select   -> results/selection_<dataset>.json
python select_lock.py <dataset> lock     -> results/LOCK_<dataset>.json (refuses to overwrite)
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from common import RESULTS, ROOT, WORK
from run import GRID, Data


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(dataset, models):
    d = Data(dataset)
    out = {}
    for m in models:
        base = WORK / "runs" / dataset / "select" / m
        table, chosen, not_converged = {}, {}, 0
        for n in d.sizes:
            means = {}
            for h in GRID[m]:
                vals = []
                for t in d.years:
                    z = np.load(base / f"sel_1_{t}_{n}_{h}.npz")
                    vals.append(float(z["f1"]))
                    not_converged += int(not bool(z["ok"]))
                means[h] = float(np.mean(vals))
            table[str(n)] = {str(h): round(v, 6) for h, v in means.items()}
            best = max(means.values())
            chosen[str(n)] = min(h for h, v in means.items() if v >= best - 1e-12)   # ties: the smaller value
        out[m] = {"validation_macro_f1": table, "selected": chosen, "fits_at_max_iter": not_converged}
    json.dump(out, open(RESULTS / f"selection_{dataset}.json", "w"), indent=1)
    for m in out:
        print(m, out[m]["selected"], "max_iter hits:", out[m]["fits_at_max_iter"])
        print("   ", out[m]["validation_macro_f1"][str(d.N)])


def lock(dataset):
    path = RESULTS / f"LOCK_{dataset}.json"
    if path.exists():
        sys.exit("lock exists; not overwritten")
    sel = json.load(open(RESULTS / f"selection_{dataset}.json"))
    files = [ROOT / "analysis_plan.md", ROOT / "AMENDMENTS.md", WORK / f"sample_{dataset}.pkl"] + \
            [ROOT / "src" / f for f in ("common.py", "features.py", "run.py", "evaluate.py", "select_lock.py")]
    json.dump({"dataset": dataset, "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
               "selected": {m: v["selected"] for m, v in sel.items()},
               "sha256": {str(f.relative_to(ROOT)): sha(f) for f in files}}, open(path, "w"), indent=1)
    print(open(path).read())


def lock_extra(dataset, model, files):
    """A-03: validation-only selection and a separate lock for a model added after the main locks."""
    import importlib
    path = RESULTS / f"LOCK_{dataset}_{model}.json"
    if path.exists():
        sys.exit(f"{model} lock exists; not overwritten")
    d = Data(dataset)
    grid = GRID[model] if model in GRID else importlib.import_module("gpu_models").GRID_M6
    base = WORK / "runs" / dataset / "select" / model
    table, chosen = {}, {}
    for n in d.sizes:
        means = {h: float(np.mean([float(np.load(base / f"sel_1_{t}_{n}_{h}.npz")["f1"]) for t in d.years])) for h in grid}
        table[str(n)] = {str(h): round(v, 6) for h, v in means.items()}
        best = max(means.values())
        chosen[str(n)] = min(h for h, v in means.items() if v >= best - 1e-12)
    json.dump({"dataset": dataset, "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "selected": {model: chosen},
               "validation_macro_f1": table, "sha256": {Path(f).name: sha(f) for f in files}}, open(path, "w"), indent=1)
    print(open(path).read()[:600])


def lock_m5(dataset):
    """X-03: validation-only selection for M5 and its own lock file; the main lock is not touched."""
    path = RESULTS / f"LOCK_{dataset}_M5.json"
    if path.exists():
        sys.exit("M5 lock exists; not overwritten")
    d = Data(dataset)
    base = WORK / "runs" / dataset / "select" / "M5"
    table, chosen = {}, {}
    for n in d.sizes:
        means = {h: float(np.mean([float(np.load(base / f"sel_1_{t}_{n}_{h}.npz")["f1"]) for t in d.years])) for h in GRID["M5"]}
        table[str(n)] = {str(h): round(v, 6) for h, v in means.items()}
        best = max(means.values())
        chosen[str(n)] = min(h for h, v in means.items() if v >= best - 1e-12)
    files = [ROOT / "src" / f for f in ("run.py", "features.py", "evaluate.py", "encode_minilm.py", "minilm_np.py")] + [WORK / f"minilm_{dataset}.npy"]
    json.dump({"dataset": dataset, "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "selected": {"M5": chosen},
               "validation_macro_f1": table, "sha256": {f.name: sha(f) for f in files}}, open(path, "w"), indent=1)
    print(open(path).read()[:600])


if __name__ == "__main__":
    ds, what = sys.argv[1:3]
    if what == "select":
        select(ds, sys.argv[3:] or ["M1", "M2", "M3", "M4"])
    elif what == "lock_m5":
        lock_m5(ds)
    elif what == "lock_extra":                       # python select_lock.py <ds> lock_extra <model>
        m = sys.argv[3]
        src = ROOT / "src"
        files = [src / f for f in ("run.py", "features.py", "evaluate.py", "select_lock.py")]
        files += [src / "gpu_models.py", WORK / f"minilm_{ds}_full.npy"] if m == "M5F" else [src / "gpu_models.py"]
        lock_extra(ds, m, files)
    else:
        lock(ds)
