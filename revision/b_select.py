"""Aggregate revision selection runs with the registered rule (mean validation Macro-F1 over years, ties -> smaller).
python b_select.py <dataset> widegrid|mixtune"""
import json, sys
import numpy as np
from b_runs import RW, WIDE, run
ds, variant = sys.argv[1:3]
import common_rev
d = run.Data(ds) if False else None
gate = json.load(open(common_rev.ROOT / "results" / "gate.json"))[ds]["decision"]
years, sizes = gate["years"], [25, 50, 100, 200, 500, 1000, 2000][: [25, 50, 100, 200, 500, 1000, 2000].index(gate["size"][0]) + 1]
base = RW / variant / "runs" / ds / "select" / "M1"
out, table = {}, {}
if variant == "widegrid":
    for n in sizes:
        means = {h: float(np.mean([float(np.load(base / f"sel_1_{t}_{n}_{h}.npz")["f1"]) for t in years])) for h in WIDE}
        best = max(means.values()); out[str(n)] = min(h for h, v in means.items() if v >= best - 1e-12)
        table[str(n)] = {str(h): round(v, 6) for h, v in means.items()}
else:
    gmax = years[-1] - years[0]
    for n in sizes[:-1]:
        ts = [t for t in years if t - gmax >= years[0]]
        means = {h: float(np.mean([float(np.load(base / f"msel_{t}_{n}_{gmax}_{h}.npz")["f1"]) for t in ts])) for h in run.GRID["M1"]}
        best = max(means.values()); out[f"{gmax}_{n}"] = min(h for h, v in means.items() if v >= best - 1e-12)
        table[str(n)] = {str(h): round(v, 6) for h, v in means.items()}
json.dump(out, open(RW / variant / f"selection_{ds}.json", "w"), indent=1)
json.dump(table, open(RW / variant / f"selection_table_{ds}.json", "w"), indent=1)
print(ds, variant, out)
