"""Rebuild work/sample_<dataset>.pkl (labels, years, parts only; no text) from the public sample index.
Row order of the index equals the row order of the original sample file (src/export_index.py)."""
import csv, gzip, json, pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def build(dataset: str) -> None:
    dec = json.load(open(ROOT / "results" / "gate.json"))[dataset]["decision"]
    cls_of = {c: i for i, c in enumerate(dec["classes"])}
    rows = list(csv.reader(gzip.open(ROOT / "results" / f"sample_index_{dataset}.csv.gz", "rt")))[1:]
    s = {"cls": np.array([cls_of[r[0]] for r in rows], np.int8),
         "year": np.array([int(r[-2]) for r in rows], np.int16),
         "part": np.array([int(r[-1]) for r in rows], np.int8),
         "item": [r[1] if dataset == "reddit" else f"{r[1]}:{r[2]}" for r in rows],
         "classes": dec["classes"], "years": dec["years"], "size": tuple(dec["size"])}
    pickle.dump(s, open(ROOT / "work" / f"sample_{dataset}.pkl", "wb"))
    print(dataset, len(rows), {p: int((s["part"] == p).sum()) for p in (0, 1, 2)})

if __name__ == "__main__":
    for ds in sys.argv[1:] or ["reddit", "stackexchange"]:
        build(ds)
