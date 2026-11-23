"""RV-02 GPU additions (registered in AMENDMENT_RV-02.md before any score of these runs).

M6G  fine-tuned BERT with the registered A-03 recipe and a wider learning-rate grid G6 (Reviewer 1, comment 9).
M6L  as M6G but input truncated to 256 word pieces; learning rate taken from LOCK_<ds>_M6G.json (Reviewer 2, comment 6).

  python gpu_models_rv.py select M6G <dataset> [wi nw]      validation runs, replicate 1, grid G6 (G6+GX if extended)
  python gpu_models_rv.py lock   M6G <dataset>              LOCK_<dataset>_M6G.json; prints EDGE / SAME_AS_M6 flags
  python gpu_models_rv.py extend M6G <dataset> [wi nw]      only if the lock printed EDGE: runs the registered GX values
  python gpu_models_rv.py test   M6G <dataset> [wi nw]      E1, E2 (5 replicates) and E3 (replicate 1), as A-03
  python gpu_models_rv.py test   M6L <dataset> [wi nw]      E1 and E2 (5 replicates)
  python gpu_models_rv.py smoke  <dataset>                  two validation jobs into work/smoke_rv02/ (installation check)
Run files use the registered names: work/runs/<dataset>/<stage>/<model>/<kind>_<r>_<t>_<n>_<extra>.npz
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import gpu_models as gm
from common import RESULTS, ROOT, WORK
from select_lock import sha

G6 = [1e-5, 2e-5, 3e-5, 5e-5, 8e-5, 1e-4]
GX = [2e-4, 3e-4]                     # registered one-step extension, used only if 1e-4 is selected at some size


def grid_for(ds: str, model: str) -> list[float]:
    ext = (WORK / "runs" / ds / "select" / model / "EXTENDED").exists()
    return G6 + (GX if ext else [])


def bert_obj(ds: str, maxlen: int) -> gm.Bert:
    gm.MAXLEN_M6 = maxlen             # module constant read by Bert.enc
    return gm.Bert(ds)


def run_select(ds: str, model: str, grid: list[float], wi: int, nw: int, smoke: bool = False) -> None:
    from sklearn.metrics import f1_score
    b = bert_obj(ds, 128)
    d = b.d
    out = WORK / ("smoke_rv02" if smoke else "runs") / ds / "select" / model
    out.mkdir(parents=True, exist_ok=True)
    jobs = [("sel", 1, t, n, h) for t in d.years for n in d.sizes for h in grid]
    if smoke:
        jobs = [("sel", 1, d.years[0], d.sizes[0], grid[0]), ("sel", 1, d.years[-1], d.sizes[1], grid[-1])]
    for j, job in enumerate(jobs):
        if j % nw != wi:
            continue
        kind, r, t, n, h = job
        path = out / ("_".join(str(v) for v in job) + ".npz")
        if path.exists():
            continue
        t0 = time.time()
        rows = d.rows(1, t)
        # same seed formula as A-03 (gm.bert): values shared with M6 reproduce its runs up to GPU nondeterminism
        (pred,), ok, ep = b.fit_predict(d.train_rows(t, r, n), h, gm.seed_of(ds, "sel", r, t, n, h), [rows])
        np.savez(path, f1=f1_score(d.cls[rows], pred, average="macro"), acc=np.mean(d.cls[rows] == pred), ok=ok,
                 epochs=ep, sec=time.time() - t0)
        print(model, job, round(time.time() - t0, 1), flush=True)


def lock(ds: str, model: str) -> None:
    from run import Data
    path = RESULTS / f"LOCK_{ds}_{model}.json"
    d = Data(ds)
    grid = grid_for(ds, model)
    base = WORK / "runs" / ds / "select" / model
    missing = [f"sel_1_{t}_{n}_{h}" for t in d.years for n in d.sizes for h in grid
               if not (base / f"sel_1_{t}_{n}_{h}.npz").exists()]
    if missing:
        raise SystemExit(f"STOP: {len(missing)} select runs missing, e.g. {missing[:3]}")
    table, chosen = {}, {}
    for n in d.sizes:
        means = {h: float(np.mean([float(np.load(base / f"sel_1_{t}_{n}_{h}.npz")["f1"]) for t in d.years])) for h in grid}
        table[str(n)] = {str(h): round(v, 6) for h, v in means.items()}
        best = max(means.values())
        chosen[str(n)] = min(h for h, v in means.items() if v >= best - 1e-12)
    edge = any(float(v) == max(grid) for v in chosen.values())
    if edge and not (base / "EXTENDED").exists():
        (base / "EXTENDED").write_text("1e-4 selected at some size; registered extension GX added\n")
        print("EDGE: the largest learning rate was selected; run `extend` then `lock` again (no lock written)")
        return
    if path.exists():
        raise SystemExit(f"{path.name} exists; not overwritten")
    files = [ROOT / "src" / f for f in ("gpu_models.py", "gpu_models_rv.py", "run.py", "select_lock.py")]
    m6 = json.load(open(RESULTS / f"LOCK_{ds}_M6.json"))["selected"]["M6"] if (RESULTS / f"LOCK_{ds}_M6.json").exists() else None
    same = m6 is not None and all(float(m6[k]) == float(chosen[k]) for k in chosen)
    json.dump({"dataset": ds, "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "selected": {model: chosen},
               "grid": grid, "validation_macro_f1": table, "same_as_M6_lock": same,
               "sha256": {Path(f).name: sha(f) for f in files}}, open(path, "w"), indent=1)
    print(open(path).read()[:900])
    print("SAME_AS_M6" if same else "DIFFERENT_FROM_M6")


def test(ds: str, model: str, wi: int, nw: int) -> None:
    from run import test_jobs
    src = "M6G" if model in ("M6G", "M6L") else model
    lk = json.load(open(RESULTS / f"LOCK_{ds}_{src}.json"))
    sel = lk["selected"][src]
    b = bert_obj(ds, 256 if model == "M6L" else 128)
    d = b.d
    jobs = [j for j in test_jobs(d) if not (j[0] == "e3" and j[1] != 1)]     # E3 on replicate 1 only (A-03)
    if model == "M6L":
        jobs = [j for j in jobs if j[0] != "e3"]
    out = WORK / "runs" / ds / "test" / model
    out.mkdir(parents=True, exist_ok=True)
    for j, job in enumerate(jobs):
        if j % nw != wi:
            continue
        kind, r, t, n, extra = job
        path = out / ("_".join(str(v) for v in job) + ".npz")
        if path.exists():
            continue
        t0 = time.time()
        if kind == "e1":
            tr, h, evals = d.train_rows(t, r, n), sel[str(n)], [d.rows(2, u) for u in d.years]
        elif kind == "e2":
            tr, h, evals = d.train_rows(t, r, n), sel[str(n)], [d.rows(2, t)]
        else:
            tr = np.sort(np.concatenate([d.train_rows(t - extra, r, d.N), d.train_rows(t, r, n)]))
            h, evals = sel[str(d.N)], [d.rows(2, t)]
        preds, ok, ep = b.fit_predict(tr, float(h), gm.seed_of(ds, kind, r, t, n, extra, model), evals)
        np.savez(path, pred=np.stack(preds), ok=ok, epochs=ep, sec=time.time() - t0)
        print(model, job, round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "smoke":
        run_select(a[1], "M6G", G6, 0, 1, smoke=True)
    else:
        what, model, ds = a[0], a[1], a[2]
        wi, nw = (int(a[3]), int(a[4])) if len(a) > 4 else (0, 1)
        if what == "select":
            run_select(ds, model, G6, wi, nw)
        elif what == "extend":
            if not (WORK / "runs" / ds / "select" / model / "EXTENDED").exists():
                raise SystemExit("no EXTENDED flag: the registered extension is not triggered")
            run_select(ds, model, GX, wi, nw)
        elif what == "lock":
            lock(ds, model)
        elif what == "test":
            test(ds, model, wi, nw)
