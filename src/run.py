"""Resumable experiment runner (plan Sections 4, 5, 7).

Usage: python run.py <dataset> <stage> <model> [worker_index n_workers]
  stage "select": replicate 1, every (training year, size n, grid value) scored on the SAME-YEAR validation set.
  stage "dry"   : pipeline check; same jobs as "test" for replicate 1 but scored on validation sets.
  stage "test"  : requires results/LOCK_<dataset>.json; writes test predictions for E1, E2 and E3.
Each job writes one small file under work/runs/<dataset>/<stage>/<model>/; existing files are skipped.
"""
import json
import os
import pickle
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from common import RESULTS, SIZES, WORK
from features import TrainVocab, global_counts, glove_means

GRID = {"M1": [1, 10, 100], "M2": [0.1, 1, 10], "M3": [0.1, 0.5, 1.0], "M4": [0.1, 1, 10], "M5": [0.1, 1, 10],
        "M5F": [0.1, 1, 10]}          # A-03: M5F = MiniLM, full design (256 word pieces, all items, five replicates)


class Data:
    def __init__(self, dataset, field="tok"):
        s = pickle.load(open(WORK / f"sample_{dataset}.pkl", "rb"))
        self.cls, self.year, self.part = s["cls"], s["year"], s["part"]
        self.years, self.K, (self.N, self.V, self.T) = s["years"], len(s["classes"]), s["size"]
        self.sizes = [n for n in SIZES if n <= self.N]
        self.dataset, self.field = dataset, field
        self._x = self._g = self._e = self._ef = None

    def rows(self, part, year):
        return np.flatnonzero((self.part == part) & (self.year == year))

    def train_rows(self, year, rep, n):
        """n items per class from the replicate's shuffled draw of N out of the 2N pool; sizes are nested."""
        out = []
        for c in range(self.K):
            pool = np.flatnonzero((self.part == 0) & (self.year == year) & (self.cls == c))
            rng = np.random.default_rng([rep, c, year])
            out.append(rng.permutation(pool)[: self.N][:n])
        return np.sort(np.concatenate(out))

    @property
    def x(self):
        if self._x is None:
            self._x = global_counts(self.dataset, self.field)
        return self._x

    @property
    def g(self):
        if self._g is None:
            self._g = glove_means(self.dataset, self.field)
        return self._g

    @property
    def ef(self):                                   # A-03
        if self._ef is None:
            self._ef = np.load(WORK / f"minilm_{self.dataset}_full.npy")
        return self._ef

    @property
    def e(self):
        if self._e is None:
            self._e = np.load(WORK / f"minilm_{self.dataset}.npy")
        return self._e


def fit_predict(d, model, h, tr, eval_sets):
    """Train on rows tr, return predictions for each array of rows in eval_sets, plus a convergence flag."""
    y = d.cls[tr]
    flag = True
    if model in ("M1", "M2", "M3"):
        tv = TrainVocab(d.x, tr, tfidf=(model != "M3"), max_features=(None if model == "M3" else 200_000))
        xtr = tv.transform(tr)
        f = tv.transform
    else:
        mat = d.g if model == "M4" else (d.ef if model == "M5F" else d.e)
        if model == "M4":
            sc = StandardScaler().fit(mat[tr])
            f = lambda r: sc.transform(mat[r])
        else:
            f = lambda r: mat[r]
        xtr = f(tr)
    if model == "M2":
        clf = LinearSVC(C=h).fit(xtr, y)
    elif model == "M3":
        clf = ComplementNB(alpha=h).fit(xtr, y)
    else:
        clf = LogisticRegression(C=h, solver="lbfgs", max_iter=1000).fit(xtr, y)
        flag = bool(np.all(clf.n_iter_ < 1000))
    return [clf.predict(f(r)).astype(np.int8) for r in eval_sets], flag


def select_jobs(d, model):
    return [("sel", 1, t, n, h) for t in d.years for n in d.sizes for h in GRID[model]]


def test_jobs(d):
    jobs = []
    gmax = d.years[-1] - d.years[0]
    for r in range(1, 6):
        for t in d.years:
            jobs.append(("e1", r, t, d.N, None))                 # scored on every test year
            for n in d.sizes[:-1]:
                jobs.append(("e2", r, t, n, None))               # fresh n, scored on test year t
                for g in sorted({1, gmax}):
                    if t - g >= d.years[0]:
                        jobs.append(("e3", r, t, n, g))          # stale N from t-g plus fresh n from t
    return jobs


def main():
    dataset, stage, model = sys.argv[1:4]
    wi, nw = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)
    field = sys.argv[6] if len(sys.argv) > 6 else "tok"
    d = Data(dataset, field)
    tag = "" if field == "tok" else f"_{field}"
    out = WORK / "runs" / f"{dataset}{tag}" / stage / model
    out.mkdir(parents=True, exist_ok=True)
    if stage == "select":
        jobs = select_jobs(d, model)
    elif stage == "dry":          # pipeline check on VALIDATION sets, replicate 1 only; never reads test rows
        sel = json.load(open(RESULTS / f"selection_{dataset}.json"))[model]["selected"]
        jobs = [j for j in test_jobs(d) if j[1] == 1]
    else:
        lock = json.load(open(RESULTS / f"LOCK_{dataset}.json"))
        if model in lock["selected"]:
            sel = lock["selected"][model]
        else:                                   # X-03: M5 is selected and locked separately, before its own scoring
            sel = json.load(open(RESULTS / f"LOCK_{dataset}_{model}.json"))["selected"][model]
        jobs = test_jobs(d)
        if model == "M5":                       # A-02: M5 on replicate 1 only
            jobs = [j for j in jobs if j[1] == 1]
        if os.environ.get("SKIP_E3") == "1":    # sensitivity runs: E1 and E2 only
            jobs = [j for j in jobs if j[0] != "e3"]
        if os.environ.get("HALF") == "1":       # E5: stale model at half the cell size, scored on every test year
            jobs = [("h1", r, t, int(os.environ.get("HALF_N", d.N // 2)), None) for r in range(1, 6) for t in d.years]
    ep = 1 if stage == "dry" else 2
    for j, job in enumerate(jobs):
        if j % nw != wi:
            continue
        kind, r, t, n, extra = job
        path = out / ("_".join(str(v) for v in job) + ".npz")
        if path.exists():
            continue
        t0 = time.time()
        if kind == "sel":
            rows = d.rows(1, t)
            (pred,), ok = fit_predict(d, model, extra, d.train_rows(t, r, n), [rows])
            np.savez(path, f1=f1_score(d.cls[rows], pred, average="macro"), acc=np.mean(d.cls[rows] == pred), ok=ok,
                     sec=time.time() - t0)
        else:
            if kind in ("e1", "h1"):
                tr, h, evals = d.train_rows(t, r, n), sel[str(n)], [d.rows(ep, u) for u in d.years]
            elif kind == "e2":
                tr, h, evals = d.train_rows(t, r, n), sel[str(n)], [d.rows(ep, t)]
            else:
                tr = np.sort(np.concatenate([d.train_rows(t - extra, r, d.N), d.train_rows(t, r, n)]))
                h, evals = sel[str(d.N)], [d.rows(ep, t)]
            preds, ok = fit_predict(d, model, h, tr, evals)
            np.savez(path, pred=np.stack(preds), ok=ok, sec=time.time() - t0)
        print(model, job, round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
