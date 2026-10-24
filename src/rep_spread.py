"""Break-even size and ratio for each training replicate separately (X-12; descriptive, after all results were known).

Same interpolation as evaluate.py on the U_g-averaged current-year curve and stale score, but without averaging
over replicates, so that the spread between training draws can be reported for n*(g) and rho(g).
Stored test predictions only; no refitting, no bootstrap.

python rep_spread.py <dataset> -> results/rep_spread_<dataset>.json
"""
import json
import sys

import numpy as np

import evaluate as ev
from common import RESULTS, WORK
from run import Data
from year_pairs import f1


def main(dataset):
    d = Data(dataset)
    Y, S, N, K = d.years, d.sizes, d.N, d.K
    ys = {u: d.cls[d.rows(2, u)].astype(np.int64) for u in Y}
    path = RESULTS / f"rep_spread_{dataset}.json"
    if path.exists():
        sys.exit(f"{path.name} exists")
    out = {"dataset": dataset, "N": N, "models": {}}
    for m in ("M1", "M2", "M3", "M4", "M5F", "M6"):
        base = WORK / "runs" / dataset / "test" / m
        res = {}
        for r in (1, 2, 3, 4, 5):
            F = np.zeros((len(Y), len(S))); G = np.zeros((len(Y), len(Y)))
            for ti, t in enumerate(Y):
                p = np.load(base / f"e1_{r}_{t}_{N}_None.npz")["pred"]
                for ui, u in enumerate(Y):
                    G[ti, ui] = f1(ys[u], p[ui], K)
                F[ti, -1] = G[ti, ti]
                for ni, n in enumerate(S[:-1]):
                    F[ti, ni] = f1(ys[t], np.load(base / f"e2_{r}_{t}_{n}_None.npz")["pred"][0], K)
            for g in (1, len(Y) - 1):
                us = list(range(g, len(Y)))
                ns = ev.breakeven(S, list(F[us].mean(0)), float(np.mean([G[u - g, u] for u in us])))
                res.setdefault(str(g), []).append(None if (ns == 0.0 or np.isinf(ns)) else round(ns, 1))
        out["models"][m] = {g: {"n_star_by_replicate": v,
                                "rho_sd": (round(float(np.std([x / N for x in v], ddof=1)), 4) if None not in v else None),
                                "n_star_min": min(x for x in v if x is not None), "n_star_max": max(x for x in v if x is not None),
                                "n_censored": v.count(None)} for g, v in res.items()}
    json.dump(out, open(path, "w"), indent=1)
    for m, v in out["models"].items():
        print(dataset, m, {g: (x["n_star_by_replicate"], x["rho_sd"]) for g, x in v.items()})


if __name__ == "__main__":
    main(sys.argv[1])
