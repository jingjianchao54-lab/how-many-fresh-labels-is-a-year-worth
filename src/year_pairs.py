"""Ratio for each single pair of years (X-11; descriptive, computed after all results were known).

The main estimate averages the current-year curve and the stale score over the test years U_g before
interpolating. Here the same interpolation is applied to each pair (u - g -> u) separately, using the
stored test predictions only (no model is refitted and no bootstrap is drawn). The spread across pairs
shows how much the ratio varies between calendar years, which the bootstrap intervals do not cover.

python year_pairs.py <dataset> -> results/year_pairs_<dataset>.json
"""
import json
import sys

import numpy as np

import evaluate as ev
from common import RESULTS, WORK
from run import Data


def f1(y, pred, K):
    conf = np.bincount(y * K + pred.astype(np.int64), minlength=K * K).reshape(K, K).astype(float)
    return float(ev.macro_f1_from_conf(conf)[0])


def main(dataset):
    d = Data(dataset)
    Y, S, N, K = d.years, d.sizes, d.N, d.K
    ys = {u: d.cls[d.rows(2, u)].astype(np.int64) for u in Y}
    out = {"dataset": dataset, "sizes": S, "N": N, "models": {}}
    path = RESULTS / f"year_pairs_{dataset}.json"
    if path.exists():
        sys.exit(f"{path.name} exists")
    for m in ("M1", "M2", "M3", "M4", "M5F", "M6"):
        base = WORK / "runs" / dataset / "test" / m
        reps = [1, 2, 3, 4, 5]
        F = np.zeros((len(reps), len(Y), len(S)))          # F[r, test year, size]
        G = np.zeros((len(reps), len(Y), len(Y)))          # G[r, train year, test year] at size N
        for ri, r in enumerate(reps):
            for ti, t in enumerate(Y):
                p = np.load(base / f"e1_{r}_{t}_{N}_None.npz")["pred"]
                for ui, u in enumerate(Y):
                    G[ri, ti, ui] = f1(ys[u], p[ui], K)
                F[ri, ti, -1] = G[ri, ti, ti]
                for ni, n in enumerate(S[:-1]):
                    F[ri, ti, ni] = f1(ys[t], np.load(base / f"e2_{r}_{t}_{n}_None.npz")["pred"][0], K)
        Fm, Gm = F.mean(0), G.mean(0)
        res = {}
        for g in range(1, len(Y)):
            pairs = []
            for ui in range(g, len(Y)):
                ns = ev.breakeven(S, list(Fm[ui]), Gm[ui - g, ui])
                cens = ns == 0.0 or np.isinf(ns)
                pairs.append({"train": Y[ui - g], "test": Y[ui], "cost": round(float(Fm[ui, -1] - Gm[ui - g, ui]), 5),
                              "n_star": ev.fmt_n(ns, S), "rho": None if cens else round(ns / N, 4)})
            r = [p["rho"] for p in pairs if p["rho"] is not None]
            res[str(g)] = {"pairs": pairs, "n_pairs": len(pairs), "n_censored": len(pairs) - len(r),
                           "rho_min": min(r) if r else None, "rho_median": float(np.median(r)) if r else None,
                           "rho_max": max(r) if r else None}
        out["models"][m] = res
    json.dump(out, open(path, "w"), indent=1)
    for m in out["models"]:
        print(dataset, m, {g: (v["rho_min"], v["rho_max"], v["n_censored"]) for g, v in out["models"][m].items()})


if __name__ == "__main__":
    main(sys.argv[1])
