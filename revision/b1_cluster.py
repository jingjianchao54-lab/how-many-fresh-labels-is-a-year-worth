"""R-B1: author-clustered stratified bootstrap (Reviewer 2, comment 4). Stored test predictions only; no refitting.

Within every (class, year) test cell, authors (not items) are resampled with replacement, keeping all items of a drawn
author; 2,000 resamples, seed 20261009 as in the registered analysis. Everything else (replicate averaging, U_g
averaging, break-even interpolation, censoring rule, one-sided p-values, Holm within dataset) is unchanged.
python b1_cluster.py <dataset> [models...] -> revision/out/cluster_<dataset>.json
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np

from common_rev import B, REPS5, ROOT, CellScorer, Data, breakeven, censored_ci, holm, load_grid, pct_ci

OUT = ROOT / "revision" / "out"


def nfmt(v: float, N: int):
    return "<25" if v == 0 else (f">{N}" if np.isinf(v) else round(float(v), 1))


def summarise(d: Data, A: dict, gmax: int) -> dict:
    Y, S, N = d.years, d.sizes, d.N
    ny = len(Y)
    out = {"cost": {}, "exchange": {}}
    store = {}
    for g in range(1, gmax + 1):
        pairs = [(u - g, u) for u in range(ny) if u - g >= 0]
        pt = float(np.mean([A["P"][u, u] - A["P"][t, u] for t, u in pairs]))
        bt = np.mean([A["Pb"][u, u] - A["Pb"][t, u] for t, u in pairs], axis=0)
        store[g] = bt
        out["cost"][str(g)] = {"cost": round(pt, 5), "ci": [round(x, 5) for x in pct_ci(bt)]}
        us = [u for u in range(ny) if u - g >= 0]
        fresh, stale = A["F"][us].mean(0), np.mean([A["P"][u - g, u] for u in us])
        fb, sb = A["Fb"][us].mean(0), np.mean([A["Pb"][u - g, u] for u in us], axis=0)
        p = breakeven(S, fresh, stale)
        bb = np.array([breakeven(S, fb[:, b], sb[b]) for b in range(B)])
        lo, hi = censored_ci(bb)
        out["exchange"][str(g)] = {"n_star": nfmt(p, N), "n_star_ci": [nfmt(lo, N), nfmt(hi, N)],
                                   "rho_ci": [None if (v == 0 or np.isinf(v)) else round(v / N, 4) for v in (lo, hi)],
                                   "share_censored_high": round(float(np.mean(np.isinf(bb))), 4)}
        if g == gmax:
            nb = bb
    dbt = store[gmax] - store[1]
    out["diff_gmax_minus_1_ci"] = [round(x, 5) for x in pct_ci(dbt)]
    pv = {"H1": float(np.mean(store[1] <= 0)), "H2": float(np.mean(dbt <= 0)), "H3": float(np.mean(nb >= N / 2))}
    out["p_one_sided"] = {k: round(v, 4) for k, v in pv.items()}
    out["holm"] = {k: round(v, 4) for k, v in holm(pv).items()}
    return out


def main() -> None:
    ds = sys.argv[1]
    models = sys.argv[2:] or ["M1", "M2", "M3", "M4", "M5F", "M6"]
    s = pickle.load(open(ROOT / "revision" / "full" / f"sample_{ds}.pkl", "rb"))
    authors = np.array(s["author"])
    d = Data(ds)
    gmax = d.years[-1] - d.years[0]
    # cluster structure of the test cells
    sizes = []
    for u in d.years:
        rows = d.rows(2, u)
        for c in range(d.K):
            a = authors[rows][d.cls[rows] == c]
            _, cnt = np.unique(a, return_counts=True)
            sizes.append((len(a), len(cnt), float(np.mean(cnt)), int(cnt.max()), float(np.mean(cnt == 1))))
    sizes = np.array(sizes)
    res = {"test_cells": {"items_per_cell": int(sizes[0, 0]), "authors_per_cell_mean": round(float(sizes[:, 1].mean()), 1),
                          "authors_per_cell_min": int(sizes[:, 1].min()),
                          "items_per_author_mean": round(float(sizes[:, 2].mean()), 3),
                          "max_items_per_author": int(sizes[:, 3].max()),
                          "share_single_item_authors": round(float(sizes[:, 4].mean()), 3)},
           "models": {}}
    sc_item = CellScorer(d, 2)
    sc_cl = CellScorer(d, 2, clusters=authors)
    for m in models:
        A_cl = load_grid(ds, m, REPS5, d, sc_cl)
        A_it = load_grid(ds, m, REPS5, d, sc_item) if m == "M1" else None
        r = {"cluster": summarise(d, A_cl, gmax)}
        if A_it is not None:
            r["item_recomputed"] = summarise(d, A_it, gmax)
            w = lambda A, g: np.diff(pct_ci(np.mean([A["Pb"][u, u] - A["Pb"][u - g, u]
                                                      for u in range(len(d.years)) if u - g >= 0], axis=0)))[0]
            r["ci_width_ratio_cluster_over_item"] = {str(g): round(float(w(A_cl, g) / w(A_it, g)), 3) for g in (1, gmax)}
        res["models"][m] = r
        print(ds, m, json.dumps(r["cluster"]["cost"]["1"]), r["cluster"]["exchange"][str(gmax)], r["cluster"]["holm"], flush=True)
    json.dump(res, open(OUT / f"cluster_{ds}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
