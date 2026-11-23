"""Revision analyses R-A1 to R-A5 (descriptive, added after review; stored test predictions only, no refitting).

python a_analyses.py <dataset> <analysis> [models...]
  classwise   R-A1: per-class staleness cost C_c(g) and class-specific break-even size n*_c(g) (Reviewer 1, comment 8)
  curves      R-A2: alternative learning-curve models and estimators of n*(g) (Reviewer 1, comments 6-7)
  confusion   R-A3: where the extra errors of old labels go (class-boundary migration; Reviewer 1, comment 4)
  posttype    R-A4: Stack Exchange costs and ratios by post type of the test item (Reviewer 2, comment 6)
Writes revision/out/<analysis>_<dataset>.json
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
from scipy.optimize import brentq, curve_fit

from common_rev import (B, REPS5, ROOT, CellScorer, Data, breakeven, censored_ci, exchange, gap_sets,
                        load_grid, pct_ci)

OUT = ROOT / "revision" / "out"
OUT.mkdir(parents=True, exist_ok=True)
GPU_REPS = {"M5F": REPS5, "M6": REPS5}


def nfmt(v: float, N: int):
    return None if v is None else ("<25" if v == 0 else (f">{N}" if np.isinf(v) else round(float(v), 1)))


# ---------------------------------------------------------------- R-A1 class-wise
def classwise(ds: str, models: list[str]) -> dict:
    d = Data(ds)
    sc = CellScorer(d, 2)
    Y, S, N, K = d.years, d.sizes, d.N, d.K
    gmax = Y[-1] - Y[0]
    out = {"classes": [str(c) for c in json.load(open(ROOT / "results" / "gate.json"))[ds]["decision"]["classes"]],
           "N": N, "models": {}}
    for m in models:
        A = load_grid(ds, m, REPS5, d, sc)
        res = {}
        for g in (1, gmax):
            us = gap_sets(Y, g)
            per = []
            for c in range(K):
                cost = float(np.mean([A["PC"][u, u, c] - A["PC"][u - g, u, c] for u in us]))
                cost_b = np.mean([A["PCB"][u, u, :, c] - A["PCB"][u - g, u, :, c] for u in us], axis=0)
                fresh = A["FC"][us, :, c].mean(0)
                stale = float(np.mean([A["PC"][u - g, u, c] for u in us]))
                fresh_b = A["FCB"][us, :, :, c].mean(0)
                stale_b = np.mean([A["PCB"][u - g, u, :, c] for u in us], axis=0)
                pt = breakeven(S, fresh, stale)
                bt = np.array([breakeven(S, fresh_b[:, b], stale_b[b]) for b in range(B)])
                lo, hi = censored_ci(bt)
                top = (fresh[-1] - fresh[-2]) / np.log(S[-1] / S[-2])
                per.append({"class": out["classes"][c], "same_year_f1": round(float(fresh[-1]), 5),
                            "cost": round(cost, 5), "cost_ci": [round(x, 5) for x in pct_ci(cost_b)],
                            "n_star": nfmt(pt, N), "n_star_ci": [nfmt(lo, N), nfmt(hi, N)],
                            "rho": None if (pt == 0 or np.isinf(pt)) else round(pt / N, 4),
                            "top_slope_points": round(float(top) * 100, 3),
                            "share_censored_high": round(float(np.mean(np.isinf(bt))), 4)})
            res[str(g)] = per
        out["models"][m] = res
        print(ds, m, "done", flush=True)
    return out


# ---------------------------------------------------------------- R-A2 learning-curve models
def ipl(n, a, b, c):
    return a - b * np.power(n, -c)


def fit_ipl(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, _ = curve_fit(ipl, x, y, p0=[min(1.0, y[-1] + 0.05), 1.0, 0.5],
                         bounds=([0.0, 0.0, 0.01], [1.0, 100.0, 3.0]), maxfev=20000)
    return p


def n_from_ipl(p, stale):
    a, b, c = p
    return np.inf if stale >= a else float((b / (a - stale)) ** (1.0 / c))


def fit_loglin(x, y):
    s, a = np.polyfit(np.log(x), y, 1)
    return a, s


def n_from_loglin(p, stale):
    a, s = p
    return np.inf if s <= 0 else float(np.exp((stale - a) / s))


def breakeven_linear_n(sizes, fresh, stale):
    """Same first-crossing rule as the registered estimator, but linear in n instead of log n."""
    sizes = np.asarray(sizes, float)
    if fresh[0] >= stale:
        return 0.0
    for i in range(1, len(sizes)):
        if fresh[i] >= stale:
            a = (stale - fresh[i - 1]) / (fresh[i] - fresh[i - 1])
            return float(sizes[i - 1] + a * (sizes[i] - sizes[i - 1]))
    return np.inf


def estimators(S, fresh, stale, pilot_max):
    x, y = np.asarray(S, float), np.asarray(fresh, float)
    pil = x <= pilot_max
    out = {"loglinear_interp": breakeven(S, fresh, stale),          # registered estimator
           "linear_interp": breakeven_linear_n(S, fresh, stale)}
    try:
        out["ipl_fit_all"] = n_from_ipl(fit_ipl(x, y), stale)
    except RuntimeError:
        out["ipl_fit_all"] = np.nan
    out["loglin_fit_all"] = n_from_loglin(fit_loglin(x, y), stale)
    try:
        out["ipl_pilot_extrapolation"] = n_from_ipl(fit_ipl(x[pil], y[pil]), stale)
    except RuntimeError:
        out["ipl_pilot_extrapolation"] = np.nan
    return out


def holdout_top(S, fresh):
    """Predict the score at N from the points below N; returns error in Macro-F1 points per curve model."""
    x, y = np.asarray(S, float), np.asarray(fresh, float)
    res = {}
    res["ipl"] = (ipl(x[-1], *fit_ipl(x[:-1], y[:-1])) - y[-1]) * 100
    a, s = fit_loglin(x[:-1], y[:-1])
    res["loglin_global"] = (a + s * np.log(x[-1]) - y[-1]) * 100
    s2 = (y[-2] - y[-3]) / np.log(x[-2] / x[-3])                          # local log-linear (registered form)
    res["loglin_local"] = (y[-2] + s2 * np.log(x[-1] / x[-2]) - y[-1]) * 100
    return {k: round(float(v), 3) for k, v in res.items()}


def fit_rmse(S, fresh):
    x, y = np.asarray(S, float), np.asarray(fresh, float)
    p = fit_ipl(x, y)
    a, s = fit_loglin(x, y)
    return {"ipl": round(float(np.sqrt(np.mean((ipl(x, *p) - y) ** 2)) * 100), 3),
            "loglin_global": round(float(np.sqrt(np.mean((a + s * np.log(x) - y) ** 2)) * 100), 3),
            "ipl_params": [round(float(v), 4) for v in p]}


def curves(ds: str, models: list[str]) -> dict:
    d = Data(ds)
    sc = CellScorer(d, 2)
    Y, S, N = d.years, d.sizes, d.N
    gmax = Y[-1] - Y[0]
    pilot_max = N / 4
    out = {"N": N, "sizes": S, "pilot_max": pilot_max, "models": {}}
    for m in models:
        A = load_grid(ds, m, REPS5, d, sc)
        res = {"fit_all_year_curve": fit_rmse(S, A["F"].mean(0)), "holdout_top_all_year_curve": holdout_top(S, A["F"].mean(0)),
               "gaps": {}}
        for g in range(1, gmax + 1):
            _, _, fresh, stale, fresh_b, stale_b = exchange(S, g, A)
            pt = estimators(S, fresh, stale, pilot_max)
            entry = {k: {"n_star": nfmt(v, N) if not np.isnan(v) else None} for k, v in pt.items()}
            if g in (1, gmax) and m == "M1":                         # bootstrap intervals for the primary model
                bs = {k: [] for k in pt}
                for b in range(B):
                    e = estimators(S, fresh_b[:, b], stale_b[b], pilot_max)
                    for k in e:
                        bs[k].append(e[k])
                for k in pt:
                    arr = np.array(bs[k], float)
                    arr = arr[~np.isnan(arr)]
                    lo, hi = censored_ci(arr)
                    entry[k]["n_star_ci"] = [nfmt(lo, N), nfmt(hi, N)]
                    entry[k]["H3_upper_below_half"] = bool(hi < N / 2)
                    entry[k]["H3_lower_at_or_above_half"] = bool(lo >= N / 2)
            for k in entry:
                v = pt[k]
                entry[k]["rho"] = None if (np.isnan(v) or v == 0 or np.isinf(v)) else round(v / N, 4)
            res["gaps"][str(g)] = entry
        out["models"][m] = res
        print(ds, m, "done", flush=True)
    return out


# ---------------------------------------------------------------- R-A3 confusion migration
def confusion(ds: str, models: list[str]) -> dict:
    d = Data(ds)
    sc = CellScorer(d, 2)
    Y, K = d.years, d.K
    gmax = Y[-1] - Y[0]
    classes = [str(c) for c in json.load(open(ROOT / "results" / "gate.json"))[ds]["decision"]["classes"]]
    out = {"classes": classes, "models": {}}
    for m in models:
        A = load_grid(ds, m, REPS5, d, sc, want_conf=True)
        res = {}
        for g in (1, gmax):
            us = gap_sets(Y, g)
            delta = sum(A["CONF"][u - g, u] - A["CONF"][u, u] for u in us) / len(us)   # stale minus fresh, items per test year
            off = delta.copy()
            np.fill_diagonal(off, 0)
            pair = off + off.T
            iu = np.triu_indices(K, 1)
            vals = pair[iu]
            order = np.argsort(-vals)
            pos_total = float(vals[vals > 0].sum())
            tops = [{"pair": [classes[iu[0][i]], classes[iu[1][i]]], "extra_errors_per_test_year": round(float(vals[i]), 2),
                     "share_of_net_extra": round(float(vals[i] / pos_total), 3) if pos_total > 0 else None}
                    for i in order[:5]]
            fresh_conf = sum(A["CONF"][u, u] for u in us) / len(us)
            n_per_class = fresh_conf.sum(1)
            recall_change = {classes[c]: round(float(-delta[c, c] / n_per_class[c] * 100), 3) for c in range(K)}
            # concentration: share of net extra pairwise confusion carried by the top 3 of 45 pairs
            res[str(g)] = {"total_extra_errors_per_test_year": round(float(-np.trace(delta)), 2),
                           "top_pairs": tops,
                           "top3_share": round(float(vals[order[:3]].clip(min=0).sum() / pos_total), 3) if pos_total > 0 else None,
                           "recall_loss_points_by_class": recall_change,
                           "n_pairs_increase": int((vals > 0).sum()), "n_pairs_total": int(len(vals))}
        out["models"][m] = res
        print(ds, m, "done", flush=True)
    return out


# ---------------------------------------------------------------- R-A4 post type (Stack Exchange)
def posttype(ds: str, models: list[str]) -> dict:
    assert ds == "stackexchange"
    d = Data(ds)
    import pickle
    s = pickle.load(open(ROOT / "work" / f"sample_{ds}.pkl", "rb"))
    is_post = np.array([it.startswith("post:") for it in s["item"]])
    Y, S, N = d.years, d.sizes, d.N
    gmax = Y[-1] - Y[0]
    shares = {}
    for part, name in ((0, "training_pool"), (2, "test")):
        shares[name] = {str(y): round(float(is_post[(d.part == part) & (d.year == y)].mean()), 4) for y in Y}
    out = {"post_share": shares, "models": {}}
    for m in models:
        out["models"][m] = {}
        for label, mask in (("posts", is_post), ("comments", ~is_post)):
            sc = CellScorer(d, 2, mask=mask)
            A = load_grid(ds, m, REPS5, d, sc)
            res = {"same_year_f1": round(float(np.mean(np.diag(A["P"]))), 5)}
            for g in (1, gmax):
                us = gap_sets(Y, g)
                cost = float(np.mean([A["P"][u, u] - A["P"][u - g, u] for u in us]))
                cost_b = np.mean([A["Pb"][u, u] - A["Pb"][u - g, u] for u in us], axis=0)
                pt, bt, *_ = exchange(S, g, A)
                lo, hi = censored_ci(bt)
                res[str(g)] = {"cost_points": round(cost * 100, 3), "cost_ci_points": [round(x * 100, 3) for x in pct_ci(cost_b)],
                               "n_star": nfmt(pt, N), "n_star_ci": [nfmt(lo, N), nfmt(hi, N)],
                               "rho": None if (pt == 0 or np.isinf(pt)) else round(pt / N, 4)}
            out["models"][m][label] = res
            print(ds, m, label, "done", flush=True)
    return out


if __name__ == "__main__":
    ds, what = sys.argv[1], sys.argv[2]
    models = sys.argv[3:] or ["M1"]
    fn = {"classwise": classwise, "curves": curves, "confusion": confusion, "posttype": posttype}[what]
    res = fn(ds, models)
    json.dump(res, open(OUT / f"{what}_{ds}.json", "w"), indent=1, default=lambda o: None if o is None else str(o))
    print("written", OUT / f"{what}_{ds}.json")
