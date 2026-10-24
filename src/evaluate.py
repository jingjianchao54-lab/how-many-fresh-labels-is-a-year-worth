"""Scoring for E1-E3 with a paired, stratified bootstrap (plan Sections 5 and 6).

python evaluate.py <dataset> <stage> [models...]     stage: "test" (locked) or "dry" (validation sets, replicate 1)
Writes results/results_<dataset>[_dry].json and results/summary_<dataset>[_dry].csv.
"""
import csv
import json
import pickle
import sys

import numpy as np

from common import RESULTS, WORK
from run import Data

B = 2000
SEED = 20261009


def macro_f1_from_conf(conf):
    """conf[..., K, K] (truth, prediction) -> macro-F1 and accuracy over the leading axes."""
    tp = np.einsum("...ii->...i", conf)
    fp = conf.sum(-2) - tp
    fn = conf.sum(-1) - tp
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)
    return f1.mean(-1), tp.sum(-1) / conf.sum((-1, -2)), f1


class Scorer:
    """Holds bootstrap weights per evaluation year; scores a prediction vector on the point sample and all resamples."""

    def __init__(self, d, part):
        self.d, self.K = d, d.K
        rng = np.random.default_rng(SEED)
        self.y, self.w = {}, {}
        for u in d.years:
            rows = d.rows(part, u)
            y = d.cls[rows].astype(np.int64)
            w = np.zeros((B, len(y)), dtype=np.float32)
            for c in range(self.K):                      # resample within each (class, year) cell
                idx = np.flatnonzero(y == c)
                draw = rng.integers(0, len(idx), size=(B, len(idx)))
                for b in range(B):
                    w[b, idx] = np.bincount(draw[b], minlength=len(idx))
            self.y[u], self.w[u] = y, w

    def score(self, u, pred):
        """Return (point macro-F1, point accuracy, bootstrap macro-F1 [B], per-class F1 point, per-class F1 boot)."""
        y, K = self.y[u], self.K
        code = y * K + pred.astype(np.int64)
        conf = np.bincount(code, minlength=K * K).reshape(K, K).astype(np.float64)
        order = np.argsort(code, kind="stable")
        starts = np.searchsorted(code[order], np.arange(K * K))
        cb = np.zeros((B, K * K))
        present = np.flatnonzero(np.bincount(code, minlength=K * K) > 0)
        red = np.add.reduceat(self.w[u][:, order], starts[present], axis=1)
        cb[:, present] = red
        f, a, pc = macro_f1_from_conf(conf)
        fb, _, pcb = macro_f1_from_conf(cb.reshape(B, K, K))
        return float(f), float(a), fb, pc, pcb


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [round(float(lo), 5), round(float(hi), 5)]


def breakeven(sizes, fresh, stale):
    """Smallest n (linear interpolation in log n) at which the fresh curve reaches the stale score.
    Returns np.inf if never reached at the largest size, 0.0 if already reached at the smallest."""
    sizes = np.asarray(sizes, float)
    if fresh[0] >= stale:
        return 0.0
    for i in range(1, len(sizes)):
        if fresh[i] >= stale:
            f0, f1 = fresh[i - 1], fresh[i]
            a = (stale - f0) / (f1 - f0)
            return float(np.exp(np.log(sizes[i - 1]) + a * (np.log(sizes[i]) - np.log(sizes[i - 1]))))
    return np.inf


def fmt_n(v, sizes):
    if v == 0.0:
        return f"<{sizes[0]}"
    if np.isinf(v):
        return f">{sizes[-1]}"
    return round(float(v), 1)


def main():
    dataset, stage = sys.argv[1:3]
    models = sys.argv[3:] or ["M1", "M2", "M3", "M4"]
    tag = sys.argv[sys.argv.index("--field") + 1] if "--field" in sys.argv else "tok"
    skip = {sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--out" and i + 1 < len(sys.argv)}
    models = [m for m in models if not m.startswith("--") and m != tag and m not in skip]
    d = Data(dataset, tag)
    part = 1 if stage == "dry" else 2
    reps = [1] if (stage == "dry" or "--reps1" in sys.argv) else [1, 2, 3, 4, 5]     # --reps1: M5 (A-02) and comparisons with it
    sc = Scorer(d, part)
    Y, S, N = d.years, d.sizes, d.N
    gmax = Y[-1] - Y[0]
    run_dir = WORK / "runs" / (dataset if tag == "tok" else f"{dataset}_{tag}") / stage
    res, rows = {}, []
    for m in models:
        base = run_dir / m
        ny, ns = len(Y), len(S)
        P = np.zeros((len(reps), ny, ny)); A = np.zeros_like(P); Pb = np.zeros((ny, ny, B))
        PC = np.zeros((ny, ny, d.K))
        F = np.zeros((len(reps), ny, ns)); Fb = np.zeros((ny, ns, B))
        G = {g: np.full((len(reps), ny, ns - 1), np.nan) for g in sorted({1, gmax})}
        Gb = {g: np.zeros((ny, ns - 1, B)) for g in G}
        nconv = 0
        for ri, r in enumerate(reps):
            for ti, t in enumerate(Y):
                z = np.load(base / f"e1_{r}_{t}_{N}_None.npz")
                nconv += int(not bool(z["ok"]))
                for ui, u in enumerate(Y):
                    f, a, fb, pc, _ = sc.score(u, z["pred"][ui])
                    P[ri, ti, ui], A[ri, ti, ui] = f, a
                    Pb[ti, ui] += fb / len(reps)
                    PC[ti, ui] += pc / len(reps)
                F[ri, ti, -1] = P[ri, ti, ti]
                for ni, n in enumerate(S[:-1]):
                    z = np.load(base / f"e2_{r}_{t}_{n}_None.npz")
                    nconv += int(not bool(z["ok"]))
                    f, _, fb, _, _ = sc.score(t, z["pred"][0])
                    F[ri, ti, ni] = f
                    Fb[ti, ni] += fb / len(reps)
                    for g in G:
                        if t - g >= Y[0] and (base / f"e3_{r}_{t}_{n}_{g}.npz").exists():
                            z = np.load(base / f"e3_{r}_{t}_{n}_{g}.npz")
                            nconv += int(not bool(z["ok"]))
                            f, _, fb, _, _ = sc.score(t, z["pred"][0])
                            G[g][ri, ti, ni] = f
                            Gb[g][ti, ni] += fb / len(reps)
        for ti in range(ny):
            Fb[ti, -1] = Pb[ti, ti]
        e3_complete = all(np.isnan(G[g]).sum() == np.isnan(G[g][0:1]).sum() * len(reps) for g in G)   # A-03
        if not e3_complete:
            G = {}                                   # E3 not run for every scored replicate: not summarised here
        Pm, Fm = P.mean(0), F.mean(0)                                # replicate means
        out = {"grid_macro_f1": {str(t): {str(u): round(float(Pm[ti, ui]), 5) for ui, u in enumerate(Y)} for ti, t in enumerate(Y)},
               "grid_accuracy": {str(t): {str(u): round(float(A.mean(0)[ti, ui]), 5) for ui, u in enumerate(Y)} for ti, t in enumerate(Y)},
               "grid_replicate_sd_max": round(float(P.std(0, ddof=1).max()), 5) if len(reps) > 1 else None,
               "fits_at_max_iter": nconv, "staleness": {}, "exchange": {}, "mixing": {}, "fresh_curve": {}}
        same = float(np.mean(np.diag(Pm)))
        out["same_year_macro_f1"] = round(same, 5)
        # E1 staleness cost, forward (train earlier) and backward (train later)
        Cb_store = {}
        for g in range(1, gmax + 1):
            for name, sgn in (("forward", 1), ("backward", -1)):
                pairs = [(ti - sgn * g, ti) for ti in range(ny) if 0 <= ti - sgn * g < ny]   # (train idx, test idx)
                pt = np.mean([Pm[u, u] - Pm[t, u] for t, u in pairs])
                bt = np.mean([Pb[u, u] - Pb[t, u] for t, u in pairs], axis=0)
                rep_sd = np.std([np.mean([P[ri, u, u] - P[ri, t, u] for t, u in pairs]) for ri in range(len(reps))], ddof=1) if len(reps) > 1 else None
                out["staleness"].setdefault(name, {})[str(g)] = {"cost": round(float(pt), 5), "ci": ci(bt), "n_pairs": len(pairs),
                                                                  "replicate_sd": None if rep_sd is None else round(float(rep_sd), 5)}
                if name == "forward":
                    Cb_store[g] = (pt, bt)
                rows.append([dataset, m, f"staleness_{name}", g, round(float(pt), 5)] + ci(bt))
        dpt = Cb_store[gmax][0] - Cb_store[1][0]
        dbt = Cb_store[gmax][1] - Cb_store[1][1]
        out["staleness"]["forward_gmax_minus_g1"] = {"diff": round(float(dpt), 5), "ci": ci(dbt)}
        # temporal degradation score of Luu et al. (2022): mean absolute slope of D(t'->u) on t'
        td = []
        for ui in range(ny):
            D = np.array([-(Pm[ti, ui] - Pm[ui, ui]) * np.sign(Y[ti] - Y[ui]) for ti in range(ny)])
            td.append(abs(np.polyfit(np.array(Y, float), D, 1)[0]))
        out["td_score_per_year_f1_points"] = round(float(np.mean(td)) * 100, 3)
        # fresh learning curve (mean over all years)
        out["fresh_curve"] = {str(n): round(float(Fm[:, ni].mean()), 5) for ni, n in enumerate(S)}
        # E2 exchange rate
        for g in range(1, gmax + 1):
            us = [ui for ui in range(ny) if ui - g >= 0]
            fresh = Fm[us].mean(0); stale = np.mean([Pm[ui - g, ui] for ui in us])
            fresh_b = Fb[us].mean(0); stale_b = np.mean([Pb[ui - g, ui] for ui in us], axis=0)
            pt = breakeven(S, fresh, stale)
            bt = np.array([breakeven(S, fresh_b[:, b], stale_b[b]) for b in range(B)])
            lo, hi = np.percentile(np.where(np.isinf(bt), 1e12, bt), [2.5, 97.5])
            lo, hi = (np.inf if lo >= 1e9 else lo), (np.inf if hi >= 1e9 else hi)   # X-02: a percentile interpolated between a finite and a censored resample is censored
            out["exchange"][str(g)] = {"stale_macro_f1": round(float(stale), 5), "n_star": fmt_n(pt, S),
                                        "n_star_ci": [fmt_n(lo, S), fmt_n(hi, S)],
                                        "rho": None if (pt == 0 or np.isinf(pt)) else round(pt / N, 4),
                                        "rho_ci": [None if (v == 0 or np.isinf(v)) else round(v / N, 4) for v in (lo, hi)],
                                        "share_resamples_censored_high": round(float(np.mean(np.isinf(bt))), 4),
                                        "n_test_years": len(us)}
            rows.append([dataset, m, "n_star", g, fmt_n(pt, S), fmt_n(lo, S), fmt_n(hi, S)])
        # E3 stale N + fresh n versus fresh-only n and stale-only N (same test years)
        for g in G:
            us = [ui for ui in range(ny) if ui - g >= 0]
            stale = np.mean([Pm[ui - g, ui] for ui in us]); stale_b = np.mean([Pb[ui - g, ui] for ui in us], axis=0)
            mix = np.nanmean(G[g], axis=0)
            for ni, n in enumerate(S[:-1]):
                mx = float(np.mean([mix[ui, ni] for ui in us])); mxb = np.mean([Gb[g][ui, ni] for ui in us], axis=0)
                fr = float(Fm[us, ni].mean()); frb = Fb[us, ni].mean(0)
                out["mixing"].setdefault(str(g), {})[str(n)] = {
                    "stale_plus_fresh": round(mx, 5), "fresh_only": round(fr, 5), "stale_only": round(float(stale), 5),
                    "mix_minus_fresh": [round(mx - fr, 5)] + ci(mxb - frb), "mix_minus_stale": [round(mx - float(stale), 5)] + ci(mxb - stale_b)}
                rows.append([dataset, m, f"mix_minus_fresh_g{g}", n, round(mx - fr, 5)] + ci(mxb - frb))
                rows.append([dataset, m, f"mix_minus_stale_g{g}", n, round(mx - float(stale), 5)] + ci(mxb - stale_b))
        us = [ui for ui in range(ny) if ui - gmax >= 0]
        nb = np.array([breakeven(S, Fb[us].mean(0)[:, b], np.mean([Pb[ui - gmax, ui] for ui in us], axis=0)[b]) for b in range(B)])
        pv = {"H1": float(np.mean(Cb_store[1][1] <= 0)), "H2": float(np.mean(dbt <= 0)), "H3": float(np.mean(nb >= N / 2))}
        order = sorted(pv, key=pv.get); holm, run = {}, 0.0
        for i, k in enumerate(order):
            run = max(run, min(1.0, (len(order) - i) * pv[k])); holm[k] = round(run, 4)
        out["hypotheses"] = {"bootstrap_p_one_sided": {k: round(v, 4) for k, v in pv.items()}, "holm_p": holm}
        out["per_class_f1_grid"] = np.round(PC, 5).tolist()
        res[m] = out
        print(m, "same-year", out["same_year_macro_f1"], "| C(1)", out["staleness"]["forward"]["1"], "| C(gmax)", out["staleness"]["forward"][str(gmax)],
              "| n*(1)", out["exchange"]["1"]["n_star"], out["exchange"]["1"]["n_star_ci"], "| n*(gmax)", out["exchange"][str(gmax)]["n_star"],
              out["exchange"][str(gmax)]["n_star_ci"], flush=True)
    suffix = ("" if tag == "tok" else f"_{tag}") + ("_dry" if stage == "dry" else "") + ("_rep1" if "--reps1" in sys.argv else "")
    if "--out" in sys.argv:                          # A-03: new scorings get their own file name
        suffix += "_" + sys.argv[sys.argv.index("--out") + 1]
    out_path = RESULTS / f"results_{dataset}{suffix}.json"
    if out_path.exists() and "--out" in sys.argv:
        sys.exit(f"{out_path.name} exists; not overwritten")
    meta = {"dataset": dataset, "stage": stage, "years": Y, "sizes": S, "N": N, "classes": d.K, "bootstrap": B, "seed": SEED,
            "replicates": reps, "eval_items_per_cell": d.V if stage == "dry" else d.T}
    json.dump({"meta": meta, "models": res}, open(RESULTS / f"results_{dataset}{suffix}.json", "w"), indent=1)
    with open(RESULTS / f"summary_{dataset}{suffix}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "quantity", "key", "value", "ci_low", "ci_high"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
