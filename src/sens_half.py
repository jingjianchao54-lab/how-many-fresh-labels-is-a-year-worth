"""E5 sensitivity: primary model with the stale training set at half the cell size (N/2 labels per class).

Stale score: model trained on N/2 labels from year u-g (run files h1_*), tested on year u.
Fresh curve: sizes n <= N/2 from year u (existing e2 files; the point n = N/2 is the diagonal of the h1 grid).
python sens_half.py <dataset>  -> results/sens_half_<dataset>.json
"""
import json
import sys

import numpy as np

import evaluate as ev
from common import RESULTS, WORK
from run import Data


def main(dataset):
    d = Data(dataset)
    sc = ev.Scorer(d, 2)
    Y, H = d.years, (int(sys.argv[2]) if len(sys.argv) > 2 else d.N // 2)
    S = [n for n in d.sizes if n <= H]
    base = WORK / "runs" / dataset / "test" / "M1"
    reps, ny, B = [1, 2, 3, 4, 5], len(Y), ev.B
    P = np.zeros((len(reps), ny, ny)); Pb = np.zeros((ny, ny, B))
    F = np.zeros((len(reps), ny, len(S))); Fb = np.zeros((ny, len(S), B))
    for ri, r in enumerate(reps):
        for ti, t in enumerate(Y):
            z = np.load(base / f"h1_{r}_{t}_{H}_None.npz")
            for ui, u in enumerate(Y):
                f, _, fb, _, _ = sc.score(u, z["pred"][ui])
                P[ri, ti, ui] = f; Pb[ti, ui] += fb / len(reps)
            F[ri, ti, -1] = P[ri, ti, ti]
            for ni, n in enumerate(S[:-1]):
                z = np.load(base / f"e2_{r}_{t}_{n}_None.npz")
                f, _, fb, _, _ = sc.score(t, z["pred"][0])
                F[ri, ti, ni] = f; Fb[ti, ni] += fb / len(reps)
    for ti in range(ny):
        Fb[ti, -1] = Pb[ti, ti]
    Pm, Fm = P.mean(0), F.mean(0)
    out = {"dataset": dataset, "stale_size": H, "sizes": S, "same_year_macro_f1": round(float(np.mean(np.diag(Pm))), 5), "by_gap": {}}
    for g in range(1, Y[-1] - Y[0] + 1):
        us = [ui for ui in range(ny) if ui - g >= 0]
        cost = np.mean([Pm[u, u] - Pm[u - g, u] for u in us]); cost_b = np.mean([Pb[u, u] - Pb[u - g, u] for u in us], axis=0)
        fresh = Fm[us].mean(0); stale = np.mean([Pm[u - g, u] for u in us])
        fresh_b = Fb[us].mean(0); stale_b = np.mean([Pb[u - g, u] for u in us], axis=0)
        pt = ev.breakeven(S, fresh, stale)
        bt = np.array([ev.breakeven(S, fresh_b[:, b], stale_b[b]) for b in range(B)])
        lo, hi = np.percentile(np.where(np.isinf(bt), 1e12, bt), [2.5, 97.5])
        lo, hi = (np.inf if lo >= 1e9 else lo), (np.inf if hi >= 1e9 else hi)
        out["by_gap"][str(g)] = {"cost": round(float(cost), 5), "cost_ci": ev.ci(cost_b), "n_star": ev.fmt_n(pt, S),
                                 "n_star_ci": [ev.fmt_n(lo, S), ev.fmt_n(hi, S)],
                                 "rho": None if (pt == 0 or np.isinf(pt)) else round(pt / H, 4)}
    json.dump(out, open(RESULTS / f"sens_half_{dataset}.json", "w"), indent=1)
    print(dataset, out["same_year_macro_f1"], {g: (v["cost"], v["n_star"], v["rho"]) for g, v in out["by_gap"].items()})


if __name__ == "__main__":
    main(sys.argv[1])
