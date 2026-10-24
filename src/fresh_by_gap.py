"""Point estimates of the current-year learning curve averaged over the test years U_g of each gap (X-06).

evaluate.py computes these curves internally for the break-even size but stores only the all-year curve.
This script recomputes the same point estimates from the existing test predictions (no new model fit,
no bootstrap) so that Figure 3 and the slope discussion use exactly the curve behind n*(g).

python fresh_by_gap.py <dataset> -> results/fresh_by_gap_<dataset>.json
python fresh_by_gap.py <dataset> A03 -> results/fresh_by_gap_<dataset>_A03.json (M5F and M6, amendment A-03; does not touch the first file)
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


def main(dataset, tag=None):
    d = Data(dataset)
    Y, S, N, K = d.years, d.sizes, d.N, d.K
    ys = {u: d.cls[d.rows(2, u)].astype(np.int64) for u in Y}
    out = {"dataset": dataset, "sizes": S, "models": {}}
    for m in (("M5F", "M6") if tag == "A03" else ("M1", "M2", "M3", "M4", "M5")):
        reps = [1] if m == "M5" else [1, 2, 3, 4, 5]
        base = WORK / "runs" / dataset / "test" / m
        F = np.zeros((len(reps), len(Y), len(S)))
        for ri, r in enumerate(reps):
            for ti, t in enumerate(Y):
                F[ri, ti, -1] = f1(ys[t], np.load(base / f"e1_{r}_{t}_{N}_None.npz")["pred"][ti], K)
                for ni, n in enumerate(S[:-1]):
                    F[ri, ti, ni] = f1(ys[t], np.load(base / f"e2_{r}_{t}_{n}_None.npz")["pred"][0], K)
        Fm = F.mean(0)
        out["models"][m] = {"replicates": reps, "by_gap": {}}
        for g in range(1, Y[-1] - Y[0] + 1):
            us = [ui for ui in range(len(Y)) if ui - g >= 0]
            out["models"][m]["by_gap"][str(g)] = {"test_years": [Y[u] for u in us],
                                                  "fresh_curve": {str(n): round(float(v), 5) for n, v in zip(S, Fm[us].mean(0))}}
    path = RESULTS / (f"fresh_by_gap_{dataset}_{tag}.json" if tag else f"fresh_by_gap_{dataset}.json")
    if path.exists():
        sys.exit(f"{path.name} exists")
    json.dump(out, open(path, "w"), indent=1)
    print(dataset, {m: out["models"][m]["by_gap"]["6"]["fresh_curve"] for m in out["models"]})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
