"""Score the revision refits of M1 with the registered procedure (item-level stratified bootstrap, seed 20261009).

python b_eval.py <dataset> <variant> [mask]
A variant that refits only E2 (fixedC) or only E3 (mixw, mixtune) takes the other experiments from the registered
predictions. Writes revision/out/variant_<variant>_<dataset>.json with C(1), C(gmax), their difference, n*(1), n*(gmax),
the one-sided p-values and Holm adjustment of H1-H3, and (E3 variants) the mixing differences.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import common_rev as cr
from common_rev import B, REPS5, ROOT, CellScorer, Data, breakeven, censored_ci, holm, pct_ci

RW = ROOT / "revision" / "rwork"
OUT = ROOT / "revision" / "out"
E1_FROM_VARIANT = {"pii", "emoji", "sitemask", "widegrid"}
E2_FROM_VARIANT = {"pii", "emoji", "sitemask", "fixedC", "widegrid"}


def nf(v, N):
    return "<25" if v == 0 else (f">{N}" if np.isinf(v) else round(float(v), 1))


def grids(ds: str, variant: str, d: Data, sc) -> dict:
    Y, S, N, K = d.years, d.sizes, d.N, d.K
    ny, ns = len(Y), len(S)
    reg = cr.RUNS / ds / "test" / "M1"
    var = RW / variant / "runs" / ds / "test" / "M1"
    e1dir = var if variant in E1_FROM_VARIANT else reg
    e2dir = var if variant in E2_FROM_VARIANT else reg
    P = np.zeros((ny, ny)); Pb = np.zeros((ny, ny, B)); F = np.zeros((ny, ns)); Fb = np.zeros((ny, ns, B))
    gmax = Y[-1] - Y[0]
    G = {g: np.zeros((ny, ns - 1)) for g in (1, gmax)}
    Gb = {g: np.zeros((ny, ns - 1, B)) for g in (1, gmax)}
    for r in REPS5:
        for ti, t in enumerate(Y):
            z = np.load(e1dir / f"e1_{r}_{t}_{N}_None.npz")
            for ui, u in enumerate(Y):
                f, _, fb, _, _ = sc.score(u, z["pred"][ui])
                P[ti, ui] += f / 5; Pb[ti, ui] += fb / 5
            for ni, n in enumerate(S[:-1]):
                z = np.load(e2dir / f"e2_{r}_{t}_{n}_None.npz")
                f, _, fb, _, _ = sc.score(t, z["pred"][0])
                F[ti, ni] += f / 5; Fb[ti, ni] += fb / 5
                if variant in ("mixw", "mixtune"):
                    for g in (1, gmax):
                        p = var / f"e3_{r}_{t}_{n}_{g}.npz"
                        if p.exists():
                            f, _, fb, _, _ = sc.score(t, np.load(p)["pred"][0])
                            G[g][ti, ni] += f / 5; Gb[g][ti, ni] += fb / 5
    for ti in range(ny):
        F[ti, -1], Fb[ti, -1] = P[ti, ti], Pb[ti, ti]
    return dict(P=P, Pb=Pb, F=F, Fb=Fb, G=G, Gb=Gb)


def main() -> None:
    ds, variant = sys.argv[1], sys.argv[2]
    mask_name = sys.argv[3] if len(sys.argv) > 3 else None
    d = Data(ds)
    mask = None
    if mask_name:
        mask = np.load(OUT / f"mask_{mask_name}_{ds}.npy")
    sc = CellScorer(d, 2, mask=mask)
    runs_variant = "registered_preds" if variant == "english" else variant
    A = grids(ds, runs_variant, d, sc)
    Y, S, N = d.years, d.sizes, d.N
    ny, gmax = len(Y), Y[-1] - Y[0]
    out = {"variant": variant, "mask": mask_name, "same_year_f1": round(float(np.mean(np.diag(A["P"]))), 5)}
    cb = {}
    for g in (1, gmax):
        pairs = [(u - g, u) for u in range(ny) if u - g >= 0]
        c = float(np.mean([A["P"][u, u] - A["P"][t, u] for t, u in pairs]))
        cb[g] = np.mean([A["Pb"][u, u] - A["Pb"][t, u] for t, u in pairs], axis=0)
        us = [u for u in range(ny) if u - g >= 0]
        fresh, stale = A["F"][us].mean(0), np.mean([A["P"][u - g, u] for u in us])
        fb, sb = A["Fb"][us].mean(0), np.mean([A["Pb"][u - g, u] for u in us], axis=0)
        p = breakeven(S, fresh, stale)
        bb = np.array([breakeven(S, fb[:, b], sb[b]) for b in range(B)])
        lo, hi = censored_ci(bb)
        out[str(g)] = {"cost_points": round(c * 100, 3), "cost_ci_points": [round(x * 100, 3) for x in pct_ci(cb[g])],
                       "n_star": nf(p, N), "n_star_ci": [nf(lo, N), nf(hi, N)],
                       "rho": None if (p == 0 or np.isinf(p)) else round(p / N, 4)}
        if g == gmax:
            nb = bb
        if variant in ("mixw", "mixtune") and g == gmax:
            mx = {}
            for ni, n in enumerate(S[:-1]):
                m_ = float(np.mean([A["G"][g][u, ni] for u in us])); mb = np.mean([A["Gb"][g][u, ni] for u in us], axis=0)
                fr = float(A["F"][us, ni].mean()); frb = A["Fb"][us, ni].mean(0)
                mx[str(n)] = {"mix": round(m_, 5), "mix_minus_fresh_points": [round((m_ - fr) * 100, 3)] + [round(x * 100, 3) for x in pct_ci(mb - frb)],
                              "mix_minus_stale_points": [round((m_ - stale) * 100, 3)] + [round(x * 100, 3) for x in pct_ci(mb - sb)]}
            out[str(g)]["mixing"] = mx
    db = cb[gmax] - cb[1]
    out["diff_points"] = [round((out[str(gmax)]["cost_points"] - out["1"]["cost_points"]), 3)] + [round(x * 100, 3) for x in pct_ci(db)]
    pv = {"H1": float(np.mean(cb[1] <= 0)), "H2": float(np.mean(db <= 0)), "H3": float(np.mean(nb >= N / 2))}
    out["p_one_sided"] = {k: round(v, 4) for k, v in pv.items()}
    out["holm"] = {k: round(v, 4) for k, v in holm(pv).items()}
    tag = variant + (f"_{mask_name}" if mask_name else "")
    json.dump(out, open(OUT / f"variant_{tag}_{ds}.json", "w"), indent=1)
    print(tag, ds, json.dumps(out)[:600])


if __name__ == "__main__":
    main()
