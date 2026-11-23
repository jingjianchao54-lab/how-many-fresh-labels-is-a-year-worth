"""Shared helpers for the Round-1 revision analyses (stored test predictions only; no model is refitted).

Every function re-uses the registered quantities of src/evaluate.py (Macro-F1, breakeven, percentile rule X-02).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate import B, SEED, Scorer, breakeven, macro_f1_from_conf  # noqa: E402
from run import Data  # noqa: E402

RUNS = ROOT / "work" / "runs"
REPS5 = [1, 2, 3, 4, 5]


class CellScorer:
    """Stratified bootstrap within (class, year) cells, optionally restricted to a row mask and optionally
    resampling clusters (authors) instead of items. With mask=None and clusters=None use evaluate.Scorer."""

    def __init__(self, d: Data, part: int, mask: np.ndarray | None = None, clusters: np.ndarray | None = None,
                 seed: int = SEED) -> None:
        self.d, self.K = d, d.K
        rng = np.random.default_rng(seed)
        self.y, self.w, self.keep = {}, {}, {}
        for u in d.years:
            rows = d.rows(part, u)
            keep = np.ones(len(rows), bool) if mask is None else mask[rows]
            y_all = d.cls[rows].astype(np.int64)
            w = np.zeros((B, len(rows)), dtype=np.float32)
            for c in range(self.K):
                idx = np.flatnonzero((y_all == c) & keep)
                if clusters is None:
                    draw = rng.integers(0, len(idx), size=(B, len(idx)))
                    for b in range(B):
                        w[b, idx] = np.bincount(draw[b], minlength=len(idx))
                else:
                    cl = clusters[rows][idx]
                    uniq, inv = np.unique(cl, return_inverse=True)
                    draw = rng.integers(0, len(uniq), size=(B, len(uniq)))
                    for b in range(B):
                        cw = np.bincount(draw[b], minlength=len(uniq))   # times each author is drawn
                        w[b, idx] = cw[inv]
            self.y[u], self.w[u], self.keep[u] = y_all, w, keep

    def score(self, u: int, pred: np.ndarray):
        y, K, keep = self.y[u], self.K, self.keep[u]
        code = y * K + pred.astype(np.int64)
        conf = np.bincount(code[keep], minlength=K * K).reshape(K, K).astype(np.float64)
        cb = np.zeros((B, K * K))
        for k in np.unique(code):
            sel = code == k
            cb[:, k] = self.w[u][:, sel].sum(1)
        f, a, pc = macro_f1_from_conf(conf)
        fb, _, pcb = macro_f1_from_conf(cb.reshape(B, K, K))
        return float(f), float(a), fb, pc, pcb


def load_grid(dataset: str, model: str, reps: list[int], d: Data, sc, want_conf: bool = False):
    """Return replicate-mean point and bootstrap arrays for E1 (P, Pb, PC, PCB) and E2 (F, Fb, FC, FCB).
    P[t,u], Pb[t,u,B], PC[t,u,K], PCB[t,u,B,K]; F[t,n], Fb[t,n,B] with F[t,-1] = P[t,t]."""
    Y, S, N, K = d.years, d.sizes, d.N, d.K
    ny, ns = len(Y), len(S)
    base = RUNS / dataset / "test" / model
    P = np.zeros((ny, ny)); Pb = np.zeros((ny, ny, B)); PC = np.zeros((ny, ny, K)); PCB = np.zeros((ny, ny, B, K))
    F = np.zeros((ny, ns)); Fb = np.zeros((ny, ns, B)); FC = np.zeros((ny, ns, K)); FCB = np.zeros((ny, ns, B, K))
    CONF = np.zeros((ny, ny, K, K)) if want_conf else None
    nr = len(reps)
    for r in reps:
        for ti, t in enumerate(Y):
            z = np.load(base / f"e1_{r}_{t}_{N}_None.npz")
            for ui, u in enumerate(Y):
                f, _, fb, pc, pcb = sc.score(u, z["pred"][ui])
                P[ti, ui] += f / nr; Pb[ti, ui] += fb / nr; PC[ti, ui] += pc / nr; PCB[ti, ui] += pcb / nr
                if want_conf:
                    y = sc.y[u]
                    CONF[ti, ui] += np.bincount(y * K + z["pred"][ui].astype(np.int64), minlength=K * K).reshape(K, K) / nr
            for ni, n in enumerate(S[:-1]):
                z = np.load(base / f"e2_{r}_{t}_{n}_None.npz")
                f, _, fb, pc, pcb = sc.score(t, z["pred"][0])
                F[ti, ni] += f / nr; Fb[ti, ni] += fb / nr; FC[ti, ni] += pc / nr; FCB[ti, ni] += pcb / nr
    for ti in range(ny):
        F[ti, -1], Fb[ti, -1], FC[ti, -1], FCB[ti, -1] = P[ti, ti], Pb[ti, ti], PC[ti, ti], PCB[ti, ti]
    return dict(P=P, Pb=Pb, PC=PC, PCB=PCB, F=F, Fb=Fb, FC=FC, FCB=FCB, CONF=CONF)


def pct_ci(x: np.ndarray) -> list[float]:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def censored_ci(bt: np.ndarray) -> list[float]:
    """Percentile interval of a break-even sample keeping censored resamples censored (rule X-02)."""
    lo, hi = np.percentile(np.where(np.isinf(bt), 1e12, bt), [2.5, 97.5])
    return [float(np.inf if lo >= 1e9 else lo), float(np.inf if hi >= 1e9 else hi)]


def gap_sets(Y: list[int], g: int) -> list[int]:
    return [ui for ui in range(len(Y)) if ui - g >= 0]


def exchange(S, g, A):
    """Point and bootstrap n*(g) from arrays of load_grid (Macro-F1)."""
    Y = list(range(A["P"].shape[0]))
    us = gap_sets(Y, g)
    fresh, stale = A["F"][us].mean(0), np.mean([A["P"][u - g, u] for u in us])
    fresh_b, stale_b = A["Fb"][us].mean(0), np.mean([A["Pb"][u - g, u] for u in us], axis=0)
    pt = breakeven(S, fresh, stale)
    bt = np.array([breakeven(S, fresh_b[:, b], stale_b[b]) for b in range(B)])
    return pt, bt, fresh, stale, fresh_b, stale_b


def holm(p: dict[str, float]) -> dict[str, float]:
    order = sorted(p, key=p.get)
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * p[k]))
        out[k] = run
    return out
