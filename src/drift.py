"""E4 (descriptive): vocabulary drift between years and its association with per-class F1 loss.

For each class and each ordered pair of years (t < u): Jensen-Shannon divergence (base 2) between the unigram
distributions of the two training pools, and the per-class F1 loss of the primary model,
F1_c(u -> u) - F1_c(t -> u). Spearman correlation over all (class, t, u) cells. No causal claim.
python drift.py <dataset>  -> results/drift_<dataset>.json
"""
import json
import pickle
import sys
from collections import Counter

import numpy as np
from scipy.stats import spearmanr

from common import RESULTS, WORK


def jsd(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main(dataset):
    s = pickle.load(open(WORK / f"sample_{dataset}.pkl", "rb"))
    res = json.load(open(RESULTS / f"results_{dataset}.json"))
    years, K = s["years"], len(s["classes"])
    pc = np.array(res["models"]["M1"]["per_class_f1_grid"])          # [train year, test year, class]
    vocab, counts = {}, {}
    for c in range(K):
        for y in years:
            cnt = Counter()
            for i in np.flatnonzero((s["cls"] == c) & (s["year"] == y) & (s["part"] == 0)):
                cnt.update(s["tok"][i].split())
            counts[(c, y)] = cnt
            for w in cnt:
                vocab.setdefault(w, len(vocab))
    dist = {}
    for k, cnt in counts.items():
        v = np.zeros(len(vocab))
        for w, n in cnt.items():
            v[vocab[w]] = n
        dist[k] = v / v.sum()
    cells = []
    for c in range(K):
        for ti, t in enumerate(years):
            for ui, u in enumerate(years):
                if t < u:
                    cells.append({"class": s["classes"][c], "train": t, "test": u, "gap": u - t,
                                  "jsd": round(jsd(dist[(c, t)], dist[(c, u)]), 5),
                                  "f1_loss": round(float(pc[ui, ui, c] - pc[ti, ui, c]), 5)})
    j = np.array([x["jsd"] for x in cells]); l = np.array([x["f1_loss"] for x in cells]); g = np.array([x["gap"] for x in cells])
    r_all = spearmanr(j, l)
    r_gap = spearmanr(g, j)
    by_gap = {str(k): {"mean_jsd": round(float(j[g == k].mean()), 5), "mean_f1_loss": round(float(l[g == k].mean()), 5)} for k in sorted(set(g))}
    by_class = {}
    for c in s["classes"]:
        m = np.array([x["class"] == c for x in cells])
        by_class[c] = {"mean_jsd_gap1": round(float(j[m & (g == 1)].mean()), 5), "mean_f1_loss_gap1": round(float(l[m & (g == 1)].mean()), 5),
                       "mean_f1_loss_all_gaps": round(float(l[m].mean()), 5), "same_year_f1": round(float(np.mean([pc[i, i, s["classes"].index(c)] for i in range(len(years))])), 5)}
    out = {"n_cells": len(cells), "spearman_jsd_vs_f1_loss": {"rho": round(float(r_all[0]), 4), "p": float(r_all[1])},
           "spearman_gap_vs_jsd": {"rho": round(float(r_gap[0]), 4), "p": float(r_gap[1])}, "by_gap": by_gap, "by_class": by_class, "cells": cells}
    json.dump(out, open(RESULTS / f"drift_{dataset}.json", "w"), indent=1)
    print({k: out[k] for k in ("n_cells", "spearman_jsd_vs_f1_loss", "spearman_gap_vs_jsd", "by_gap")})


if __name__ == "__main__":
    main(sys.argv[1])
