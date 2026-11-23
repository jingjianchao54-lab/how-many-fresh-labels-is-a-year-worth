"""Figure 9 (amendment RV-02): learning curves of the primary model on the three tasks and the label-equivalence ratio
by gap on arXiv-8. Drawn only from the registered result files (results/ and revision/rv02/server_results/), with the
style of Figures 1-8 (make_figures_rev.py): same colours, markers, line styles and model labels.

python make_figure_rv02.py -> revision/figures/fig9_arxiv.{pdf,svg,png} and fig9_arxiv_data.csv
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from make_figures_rev import INK, MUTED, RES, ROOT, STY, W2, legend_below, num, save

RV = ROOT / "revision" / "rv02" / "server_results"
TASKS = [("reddit", "Reddit-10", RES / "results_reddit.json", INK, "o", "-"),
         ("stackexchange", "Stack Exchange-10", RES / "results_stackexchange.json", MUTED, "s", "--"),
         ("arxiv", "arXiv-8", RV / "results_arxiv.json", "#56B4E9", "^", "-.")]


def arxiv_models() -> dict:
    m = dict(json.load(open(RV / "results_arxiv.json"))["models"])
    a03 = json.load(open(RV / "results_arxiv_A03.json"))["models"]
    assert a03["M1"] == m["M1"]
    m["M5F"], m["M6"] = a03["M5F"], a03["M6"]
    return m


def main() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.9), layout="constrained")
    rows = []
    allsizes = set()
    for ds, name, path, col, mk, ls in TASKS:
        r = json.load(open(path))
        x = r["models"]["M1"]
        sizes = [int(k) for k in x["fresh_curve"]]
        allsizes |= set(sizes)
        f1 = [x["fresh_curve"][str(n)] for n in sizes]
        gmax = r["meta"]["years"][-1] - r["meta"]["years"][0]
        stale = x["exchange"][str(gmax)]["stale_macro_f1"]
        a.plot(sizes, f1, color=col, marker=mk, ls=ls, ms=3.8, lw=1.1, label=f"{name}: current labels")
        a.hlines(stale, sizes[0], sizes[-1], color=col, ls=(0, (1, 1.5)), lw=1.0)
        rows += [["a", ds, "fresh", n, round(v, 4), "", ""] for n, v in zip(sizes, f1)]
        rows += [["a", ds, f"stale_g{gmax}", sizes[-1], round(stale, 4), "", ""]]
    a.plot([], [], color=MUTED, ls=(0, (1, 1.5)), lw=1.0, label="N old labels at the largest gap")
    S = sorted(allsizes)
    a.set_xscale("log")
    a.set_xticks(S, [f"{s:,}" for s in S])
    a.minorticks_off()
    a.tick_params(axis="x", labelrotation=45)
    a.set_xlabel("Current training labels per class, n (log scale)")
    a.set_ylabel("Macro-F1 on the test year")
    a.set_title("(a) Primary model M1, three tasks")
    m = arxiv_models()
    G = list(range(1, 8))
    mids = ["M1", "M2", "M3", "M4", "M5F", "M6"]
    offs = {mid: (k - (len(mids) - 1) / 2) * 0.08 for k, mid in enumerate(mids)}
    for mid in mids:
        name, col, mk, ls = STY[mid]
        xs, ys, lo, hi, cens = [], [], [], [], []
        for g in G:
            e = m[mid]["exchange"][str(g)]
            rows.append(["b", "arxiv", mid, g, e["n_star"], e["n_star_ci"][0], e["n_star_ci"][1]])
            v = num(e["n_star"])
            if v is None:                                   # censored point, shown with an open triangle
                b.plot(g + offs[mid], 1.055, marker="^", mfc="white", mec=col, ms=4.3, ls="none", zorder=5)
                continue
            lo_, hi_ = num(e["n_star_ci"][0]), num(e["n_star_ci"][1])
            xs.append(g + offs[mid]); ys.append(v / 500); lo.append((lo_ or 0) / 500)
            hi.append(1.0 if hi_ is None else hi_ / 500); cens.append(hi_ is None)
        xs, ys, lo, hi, cens = map(np.array, (xs, ys, lo, hi, cens))
        fill = "white" if mid == "M5F" else col
        b.plot(xs, ys, color=col, ls=ls, lw=1.1, zorder=3)
        if mid == "M1":
            b.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt=mk, color=col, ms=3.8, mfc=fill, mec=col, mew=0.8,
                       elinewidth=0.7, capsize=1.6, zorder=4, label=f"{mid} {name}")
            for xc in xs[cens]:                             # upper bound censored above N: arrow, as in Figure 4
                b.annotate("", xy=(xc, 1.08), xytext=(xc, 0.995),
                           arrowprops=dict(arrowstyle="-|>", color=col, lw=0.7, mutation_scale=6, shrinkA=0, shrinkB=0))
        else:
            b.plot(xs, ys, mk, color=col, ms=3.8, mfc=fill, mec=col, mew=0.8, ls="none", zorder=4, label=f"{mid} {name}")
    b.axhline(0.5, color=MUTED, lw=0.6, ls=(0, (1, 2)))
    b.axhline(1.0, color=MUTED, lw=0.5)
    b.set_ylim(0, 1.1)
    b.set_xticks(G)
    b.set_xlabel("Age of training labels, g (years)")
    b.set_ylabel("Label-equivalence ratio ρ(g)")
    b.set_title("(b) arXiv-8, all models, N = 500")
    ha, la = a.get_legend_handles_labels()
    hb, lb = [], []
    for mid in mids:
        name, col, mk, ls = STY[mid]
        hb.append(Line2D([], [], color=col, ls=ls, lw=1.1, marker=mk, ms=3.8, mfc="white" if mid == "M5F" else col, mec=col))
        lb.append(f"{mid} {name}" + (" (with 95% interval)" if mid == "M1" else ""))
    fig.legend(ha + hb, la + lb, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.02), handlelength=2.8,
               columnspacing=1.2)
    save(fig, "fig9_arxiv", rows, ["panel", "dataset", "series", "x", "value", "ci_low", "ci_high"])
    print("fig9 written")


if __name__ == "__main__":
    main()
