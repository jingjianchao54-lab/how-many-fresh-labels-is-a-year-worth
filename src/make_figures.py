"""Figures for the manuscript, drawn only from the locked result files. Each figure also writes its source data as CSV.

python make_figures.py   -> figures/fig{1..5}.{pdf,png,tiff} and figures/fig{1..5}_data.csv
"""
import csv
import json

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS, ROOT

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "lines.linewidth": 1.2, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 600, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
DS = [("reddit", "Reddit-10"), ("stackexchange", "Stack Exchange-10")]
MODELS = [("M1", "M1 TF-IDF + LR", "#0072B2", "o", "-"), ("M2", "M2 TF-IDF + SVM", "#E69F00", "s", "--"),
          ("M3", "M3 Counts + CNB", "#009E73", "^", "-."), ("M4", "M4 GloVe + LR", "#D55E00", "D", ":"),
          ("M5F", "M5F MiniLM + LR", "#000000", "v", (0, (5, 1))), ("M6", "M6 Fine-tuned BERT", "#CC79A7", "P", (0, (3, 1, 1, 1)))]
# M5F and M6 (amendment A-03) replace the reduced-design M5 in Figures 2 and 4; M5 stays in Table 4.
INK, MUTED = "#222222", "#777777"
# diverging map for Figure 1: orange for negative loss (another year scored higher), white at zero, blue for loss
DIVERGE = mpl.colors.LinearSegmentedColormap.from_list(
    "loss", [(0.0, "#C76A00"), (0.5, "#FFFFFF"), (0.62, "#C6DBEF"), (0.8, "#4292C6"), (1.0, "#08306B")])


def load(ds):
    r = json.load(open(RESULTS / f"results_{ds}.json"))
    models = dict(r["models"])
    try:
        models["M5"] = json.load(open(RESULTS / f"results_{ds}_rep1.json"))["models"]["M5"]
    except FileNotFoundError:
        pass
    try:
        a03 = json.load(open(RESULTS / f"results_{ds}_A03.json"))["models"]
        assert a03["M1"] == models["M1"]
        models["M5F"], models["M6"] = a03["M5F"], a03["M6"]
    except FileNotFoundError:
        pass
    return r["meta"], models


def num(v, n_max):
    """Numeric value of a possibly censored break-even size (None if censored)."""
    return None if isinstance(v, str) else float(v)


def save(fig, name, rows, header):
    for ext in ("pdf", "png", "tiff"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    with open(FIG / f"{name}_data.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def fig1():
    """Loss relative to the same-year model, primary model, full train-year x test-year grid."""
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0), layout="constrained")
    rows = []
    for ax, (ds, title), tag in zip(axes, DS, "ab"):
        meta, m = load(ds)
        Y = meta["years"]
        g = np.array([[m["M1"]["grid_macro_f1"][str(t)][str(u)] for u in Y] for t in Y])
        loss = (np.diag(g)[None, :] - g) * 100          # F1 points lost versus training on the test year
        im = ax.imshow(loss, cmap=DIVERGE, norm=mpl.colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=3.0), aspect="equal")
        for i, t in enumerate(Y):
            for j, u in enumerate(Y):
                ax.text(j, i, f"{loss[i, j] + 0.0:.1f}".replace("-0.0", "0.0"), ha="center", va="center", fontsize=6,
                        color="white" if loss[i, j] > 1.8 else INK)
                rows.append([ds, t, u, round(float(g[i, j]), 5), round(float(loss[i, j]), 3)])
        ax.set_xticks(range(len(Y)), Y, rotation=45); ax.set_yticks(range(len(Y)), Y)
        ax.set_xlabel("Test year"); ax.set_ylabel("Training year")
        ax.set_title(f"({tag}) {title}", loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Macro-F1 points lost vs. same-year training"); cb.outline.set_visible(False)
    save(fig, "fig1_aging_grid", rows, ["dataset", "train_year", "test_year", "macro_f1", "points_lost_vs_same_year"])


def fig2():
    """Staleness cost by gap, every model."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    rows = []
    for ax, (ds, title), tag in zip(axes, DS, "ab"):
        meta, m = load(ds)
        G = sorted(int(g) for g in m["M1"]["staleness"]["forward"] if g.isdigit())
        for k, (mid, name, col, mk, ls) in enumerate(MODELS):
            if mid not in m:
                continue
            c = [m[mid]["staleness"]["forward"][str(g)] for g in G]
            y = np.array([v["cost"] for v in c]) * 100
            lo = np.array([v["ci"][0] for v in c]) * 100; hi = np.array([v["ci"][1] for v in c]) * 100
            x = np.array(G) + (k - 2.5) * 0.06
            ax.errorbar(x, y, yerr=[y - lo, hi - y], color=col, marker=mk, ms=3.5, ls=ls, lw=1.0, elinewidth=0.6,
                        capsize=1.5, label=name, mfc="white" if mid in ("M5", "M5F") else col)
            rows += [[ds, mid, g, round(a / 100, 5), round(b / 100, 5), round(d / 100, 5)] for g, a, b, d in zip(G, y, lo, hi)]
        ax.axhline(0, color=MUTED, lw=0.5)
        ax.axhline(1.0, color=MUTED, lw=0.5, ls=(0, (1, 2)))
        pass  # dotted line: the pre-specified practical threshold of 0.010 Macro-F1 (stated in the caption)
        ax.set_xticks(G); ax.set_xlabel("Age of training labels (years)")
        ax.set_title(f"({tag}) {title}", loc="left")
    axes[0].set_ylabel("Staleness cost (Macro-F1 points)")
    fig.subplots_adjust(bottom=0.30)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=3, handlelength=2.6, columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig2_staleness_cost", rows, ["dataset", "model", "gap_years", "cost", "ci_low", "ci_high"])


def fig3():
    """How the break-even size is read off the current-year learning curve (primary model).
    Each gap uses the curve averaged over its own test years U_g (results/fresh_by_gap_<ds>.json, X-06)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    rows = []
    shades = {1: "#56B4E9", 6: "#08306B"}
    for ax, (ds, title), tag in zip(axes, DS, "ab"):
        meta, m = load(ds)
        S, N = meta["sizes"], meta["N"]
        fb = json.load(open(RESULTS / f"fresh_by_gap_{ds}.json"))["models"]["M1"]["by_gap"]
        lo_all = 1.0
        for g in (1, 6):
            fc = [fb[str(g)]["fresh_curve"][str(n)] for n in S]
            lo_all = min(lo_all, min(fc))
            e = m["M1"]["exchange"][str(g)]
            ns = num(e["n_star"], N)
            ax.plot(S, fc, color=shades[g], marker="o", ms=3, lw=1.1, label=f"Current labels, test years with gap {g}")
            ax.axhline(e["stale_macro_f1"], color=shades[g], lw=0.9, ls="--")
            ax.text(1.01, e["stale_macro_f1"], f"{g} year{'s' if g > 1 else ''} old", transform=ax.get_yaxis_transform(), fontsize=6, color=INK,
                    va="bottom" if g == 1 else "top")
            if ns:
                ax.plot([ns, ns], [0, e["stale_macro_f1"]], color=shades[g], lw=0.9, ls=":")
                ax.plot([ns], [e["stale_macro_f1"]], marker="D", ms=4.5, color=shades[g], mec="white", mew=0.6, zorder=5)
            rows += [[ds, g, "current", n, v, ""] for n, v in zip(S, fc)]
            rows.append([ds, g, "stale_N", N, e["stale_macro_f1"], e["n_star"]])
        ax.plot([], [], color=MUTED, lw=0.9, ls="--", label="N old labels per class")
        ax.plot([], [], color=MUTED, marker="D", ms=4, ls=":", lw=0.9, label="Break-even size n*")
        ax.set_xscale("log"); ax.set_xticks(S, [f"{s:,}" for s in S], rotation=40, ha="right", rotation_mode="anchor"); ax.minorticks_off()
        ax.set_ylim(lo_all - 0.02, max(fb["1"]["fresh_curve"][str(N)], fb["6"]["fresh_curve"][str(N)]) + 0.02)
        ax.set_xlim(S[0] * 0.8, S[-1] * 1.25)
        ax.set_xlabel("Current training labels per class"); ax.set_title(f"({tag}) {title} (N = {N:,})", loc="left")
    axes[0].set_ylabel("Macro-F1 on the test year")
    fig.subplots_adjust(wspace=0.55, bottom=0.36)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=2)
    save(fig, "fig3_breakeven_construction", rows, ["dataset", "gap", "series", "labels_per_class", "macro_f1", "n_star"])


def fig4():
    """Label-equivalence ratio by gap, every model."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    rows = []
    for ax, (ds, title), tag in zip(axes, DS, "ab"):
        meta, m = load(ds)
        N = meta["N"]
        G = sorted(int(g) for g in m["M1"]["exchange"])
        for k, (mid, name, col, mk, ls) in enumerate(MODELS):
            if mid not in m:
                continue
            xs, ys, los, his, cens = [], [], [], [], []
            for g in G:
                e = m[mid]["exchange"][str(g)]
                v = num(e["n_star"], N)
                rows.append([ds, mid, g, e["n_star"], e["n_star_ci"][0], e["n_star_ci"][1], e["rho"]])
                if v is None:
                    continue
                lo = num(e["n_star_ci"][0], N); hi = num(e["n_star_ci"][1], N)
                xs.append(g + (k - 2.5) * 0.06); ys.append(v / N)
                los.append(v / N - (lo / N if lo is not None else 0)); his.append((hi / N if hi is not None else 1.0) - v / N)
                cens.append(hi is None)
            xs, ys, los, his, cens = map(np.array, (xs, ys, los, his, cens))
            ax.plot(xs, ys, color=col, marker=mk, ms=3.5, ls=ls, lw=1.0, label=name, mfc="white" if mid in ("M5", "M5F") else col, zorder=3)
            ok = ~cens
            ax.errorbar(xs[ok], ys[ok], yerr=[los[ok], his[ok]], fmt="none", ecolor=col, elinewidth=0.6, capsize=1.5)
            if cens.any():   # upper bound censored at N: bar to 1 ending in an upward arrow, no cap
                ax.errorbar(xs[cens], ys[cens], yerr=[los[cens], his[cens]], fmt="none", ecolor=col, elinewidth=0.6, capsize=0)
                for xc in xs[cens]:
                    ax.annotate("", xy=(xc, 1.07), xytext=(xc, 0.99),
                                arrowprops=dict(arrowstyle="-|>", color=col, lw=0.7, mutation_scale=6, shrinkA=0, shrinkB=0))
        ax.axhline(1.0, color=MUTED, lw=0.5); ax.axhline(0.5, color=MUTED, lw=0.5, ls=(0, (1, 2)))
        ax.set_xticks(G); ax.set_ylim(0, 1.1); ax.set_xlabel("Age of training labels (years)")
        ax.set_title(f"({tag}) {title}", loc="left")
    axes[0].set_ylabel("Label-equivalence ratio  n*(g) / N")
    fig.subplots_adjust(bottom=0.30)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=3, handlelength=2.6, columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig4_equivalence_ratio", rows, ["dataset", "model", "gap_years", "n_star", "ci_low", "ci_high", "rho"])


def fig5():
    """Stale plus fresh labels versus either alone (primary model)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    rows = []
    for ax, (ds, title), tag in zip(axes, DS, "ab"):
        meta, m = load(ds)
        N = meta["N"]
        g = str(max(int(x) for x in m["M1"]["mixing"]))
        mx = m["M1"]["mixing"][g]
        ns = [int(n) for n in mx]
        fresh = [mx[str(n)]["fresh_only"] for n in ns]; mix = [mx[str(n)]["stale_plus_fresh"] for n in ns]
        stale = mx[str(ns[0])]["stale_only"]
        ax.plot(ns, mix, color="#0072B2", marker="o", ms=3.5, label="N old + n current")
        ax.plot(ns, fresh, color="#D55E00", marker="s", ms=3.5, ls="--", label="n current only")
        ax.axhline(stale, color=INK, lw=0.9, ls=":", label="N old only")
        rows += [[ds, g, n, a, b, stale] for n, a, b in zip(ns, mix, fresh)]
        ax.set_xscale("log"); ax.set_xticks(ns, [str(n) for n in ns]); ax.minorticks_off()
        ax.set_xlabel("Current labels per class, n"); ax.set_title(f"({tag}) {title} (N = {N}, {g} years old)", loc="left")
    axes[0].set_ylabel("Macro-F1 on the test year")
    axes[1].legend(frameon=False, loc="lower right")
    save(fig, "fig5_stale_plus_fresh", rows, ["dataset", "gap_years", "fresh_per_class", "stale_plus_fresh", "fresh_only", "stale_only"])


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5):
        f()
    print(sorted(p.name for p in FIG.iterdir()))
