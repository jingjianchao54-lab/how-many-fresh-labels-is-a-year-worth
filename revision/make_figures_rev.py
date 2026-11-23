"""Revised figures (Round 1). Drawn only from the locked result files (Figures 1-5) and the revision outputs
(Figures 6-8). Every figure is written as PDF (TrueType fonts), SVG (editable text) and 600-dpi PNG, with its source data.

python make_figures_rev.py -> revision/figures/fig*.{pdf,svg,png} and fig*_data.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, REV = ROOT / "results", ROOT / "revision" / "out"
FIG = ROOT / "revision" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.1,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "legend.frameon": False,
})
W2 = 6.3                                  # two-panel width in inches (16 cm, MDPI full text width)
INK, MUTED, LIGHT = "#222222", "#6E6E6E", "#BDBDBD"
DS = [("reddit", "Reddit-10"), ("stackexchange", "Stack Exchange-10")]
# Okabe-Ito colours with redundant line style and marker per model
STY = {"M1": ("TF-IDF + LR", "#0072B2", "o", "-"), "M2": ("TF-IDF + SVM", "#E69F00", "s", "--"),
       "M3": ("Counts + CNB", "#009E73", "^", "-."), "M4": ("GloVe + LR", "#D55E00", "D", ":"),
       "M5F": ("MiniLM + LR", "#000000", "v", (0, (5, 1.5))), "M6": ("Fine-tuned BERT", "#CC79A7", "P", (0, (3, 1, 1, 1)))}
GROUPS = [("Sparse lexical models", ["M1", "M2", "M3"]), ("Dense and pre-trained models", ["M4", "M5F", "M6"])]
DIVERGE = mpl.colors.LinearSegmentedColormap.from_list(
    "loss", [(0.0, "#B35806"), (0.5, "#FFFFFF"), (0.62, "#D1E5F0"), (0.8, "#4393C3"), (1.0, "#053061")])


def load(ds: str) -> tuple[dict, dict]:
    r = json.load(open(RES / f"results_{ds}.json"))
    models = dict(r["models"])
    a03 = json.load(open(RES / f"results_{ds}_A03.json"))["models"]
    assert a03["M1"] == models["M1"]
    models["M5F"], models["M6"] = a03["M5F"], a03["M6"]
    return r["meta"], models


def num(v):
    return None if isinstance(v, str) or v is None else float(v)


def save(fig, name: str, rows: list, header: list[str]) -> None:
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        fig.savefig(FIG / f"{name}.{ext}", **kw)
    plt.close(fig)
    with open(FIG / f"{name}_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def legend_below(fig, axes, ncol: int = 3, y: float = -0.02) -> None:
    h, l = [], []
    for ax in np.ravel(axes):
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    idx = sorted(range(len(l)), key=lambda i: ("reference" in l[i], i))
    h, l = [h[i] for i in idx], [l[i] for i in idx]
    fig.legend(h, l, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, y), handlelength=2.8, columnspacing=1.4)


# ------------------------------------------------------------------------------------------------ Figure 1
def fig1() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.95), layout="constrained")
    rows = []
    for ax, (ds, title), tag in zip(axes, DS, "ab"):
        meta, m = load(ds)
        Y = meta["years"]
        g = np.array([[m["M1"]["grid_macro_f1"][str(t)][str(u)] for u in Y] for t in Y])
        loss = (np.diag(g)[None, :] - g) * 100
        im = ax.imshow(loss, cmap=DIVERGE, norm=mpl.colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=3.0), aspect="equal")
        for i, t in enumerate(Y):
            for j, u in enumerate(Y):
                txt = "–" if i == j else f"{loss[i, j]:.1f}".replace("-0.0", "0.0").replace("-", "−")
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white" if loss[i, j] > 1.8 else INK)
                rows.append([ds, t, u, round(float(g[i, j]), 5), round(float(loss[i, j]), 3)])
        ax.set_xticks(range(len(Y)), Y, rotation=45)
        ax.set_yticks(range(len(Y)), Y)
        ax.set_xlabel("Test year")
        ax.set_ylabel("Training year")
        ax.set_title(f"({tag}) {title}, N = {meta['N']:,}")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.plot([-0.5, len(Y) - 0.5], [-0.5, len(Y) - 0.5], color=MUTED, lw=0.4, ls=":")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, ticks=[-1, 0, 1, 2, 3])
    cb.set_label("Macro-F1 points lost vs. same-year training")
    cb.outline.set_visible(False)
    save(fig, "fig1_aging_grid", rows, ["dataset", "train_year", "test_year", "macro_f1", "points_lost_vs_same_year"])


# ------------------------------------------------------------------------------------------------ Figures 2 and 4
def _panel(ax, ds, meta, m, mids, quantity, show_ref: bool) -> list:
    rows = []
    N = meta["N"]
    G = list(range(1, 7))
    offs = {mid: (k - (len(mids) - 1) / 2) * 0.11 for k, mid in enumerate(mids)}
    if show_ref:                                            # primary model as grey reference in the lower row
        ys = []
        for g in G:
            if quantity == "cost":
                ys.append(m["M1"]["staleness"]["forward"][str(g)]["cost"] * 100)
            else:
                ys.append(num(m["M1"]["exchange"][str(g)]["n_star"]) / N)
        ax.plot(G, ys, color=LIGHT, lw=2.4, zorder=1, label="M1 (reference)", solid_capstyle="round")
    for mid in mids:
        name, col, mk, ls = STY[mid]
        xs, ys, lo, hi, cens = [], [], [], [], []
        for g in G:
            x = g + offs[mid]
            if quantity == "cost":
                c = m[mid]["staleness"]["forward"][str(g)]
                v, a, b = c["cost"] * 100, c["ci"][0] * 100, c["ci"][1] * 100
                rows.append([ds, mid, g, c["cost"], c["ci"][0], c["ci"][1]])
                xs.append(x); ys.append(v); lo.append(a); hi.append(b); cens.append(False)
            else:
                e = m[mid]["exchange"][str(g)]
                rows.append([ds, mid, g, e["n_star"], e["n_star_ci"][0], e["n_star_ci"][1], e["rho"]])
                v = num(e["n_star"])
                if v is None:
                    continue
                a, b = num(e["n_star_ci"][0]), num(e["n_star_ci"][1])
                xs.append(x); ys.append(v / N); lo.append((a or 0) / N); hi.append(1.0 if b is None else b / N)
                cens.append(b is None)
        xs, ys, lo, hi, cens = map(np.array, (xs, ys, lo, hi, cens))
        fill = "white" if mid == "M5F" else col
        ax.plot(xs, ys, color=col, ls=ls, lw=1.1, zorder=3)
        ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt=mk, color=col, ms=3.8, mfc=fill, mec=col, mew=0.8,
                    elinewidth=0.7, capsize=1.6, zorder=4, label=f"{mid} {name}")
        for xc in xs[cens]:
            ax.annotate("", xy=(xc, 1.08), xytext=(xc, 0.995),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=0.7, mutation_scale=6, shrinkA=0, shrinkB=0))
    ax.set_xticks(G)
    return rows


def fig_by_group(quantity: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(W2, 4.9), sharex=True, sharey=True, layout="constrained")
    rows = []
    for col_i, (ds, title) in enumerate(DS):
        meta, m = load(ds)
        for row_i, (gname, mids) in enumerate(GROUPS):
            ax = axes[row_i, col_i]
            rows += _panel(ax, ds, meta, m, mids, quantity, show_ref=(row_i == 1))
            tag = "abcd"[row_i * 2 + col_i]
            ax.set_title(f"({tag}) {title}: {gname.lower()}")
            if quantity == "cost":
                ax.axhline(0, color=MUTED, lw=0.5)
                ax.axhline(1.0, color=MUTED, lw=0.6, ls=(0, (1, 2)))
                if row_i == 0 and col_i == 0:
                    ax.plot([], [], color=MUTED, lw=0.6, ls=(0, (1, 2)), label="practical threshold, 1 point (reference)")
                ax.set_ylim(-0.25, 3.15)
            else:
                ax.axhline(1.0, color=MUTED, lw=0.5)
                ax.axhline(0.5, color=MUTED, lw=0.6, ls=(0, (1, 2)))
                if row_i == 0 and col_i == 0:
                    ax.plot([], [], color=MUTED, lw=0.6, ls=(0, (1, 2)), label="H3 threshold, M1 only (reference)")
                ax.set_ylim(0, 1.12)
            if row_i == 1:
                ax.set_xlabel("Age of training labels, g (years)")
            if col_i == 0:
                ax.set_ylabel("Staleness cost C(g) (Macro-F1 points)" if quantity == "cost" else "Label-equivalence ratio ρ(g)")
    legend_below(fig, axes, ncol=3, y=-0.005)
    if quantity == "cost":
        save(fig, "fig2_staleness_cost", rows, ["dataset", "model", "gap_years", "cost", "ci_low", "ci_high"])
    else:
        save(fig, "fig4_equivalence_ratio", rows, ["dataset", "model", "gap_years", "n_star", "ci_low", "ci_high", "rho"])


# ------------------------------------------------------------------------------------------------ Figure 3
def fig3() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.75), layout="constrained")
    rows = []
    shade = {1: "#56B4E9", 6: "#0072B2"}
    mk = {1: "o", 6: "s"}
    for ax, (ds, title), tag in zip(axes, DS, "ab"):
        meta, m = load(ds)
        S, N = meta["sizes"], meta["N"]
        fb = json.load(open(RES / f"fresh_by_gap_{ds}.json"))["models"]["M1"]["by_gap"]
        lo_all = 1.0
        for g in (1, 6):
            fc = [fb[str(g)]["fresh_curve"][str(n)] for n in S]
            lo_all = min(lo_all, min(fc))
            e = m["M1"]["exchange"][str(g)]
            ns, st = num(e["n_star"]), e["stale_macro_f1"]
            ax.plot(S, fc, color=shade[g], marker=mk[g], ms=3.2, lw=1.1, ls="-" if g == 1 else "--",
                    label=f"Current labels (test years with g = {g})")
            ax.axhline(st, color=shade[g], lw=0.8, ls=(0, (4, 2)) if g == 1 else (0, (1, 1.5)))
            ax.annotate(f"N old labels, g = {g}", xy=(S[0] * 0.85, st), xytext=(0, 2 if g == 1 else -2),
                        textcoords="offset points", fontsize=8, color=INK, va="bottom" if g == 1 else "top")
            if ns:
                ax.plot([ns, ns], [0, st], color=shade[g], lw=0.8, ls=":")
                ax.plot([ns], [st], marker="D", ms=4.6, color=shade[g], mec="white", mew=0.6, zorder=5)
                ax.annotate(f"n* = {ns:,.0f}", xy=(ns, lo_all), xytext=(-3, 3), textcoords="offset points",
                            fontsize=8, color=shade[g], ha="right", rotation=90, va="bottom")
            rows += [[ds, g, "current", n, v, ""] for n, v in zip(S, fc)]
            rows.append([ds, g, "stale_N", N, st, e["n_star"]])
        ax.set_xscale("log")
        ax.set_xticks(S, [f"{s:,}" for s in S])
        ax.minorticks_off()
        ax.set_ylim(lo_all - 0.02, max(fb["1"]["fresh_curve"][str(N)], fb["6"]["fresh_curve"][str(N)]) + 0.02)
        ax.set_xlim(S[0] * 0.8, S[-1] * 1.2)
        ax.set_xlabel("Current training labels per class, n (log scale)")
        ax.set_title(f"({tag}) {title}, N = {N:,}")
    axes[0].set_ylabel("Macro-F1 on the test year")
    axes[0].plot([], [], marker="D", ls="none", color=MUTED, mec="white", label="Break-even size n*(g)")
    legend_below(fig, axes, ncol=3)
    save(fig, "fig3_breakeven_construction", rows, ["dataset", "gap", "series", "labels_per_class", "macro_f1", "n_star"])


# ------------------------------------------------------------------------------------------------ Figure 5
def fig5() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.7), layout="constrained")
    rows = []
    for ax, (ds, title), tag in zip(axes, DS, "ab"):
        meta, m = load(ds)
        S, N = meta["sizes"], meta["N"]
        mx = m["M1"]["mixing"]["6"]
        ns = [n for n in S[:-1]]
        mix = [mx[str(n)]["stale_plus_fresh"] for n in ns]
        fr = [mx[str(n)]["fresh_only"] for n in ns]
        stale = mx[str(ns[0])]["stale_only"]
        lo = [mx[str(n)]["stale_plus_fresh"] - (mx[str(n)]["mix_minus_stale"][0] - mx[str(n)]["mix_minus_stale"][1]) for n in ns]
        hi = [mx[str(n)]["stale_plus_fresh"] + (mx[str(n)]["mix_minus_stale"][2] - mx[str(n)]["mix_minus_stale"][0]) for n in ns]
        ax.fill_between(ns, lo, hi, color="#0072B2", alpha=0.15, lw=0, label="95% interval of the gain over old labels alone")
        ax.plot(ns, mix, color="#0072B2", marker="o", ms=3.2, lw=1.2, label="N old + n current labels")
        ax.plot(ns, fr, color="#D55E00", marker="s", ms=3.0, lw=1.0, ls="--", label="n current labels only")
        ax.axhline(stale, color=INK, lw=0.9, ls=":", label="N old labels only")
        rows += [[ds, 6, n, a, b, stale] for n, a, b in zip(ns, mix, fr)]
        ax.set_xscale("log")
        ax.set_xticks(ns, [f"{s:,}" for s in ns])
        ax.minorticks_off()
        ax.set_xlabel("Current training labels per class, n (log scale)")
        ax.set_title(f"({tag}) {title}, labels 6 years old, N = {N:,}")
    axes[0].set_ylabel("Macro-F1 on the test year")
    legend_below(fig, axes, ncol=2)
    save(fig, "fig5_stale_plus_fresh", rows, ["dataset", "gap_years", "fresh_per_class", "stale_plus_fresh", "fresh_only", "stale_only"])


# ------------------------------------------------------------------------------------------------ Figure 6 (class level)
def fig6() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(W2, 5.0), layout="constrained")
    rows = []
    for r_i, (ds, title) in enumerate(DS):
        d = json.load(open(REV / f"classwise_{ds}.json"))
        N = d["N"]
        per1 = {e["class"]: e for e in d["models"]["M1"]["6"]}
        order = sorted(per1, key=lambda c: per1[c]["cost"])
        y = np.arange(len(order))
        for mid, dy, fill in (("M1", 0.17, True), ("M6", -0.17, False)):
            name, col, mk, _ = STY[mid]
            per = {e["class"]: e for e in d["models"][mid]["6"]}
            c = np.array([per[k]["cost"] * 100 for k in order])
            cl = np.array([per[k]["cost_ci"][0] * 100 for k in order])
            ch = np.array([per[k]["cost_ci"][1] * 100 for k in order])
            axes[r_i, 0].errorbar(c, y + dy, xerr=[c - cl, ch - c], fmt=mk, color=col, ms=4.3, mfc=col if fill else "white",
                                  elinewidth=0.9, capsize=1.8, label=f"{mid} {name}")
            for j, k in enumerate(order):
                e = per[k]
                v, lo, hi = num(e["n_star"]), num(e["n_star_ci"][0]), num(e["n_star_ci"][1])
                rows.append([ds, mid, k, e["cost"], e["cost_ci"][0], e["cost_ci"][1], e["n_star"], e["n_star_ci"][0], e["n_star_ci"][1]])
                if v is None:                               # censored above N: draw at 1 with an open arrow
                    axes[r_i, 1].plot([1.0], [j + dy], marker=">", color=col, ms=4.3, mfc="white")
                    continue
                lo_r = 0.0 if lo is None else lo / N
                hi_r = 1.0 if hi is None else hi / N
                axes[r_i, 1].errorbar([v / N], [j + dy], xerr=[[v / N - lo_r], [hi_r - v / N]], fmt=mk, color=col, ms=4.3,
                                      mfc=col if fill else "white", elinewidth=0.9, capsize=1.8)
                if hi is None:
                    axes[r_i, 1].annotate("", xy=(1.06, j + dy), xytext=(0.995, j + dy),
                                          arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9, mutation_scale=6))
        tag0, tag1 = "ac"[r_i], "bd"[r_i]
        axes[r_i, 0].axvline(0, color=MUTED, lw=0.5)
        axes[r_i, 0].set_yticks(y, order)
        axes[r_i, 0].set_title(f"({tag0}) {title}: six-year cost by class")
        axes[r_i, 0].set_xlabel("Class-level staleness cost (class F1 points)")
        axes[r_i, 1].axvline(0.5, color=MUTED, lw=0.6, ls=(0, (1, 2)))
        axes[r_i, 1].axvline(1.0, color=MUTED, lw=0.5)
        axes[r_i, 1].set_yticks(y, [])
        axes[r_i, 1].set_xlim(0, 1.1)
        axes[r_i, 1].set_title(f"({tag1}) {title}: six-year ratio by class")
        axes[r_i, 1].set_xlabel("Class-level label-equivalence ratio")
    axes[0, 0].plot([], [], marker=">", ms=5, mfc="white", mec=INK, ls="none", label="Point estimate above N")
    axes[0, 0].plot([], [], marker=r"$\rightarrow$", ms=10, color=INK, ls="none", label="Upper interval bound above N")
    legend_below(fig, axes, ncol=2)
    save(fig, "fig8_classwise", rows, ["dataset", "model", "class", "cost", "cost_ci_low", "cost_ci_high", "n_star", "n_star_ci_low", "n_star_ci_high"])


# ------------------------------------------------------------------------------------------------ Figure 7 (pairs of years)
def fig7() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.9), sharey=True, layout="constrained")
    rows = []
    for ax, (ds, title), tag in zip(axes, DS, "ab"):
        yp = json.load(open(RES / f"year_pairs_{ds}.json"))
        meta, m = load(ds)
        N = meta["N"]
        last = meta["years"][-1]
        for mid, dx in (("M1", -0.16), ("M6", 0.16)):
            name, col, mk, _ = STY[mid]
            for g in range(1, 7):
                for p in yp["models"][mid][str(g)]["pairs"]:
                    rows.append([ds, mid, g, p["train"], p["test"], p["cost"], p["n_star"], p["rho"]])
                    x = g + dx + (p["test"] - meta["years"][0] - 3) * 0.018
                    hl = p["test"] == last
                    if p["rho"] is None:
                        ax.plot([x], [1.04], marker="^", ms=3.4, color=col, mfc="white", mew=0.7)
                    else:
                        ax.plot([x], [p["rho"]], marker=mk, ms=3.4 if not hl else 4.2, color=col,
                                mfc=col if mid == "M1" else "white", mec=INK if hl else col, mew=0.9 if hl else 0.7, ls="none")
            pooled = [num(m[mid]["exchange"][str(g)]["n_star"]) for g in range(1, 7)]
            ax.plot(np.arange(1, 7) + dx, [v / N if v else np.nan for v in pooled], color=col, lw=0.9, ls="-" if mid == "M1" else "--",
                    label=f"{mid} {name}: pooled estimate")
            ax.plot([], [], marker=mk, color=col, ls="none", mfc=col if mid == "M1" else "white", label=f"{mid}: single pair of years")
        ax.plot([], [], marker="o", ls="none", mfc="none", mec=INK, mew=0.9, label="last test year of the window (black edge)")
        ax.plot([], [], marker="^", ls="none", color=MUTED, mfc="white", label="censored above N")
        ax.axhline(0.5, color=MUTED, lw=0.6, ls=(0, (1, 2)))
        ax.axhline(1.0, color=MUTED, lw=0.5)
        ax.set_ylim(0, 1.1)
        ax.set_xticks(range(1, 7))
        ax.set_xlabel("Age of training labels, g (years)")
        ax.set_title(f"({tag}) {title}")
    axes[0].set_ylabel("Label-equivalence ratio ρ(g)")
    legend_below(fig, axes, ncol=2)
    save(fig, "fig6_year_pairs", rows, ["dataset", "model", "gap", "train_year", "test_year", "cost", "n_star", "rho"])


# ------------------------------------------------------------------------------------------------ Figure 8 (estimators)
EST = [("loglinear_interp", "Log-linear interpolation (registered)"), ("linear_interp", "Linear interpolation in n"),
       ("ipl_fit_all", "Inverse power law, all sizes"), ("loglin_fit_all", "Global log-linear fit"),
       ("ipl_pilot_extrapolation", "Inverse power law, sizes ≤ N/4 (extrapolated)")]
EST_STY = {"loglinear_interp": ("#0072B2", "o"), "linear_interp": ("#56B4E9", "s"), "ipl_fit_all": ("#009E73", "^"),
           "loglin_fit_all": ("#E69F00", "D"), "ipl_pilot_extrapolation": ("#CC79A7", "v")}


def fig8() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.7), sharey=True, layout="constrained")
    rows = []
    for ax, (ds, title), tag in zip(axes, DS, "ab"):
        d = json.load(open(REV / f"curves_{ds}.json"))
        N = d["N"]
        for gi, g in enumerate(("1", "6")):
            for k, (key, label) in enumerate(EST):
                e = d["models"]["M1"]["gaps"][g][key]
                v = num(e["n_star"])
                lo, hi = (num(e["n_star_ci"][0]), num(e["n_star_ci"][1])) if "n_star_ci" in e else (None, None)
                rows.append([ds, g, key, e["n_star"], *(e.get("n_star_ci") or ["", ""])])
                y = (1 - gi) * 6 + (4 - k)
                col, mk = EST_STY[key]
                ax.errorbar([v / N], [y], xerr=[[v / N - (lo or v) / N], [((hi or N) / N) - v / N]], fmt=mk, color=col, ms=3.8,
                            elinewidth=0.8, capsize=1.6, label=label if gi == 0 else None)
        ax.axvline(0.5, color=MUTED, lw=0.6, ls=(0, (1, 2)))
        ax.axhspan(5.4, 10.6, color="#F2F2F2", zorder=0, lw=0)
        ax.set_yticks([2, 8], ["g = 6", "g = 1"])
        ax.set_xlim(0.35, 1.08)
        ax.axvline(1.0, color=MUTED, lw=0.5)
        ax.set_xlabel("Label-equivalence ratio ρ(g), primary model M1")
        ax.set_title(f"({tag}) {title}, N = {N:,}")
    legend_below(fig, axes, ncol=2)
    save(fig, "fig7_estimators", rows, ["dataset", "gap", "estimator", "n_star", "ci_low", "ci_high"])


if __name__ == "__main__":
    fig1(); fig_by_group("cost"); fig3(); fig_by_group("ratio"); fig5(); fig6(); fig7(); fig8()
    print("written", sorted(p.name for p in FIG.glob("*.png")))
