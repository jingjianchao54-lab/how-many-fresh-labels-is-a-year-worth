"""Named values for the analyses added in Round 1 (Section 4.7 and related sentences).

Read only from revision/out/*.json (written by the revision scripts from stored predictions or registered-split
refits). Same formatting rules as values.py: costs in Macro-F1 points to 2 decimals, break-even sizes to whole
labels, ratios and correlations to 2 decimals, shares as percentages to 1 decimal.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REV = Path(__file__).resolve().parents[1] / "revision" / "out"
DS = {"rd": "reddit", "se": "stackexchange"}
N = {"rd": 2000, "se": 500}
T = {"rd": 1000, "se": 300}


def L(name: str) -> dict:
    return json.load(open(REV / name))


def pts(x: float) -> str:
    return f"{round(x, 2):.2f}".replace("-", "−")


def r2(x: float) -> str:
    return f"{round(x, 2):.2f}".replace("-", "−")


def pct(x: float) -> str:
    return f"{round(100 * x, 1):.1f}%"


def nst(x) -> str:
    if isinstance(x, str):
        if x.startswith(">"):
            return "> " + f"{int(x[1:]):,}"
        if x.startswith("<"):
            return "< " + x[1:]
        x = float(x)
    return f"{int(round(x)):,}"


def ci_n(c) -> str:
    return f"[{nst(c[0])}, {nst(c[1])}]"


def ci_p(c) -> str:
    return f"[{pts(c[0])}, {pts(c[1])}]"


def rho_of(n, tag: str):
    if isinstance(n, str):
        return None
    return n / N[tag]


def build() -> dict[str, str]:
    R: dict[str, str] = {}
    R["rd_T"], R["se_T"] = f"{T['rd']:,}", f"{T['se']:,}"
    # ---------------------------------------------------------------- clustered bootstrap
    holm_max = 0.0
    for tag, ds in DS.items():
        c = L(f"cluster_{ds}.json")
        tc = c["test_cells"]
        R[f"R_{tag}_auth_mean"] = f"{tc['authors_per_cell_mean']:,.0f}"
        R[f"R_{tag}_ipa"] = f"{tc['items_per_author_mean']:.2f}"
        R[f"R_{tag}_single"] = pct(tc["share_single_item_authors"])
        m1 = c["models"]["M1"]
        R[f"R_{tag}_w1"] = f"{m1['ci_width_ratio_cluster_over_item']['1']:.2f}"
        R[f"R_{tag}_w6"] = f"{m1['ci_width_ratio_cluster_over_item']['6']:.2f}"
        cl = m1["cluster"]
        holm_max = max(holm_max, cl["holm"]["H1"], cl["holm"]["H2"])
        R[f"R_{tag}_cl_n6lo"] = nst(cl["exchange"]["6"]["n_star_ci"][0])
        R[f"R_{tag}_cl_holm_H3"] = f"{cl['holm']['H3']:.4f}"
        for m, r in c["models"].items():
            e = r["cluster"]["exchange"]["6"]
            R[f"R_{tag}_{m}_cl_n6ci"] = ci_n(e["n_star_ci"])
    R["R_cl_holm_max"] = f"{holm_max:.4f}"
    # ---------------------------------------------------------------- learning-curve models
    for tag, ds in DS.items():
        c = L(f"curves_{ds}.json")["models"]["M1"]
        R[f"R_{tag}_ipl_rmse"] = f"{c['fit_all_year_curve']['ipl']:.2f}"
        R[f"R_{tag}_ll_rmse"] = f"{c['fit_all_year_curve']['loglin_global']:.2f}"
        h = c["holdout_top_all_year_curve"]
        R[f"R_{tag}_ipl_hold"] = pts(abs(h["ipl"]))
        R[f"R_{tag}_loc_hold"] = pts(abs(h["loglin_local"]))
        R[f"R_{tag}_llg_hold"] = pts(abs(h["loglin_global"]))
        for g in ("1", "6"):
            e = c["gaps"][g]["ipl_fit_all"]
            R[f"R_{tag}_ipl_n{g}"] = nst(e["n_star"])
            R[f"R_{tag}_ipl_n{g}ci"] = ci_n(e["n_star_ci"])
    # ---------------------------------------------------------------- class level
    for tag, ds in DS.items():
        c = L(f"classwise_{ds}.json")["models"]
        m1, m6 = c["M1"]["6"], c["M6"]["6"]
        c6 = [e["cost"] * 100 for e in m1]
        R[f"R_{tag}_cls_c6_min"], R[f"R_{tag}_cls_c6_max"] = pts(min(c6)), pts(max(c6))
        rhos = [e["rho"] for e in m1 if e["rho"] is not None]
        R[f"R_{tag}_cls_rho6_min"] = r2(min(rhos))
        R[f"R_{tag}_cls_n_cens"] = str(sum(1 for e in m1 if isinstance(e["n_star"], str) and e["n_star"].startswith(">")))
        R[f"R_{tag}_cls_spear"] = r2(spearmanr(c6, [e["cost"] for e in m6]).statistic)
        top = max(m1, key=lambda e: e["cost"])
        R[f"R_{tag}_cls_top_M1"] = top["class"]
        R[f"R_{tag}_cls_top_M6"] = max(m6, key=lambda e: e["cost"])["class"]
    # ---------------------------------------------------------------- confusion
    for tag, ds in DS.items():
        c = L(f"confusion_{ds}.json")["models"]
        for m in ("M1", "M6"):
            r = c[m]["6"]
            R[f"R_{tag}_{m}_top3"] = pct(r["top3_share"])
            R[f"R_{tag}_{m}_toppairs"] = ", ".join("–".join(p["pair"]) for p in r["top_pairs"][:3])
            R[f"R_{tag}_{m}_npairs_up"] = str(r["n_pairs_increase"])
    # ---------------------------------------------------------------- descriptives, language
    d = L("descriptives.json")
    pe = d["stackexchange"]["per_year"]
    R["R_se_post_first"], R["R_se_post_last"] = pct(pe["2015"]["post_share"]), pct(pe["2021"]["post_share"])
    for tag, ds in DS.items():
        R[f"R_{tag}_cap256"] = pct(d[ds]["overall"]["share_at_256_cap"])
        R[f"R_{tag}_over128"] = pct(d[ds]["overall"]["share_over_128_words"])
        js = [v["js_class_shares_vs_first_year_bits"] for v in d[ds]["per_year"].values()]
        R[f"R_{tag}_classjs_max"] = f"{max(js):.3f}"
        try:
            lg = L(f"lang_{ds}.json")
            R[f"R_lang_ne_{tag}"] = pct(lg["share_non_english_all"])
            R[f"R_lang_ne_test_{tag}"] = pct(lg["share_non_english_test"])
            R[f"R_lang_ne_max_{tag}"] = pct(max(lg["by_class"].values()))
            R[f"R_lang_ne_maxcls_{tag}"] = max(lg["by_class"], key=lg["by_class"].get)
        except FileNotFoundError:
            pass
    # ---------------------------------------------------------------- refits and subsets
    for tag, ds in DS.items():
        for v in ("pii", "emoji", "sitemask", "fixedC", "widegrid", "mixw", "mixtune", "english_english"):
            p = REV / f"variant_{v}_{ds}.json"
            if not p.exists():
                continue
            x = json.load(open(p))
            k = f"R_{tag}_{v}"
            R[f"{k}_F1"] = f"{x['same_year_f1']:.3f}"
            for g in ("1", "6"):
                R[f"{k}_C{g}"] = pts(x[g]["cost_points"])
                R[f"{k}_rho{g}"] = "> 1" if x[g]["rho"] is None else r2(x[g]["rho"])
                R[f"{k}_n{g}ci"] = ci_n(x[g]["n_star_ci"])
            R[f"{k}_holm"] = "/".join(f"{x['holm'][h]:.3f}" for h in ("H1", "H2", "H3"))
        try:
            st = L(f"stats_pii_{ds}.json")
            R[f"R_{tag}_pii_items"] = pct(st["items_with_any"] / st["items"])
            st = L(f"stats_emoji_{ds}.json")
            R[f"R_{tag}_emoji_items"] = pct(st["items_with_emoji_or_emoticon"] / st["items"])
        except FileNotFoundError:
            pass
    pt = L("posttype_stackexchange.json")["models"]["M1"]
    for k, lab in (("posts", "post"), ("comments", "com")):
        R[f"R_se_{lab}_c6"] = pts(pt[k]["6"]["cost_points"])
        R[f"R_se_{lab}_c6ci"] = ci_p(pt[k]["6"]["cost_ci_points"])
        R[f"R_se_{lab}_rho6"] = "> 1" if pt[k]["6"]["rho"] is None else r2(pt[k]["6"]["rho"])
    diffs, ok_n = [], {}
    for tag, ds in DS.items():
        reg = json.load(open(REV.parents[1] / "results" / f"results_{ds}.json"))["models"]["M1"]["mixing"]["6"]
        for var in ("mixw", "mixtune"):
            p = REV / f"variant_{var}_{ds}.json"
            if not p.exists():
                continue
            mx = json.load(open(p))["6"]["mixing"]
            for n, e in mx.items():
                diffs.append(abs(e["mix_minus_fresh_points"][0] - reg[n]["mix_minus_fresh"][0] * 100))
                good = e["mix_minus_fresh_points"][1] > 0 and e["mix_minus_stale_points"][1] > 0
                ok_n.setdefault(int(n), []).append(good)
    R["R_mix_maxdiff"] = pts(max(diffs)) if diffs else "–"
    ns = sorted(ok_n)
    nmin = next((n for n in ns if all(all(ok_n[m]) for m in ns if m >= n)), None)
    R["R_mix_nmin"] = f"{nmin:,}" if nmin else "–"
    _e = json.load(open(REV.parents[1] / "results" / "results_reddit.json"))["models"]["M1"]["exchange"]["6"]
    reg_rho6 = _e["n_star"] / N["rd"]
    wt = json.load(open(REV.parent / "selections" / "widegrid_selection_table_reddit.json"))["2000"]
    R["R_widegrid_sentence"] = (
        "The wider grid selected the registered value at every size on Stack Exchange-10, and at every size except *N* on "
        f"Reddit-10, where it chose *C* = 1,000 instead of 10 (validation Macro-F1 {float(wt['1000']):.3f} against {float(wt['10']):.3f}); "
        f"refitting with that value gave *ρ*(6) = {R['R_rd_widegrid_rho6']} against the registered {reg_rho6:.2f}.")
    R["R_check_runs"] = str(L("check_runs.json")["n_runs_identical"])
    return R
