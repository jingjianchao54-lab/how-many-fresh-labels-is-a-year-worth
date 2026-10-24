"""Named, formatted values for the manuscript, read only from the locked result files.

The manuscript template refers to values as «name»; build.py substitutes them. Formatting rules:
Macro-F1 to 3 decimals; costs in Macro-F1 points (x100) to 2 decimals; break-even sizes to whole labels;
ratios to 2 decimals; correlations to 2 decimals.
"""
import json
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "exp" / "results"
DS = {"rd": "reddit", "se": "stackexchange"}
MODELS = ["M1", "M2", "M3", "M4", "M5", "M5F", "M6"]   # M5F and M6: amendment A-03 (added analysis)


def values_models():
    return MODELS


def L(name):
    return json.load(open(RES / name))


def f3(x):
    return f"{round(x, 3):.3f}"


def pts(x):
    return f"{round(x * 100, 2):.2f}".replace("-", "\u2212")


def r2(x):
    return f"{round(x, 2):.2f}"


def nstar(x, N=None):
    if isinstance(x, str):
        return x[0] + " " + f"{int(x[1:]):,}"
    return f"{int(round(x)):,}"


def ci_pts(c):
    return f"[{pts(c[0])}, {pts(c[1])}]"


def ci_n(c):
    return f"[{nstar(c[0])}, {nstar(c[1])}]"


def models(ds):
    full = L(f"results_{ds}.json")
    m = dict(full["models"])
    m["M5"] = L(f"results_{ds}_rep1.json")["models"]["M5"]
    a03 = L(f"results_{ds}_A03.json")
    assert a03["models"]["M1"] == full["models"]["M1"], "A-03 file must reproduce M1"
    m["M5F"] = a03["models"]["M5F"]
    m["M6"] = dict(a03["models"]["M6"])
    m["M6"]["mixing"] = L(f"results_{ds}_rep1_A03e3.json")["models"]["M6"]["mixing"]   # M6 E3: replicate 1 only (A-03)
    return full["meta"], m


def build():
    V = {}
    for tag, ds in DS.items():
        meta, m = models(ds)
        V[f"{tag}_N"] = f"{meta['N']:,}"
        V[f"{tag}_halfN"] = f"{meta['N'] // 2:,}"
        for k, v in m.items():
            p = f"{tag}_{k}"
            V[f"{p}_F1"] = f3(v["same_year_macro_f1"])
            V[f"{p}_TD"] = r2(v["td_score_per_year_f1_points"])
            for g, c in v["staleness"]["forward"].items():
                V[f"{p}_C{g}"] = pts(c["cost"])
                V[f"{p}_C{g}ci"] = ci_pts(c["ci"])
            for g, c in v["staleness"]["backward"].items():
                V[f"{p}_B{g}"] = pts(c["cost"])
            d = v["staleness"]["forward_gmax_minus_g1"]
            V[f"{p}_D61"] = pts(d["diff"])
            V[f"{p}_D61ci"] = ci_pts(d["ci"])
            for g, e in v["exchange"].items():
                V[f"{p}_n{g}"] = nstar(e["n_star"])
                V[f"{p}_n{g}ci"] = ci_n(e["n_star_ci"])
                V[f"{p}_rho{g}"] = r2(e["rho"]) if e["rho"] is not None else "n/a"
                V[f"{p}_stale{g}"] = f3(e["stale_macro_f1"])
            for n, f in v["fresh_curve"].items():
                V[f"{p}_fresh{n}"] = f3(f)
            for g, mx in v["mixing"].items():
                for n, x in mx.items():
                    V[f"{p}_mixS_g{g}_n{n}"] = pts(x["mix_minus_stale"][0])
                    V[f"{p}_mixSci_g{g}_n{n}"] = ci_pts(x["mix_minus_stale"][1:])
                    V[f"{p}_mixF_g{g}_n{n}"] = pts(x["mix_minus_fresh"][0])
                    V[f"{p}_mixFci_g{g}_n{n}"] = ci_pts(x["mix_minus_fresh"][1:])
            if v.get("grid_replicate_sd_max") is not None:
                V[f"{p}_sdmax"] = pts(v["grid_replicate_sd_max"])
            S = meta["sizes"]
            V[f"{p}_topgain"] = pts(v["fresh_curve"][str(S[-1])] - v["fresh_curve"][str(S[-2])])
            V[f"{tag}_size_below_top"] = f"{S[-2]:,}"
            if "hypotheses" in v and k == "M1":          # H1-H3 are defined for the primary model only
                for h, pv in v["hypotheses"]["holm_p"].items():
                    V[f"{p}_holm_{h}"] = pv
        d = L(f"drift_{ds}.json")
        V[f"{tag}_drift_rho"] = r2(d["spearman_jsd_vs_f1_loss"]["rho"])
        V[f"{tag}_drift_gap_rho"] = r2(d["spearman_gap_vs_jsd"]["rho"])
        V[f"{tag}_drift_cells"] = str(d["n_cells"])
        s = L(f"sens_half_{ds}.json")
        V[f"{tag}_half_size"] = f"{s['stale_size']:,}"
        V[f"{tag}_half_F1"] = f3(s["same_year_macro_f1"])
        for g, x in s["by_gap"].items():
            V[f"{tag}_half_C{g}"] = pts(x["cost"])
            V[f"{tag}_half_C{g}ci"] = ci_pts(x["cost_ci"])
            V[f"{tag}_half_n{g}"] = nstar(x["n_star"])
            V[f"{tag}_half_n{g}ci"] = ci_n(x["n_star_ci"])
            V[f"{tag}_half_rho{g}"] = r2(x["rho"]) if x["rho"] is not None else "n/a"
    import math
    for tag, ds in DS.items():
        meta, m = models(ds)
        r1 = L(f"results_{ds}_rep1.json")["models"]
        for k in ("M1", "M2", "M3", "M4"):
            for g in ("1", "6"):
                e = r1[k]["exchange"][g]
                V[f"{tag}_{k}r1_rho{g}"] = r2(e["rho"]) if e["rho"] is not None else "n/a"
                V[f"{tag}_{k}r1_n{g}"] = nstar(e["n_star"])
            for g in ("1", "6"):
                V[f"{tag}_{k}_C{g}sd"] = pts(m[k]["staleness"]["forward"][g]["replicate_sd"])
        fb = L(f"fresh_by_gap_{ds}.json")["models"]
        fb.update(L(f"fresh_by_gap_{ds}_A03.json")["models"])
        S = meta["sizes"]
        for k in values_models():
            for g in ("1", "6"):
                fc = fb[k]["by_gap"][g]["fresh_curve"]
                gain = fc[str(S[-1])] - fc[str(S[-2])]
                slope = gain / math.log(S[-1] / S[-2])
                V[f"{tag}_{k}_g{g}_topgain"] = pts(gain)
                V[f"{tag}_{k}_g{g}_slope"] = pts(slope)
                V[f"{tag}_{k}_g{g}_rhoapprox"] = r2(math.exp(-m[k]["staleness"]["forward"][g]["cost"] / slope))
    for tag, ds in DS.items():                       # X-11: ratio for each single pair of years (descriptive)
        yp = L(f"year_pairs_{ds}.json")["models"]["M1"]
        r = [p["rho"] for g in ("1", "2", "3", "4", "5") for p in yp[g]["pairs"] if p["rho"] is not None]
        V[f"{tag}_yp_M1_min"] = r2(min(r)); V[f"{tag}_yp_M1_max"] = r2(max(r))
        V[f"{tag}_yp_M1_cens"] = str(sum(yp[g]["n_censored"] for g in ("1", "2", "3", "4", "5")))
        V[f"{tag}_yp_M1_spread"] = r2(max(yp[g]["rho_max"] - yp[g]["rho_min"] for g in ("1", "2", "3", "4", "5")))
    cc = L("combined_check.json")                    # X-12: replicate spread and combined-variance check
    for tag, ds in DS.items():
        rs = L(f"rep_spread_{ds}.json")["models"]["M1"]
        V[f"{tag}_rs_M1_rho1sd"] = r2(rs["1"]["rho_sd"]); V[f"{tag}_rs_M1_rho6sd"] = r2(rs["6"]["rho_sd"])
        V[f"{tag}_rs_M1_n6min"] = nstar(rs["6"]["n_star_min"]); V[f"{tag}_rs_M1_n6max"] = nstar(rs["6"]["n_star_max"])
        V[f"{tag}_cc_H1"] = pts(cc[ds]["H1_lower_combined"]); V[f"{tag}_cc_H2"] = pts(cc[ds]["H2_lower_combined"])
        V[f"{tag}_halfN"] = f"{int(cc[ds]['half_N']):,}"
    t = L("results_reddit_tok_raw.json")["models"]["M1"]
    V["raw_F1"] = f3(t["same_year_macro_f1"])
    V["raw_C1"] = pts(t["staleness"]["forward"]["1"]["cost"])
    V["raw_C6"] = pts(t["staleness"]["forward"]["6"]["cost"])
    V["raw_n1"] = nstar(t["exchange"]["1"]["n_star"])
    V["raw_n6"] = nstar(t["exchange"]["6"]["n_star"])
    V["raw_n6ci"] = ci_n(t["exchange"]["6"]["n_star_ci"])
    V["raw_n1ci"] = ci_n(t["exchange"]["1"]["n_star_ci"])
    V["raw_C1ci"] = ci_pts(t["staleness"]["forward"]["1"]["ci"]); V["raw_C6ci"] = ci_pts(t["staleness"]["forward"]["6"]["ci"])
    V["raw_rho1"] = r2(t["exchange"]["1"]["rho"])
    V["raw_rho6"] = r2(t["exchange"]["6"]["rho"])
    # ranges over models and datasets
    c1, c6, r1, r6 = [], [], [], []
    for ds in DS.values():
        _, m = models(ds)
        for k, v in m.items():
            c1.append(v["staleness"]["forward"]["1"]["cost"]); c6.append(v["staleness"]["forward"]["6"]["cost"])
            if k in ("M1", "M2", "M3"):              # sparse lexical models
                r1.append(v["exchange"]["1"]["rho"]); r6.append(v["exchange"]["6"]["rho"])
    V["rng_C1_max"] = pts(max(c1)); V["rng_C6_min"] = pts(min(c6)); V["rng_C6_max"] = pts(max(c6))
    V["rng_rho1_min"] = r2(min(r1)); V["rng_rho1_max"] = r2(max(r1))
    V["rng_rho6_min"] = r2(min(r6)); V["rng_rho6_max"] = r2(max(r6))
    return V


if __name__ == "__main__":
    V = build()
    for k in sorted(V)[:40]:
        print(k, V[k])
    print(len(V))
