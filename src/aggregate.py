"""Flatten every locked result file into results/summary.csv, the only source of numbers quoted in the manuscript.

python aggregate.py      -> results/summary.csv (columns: source, path, value)
python aggregate.py A03  -> results/summary_with_A03.csv: the same plus the A-03 result files, M5F/M6 derived values
                            and the slope identity values of Section 5 (summary.csv is left unchanged)
Derived quantities used in the text are computed here, never by hand.
"""
import csv
import json
import sys

from common import RESULTS

FILES = ["fresh_by_gap_reddit.json", "fresh_by_gap_stackexchange.json", "results_reddit.json", "results_stackexchange.json", "results_reddit_rep1.json",
         "results_stackexchange_rep1.json", "results_reddit_tok_raw.json", "sens_half_reddit.json",
         "sens_half_stackexchange.json", "drift_reddit.json", "drift_stackexchange.json",
         "selection_reddit.json", "selection_stackexchange.json", "LOCK_reddit_M5.json",
         "LOCK_stackexchange_M5.json", "minilm_reference_check.json"]


def walk(o, path, out):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, f"{path}/{k}", out)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, f"{path}[{i}]", out)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        out.append((path, o))


def derived():
    """Quantities that the text states in a form not stored directly in the result files."""
    rows = []
    for ds in ("reddit", "stackexchange"):
        full = json.load(open(RESULTS / f"results_{ds}.json"))
        m = dict(full["models"])
        m["M5"] = json.load(open(RESULTS / f"results_{ds}_rep1.json"))["models"]["M5"]
        N = full["meta"]["N"]
        for k, v in m.items():
            for g, e in v["exchange"].items():
                if isinstance(e["n_star"], (int, float)):
                    rows.append((f"derived/{ds}/{k}/labels_lost_per_class/{g}", round(N - e["n_star"], 1)))
                    rows.append((f"derived/{ds}/{k}/annual_retention_geometric/{g}", round(e["rho"] ** (1 / int(g)), 4)))
            f = v["fresh_curve"]
            rows.append((f"derived/{ds}/{k}/fresh_gain_top_step_from_" + str(full["meta"]["sizes"][-2]), round(f[str(N)] - f[str(full["meta"]["sizes"][-2])], 5)))
        g = json.load(open(RESULTS / "gate.json"))[ds]
        dec = g["decision"]
        rows.append((f"derived/{ds}/n_years", len(dec["years"])))
    # ranges quoted in the text (min and max over models and datasets)
    vals = {"C1": [], "C6": [], "rho1_sparse": [], "rho6_sparse": []}
    for ds in ("reddit", "stackexchange"):
        full = json.load(open(RESULTS / f"results_{ds}.json"))["models"]
        full["M5"] = json.load(open(RESULTS / f"results_{ds}_rep1.json"))["models"]["M5"]
        for k, v in full.items():
            vals["C1"].append(v["staleness"]["forward"]["1"]["cost"])
            vals["C6"].append(v["staleness"]["forward"]["6"]["cost"])
            if k in ("M1", "M2", "M3"):
                vals["rho1_sparse"].append(v["exchange"]["1"]["rho"])
                vals["rho6_sparse"].append(v["exchange"]["6"]["rho"])
    for k, v in vals.items():
        rows += [(f"derived/range/{k}/min", min(v)), (f"derived/range/{k}/max", max(v))]
    return rows


FILES_A03 = ["results_reddit_A03.json", "results_stackexchange_A03.json", "results_reddit_rep1_A03e3.json",
             "results_stackexchange_rep1_A03e3.json", "fresh_by_gap_reddit_A03.json", "fresh_by_gap_stackexchange_A03.json",
             "LOCK_reddit_M5F.json", "LOCK_reddit_M6.json", "LOCK_stackexchange_M5F.json", "LOCK_stackexchange_M6.json",
             "year_pairs_reddit.json", "year_pairs_stackexchange.json",   # X-11
             "rep_spread_reddit.json", "rep_spread_stackexchange.json", "combined_check.json"]   # X-12


def derived_a03():
    """M5F/M6 analogues of derived(), the ranges over all seven models, and the slope identity of Section 5."""
    import math
    rows = []
    C1, C6 = [], []
    for ds in ("reddit", "stackexchange"):
        full = json.load(open(RESULTS / f"results_{ds}.json"))
        meta, m = full["meta"], dict(full["models"])
        m["M5"] = json.load(open(RESULTS / f"results_{ds}_rep1.json"))["models"]["M5"]
        a03 = json.load(open(RESULTS / f"results_{ds}_A03.json"))["models"]
        m["M5F"], m["M6"] = a03["M5F"], a03["M6"]
        N, S = meta["N"], meta["sizes"]
        fb = json.load(open(RESULTS / f"fresh_by_gap_{ds}.json"))["models"]
        fb.update(json.load(open(RESULTS / f"fresh_by_gap_{ds}_A03.json"))["models"])
        for k, v in m.items():
            C1.append(v["staleness"]["forward"]["1"]["cost"]); C6.append(v["staleness"]["forward"]["6"]["cost"])
            for g in ("1", "6"):
                fc = fb[k]["by_gap"][g]["fresh_curve"]
                gain = fc[str(S[-1])] - fc[str(S[-2])]
                slope = gain / math.log(S[-1] / S[-2])
                rows.append((f"derived/{ds}/{k}/gap{g}/top_gain", round(gain, 5)))
                rows.append((f"derived/{ds}/{k}/gap{g}/slope_per_ln_n", round(slope, 5)))
                rows.append((f"derived/{ds}/{k}/gap{g}/exp_minus_C_over_s", round(math.exp(-v["staleness"]["forward"][g]["cost"] / slope), 4)))
            if k in ("M5F", "M6"):
                for g, e in v["exchange"].items():
                    if isinstance(e["n_star"], (int, float)):
                        rows.append((f"derived/{ds}/{k}/labels_lost_per_class/{g}", round(N - e["n_star"], 1)))
    rows += [("derived/range_all7/C1/max", max(C1)), ("derived/range_all7/C6/min", min(C6)), ("derived/range_all7/C6/max", max(C6))]
    return rows


def main():
    a03 = len(sys.argv) > 1 and sys.argv[1] == "A03"
    out = []
    for f in FILES + (FILES_A03 if a03 else []):
        rows = []
        walk(json.load(open(RESULTS / f)), "", rows)
        out += [(f, p, v) for p, v in rows]
    out += [("derived", p, v) for p, v in derived()]
    if a03:
        out += [("derived", p, v) for p, v in derived_a03()]
    with open(RESULTS / ("summary_with_A03.csv" if a03 else "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "path", "value"])
        w.writerows(out)
    print(len(out), "rows")


if __name__ == "__main__":
    main()
