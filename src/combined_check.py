"""H1/H2 lower bounds with replicate variation added (X-12; descriptive normal approximation, after the results).

se_total^2 = se_boot^2 + sd_rep^2 / 5, with se_boot = (upper - lower) / (2 * 1.96) from the reported 95% bootstrap
interval and sd_rep the standard deviation across the five replicates. For the H2 difference sd_rep is bounded
above by sd_rep(C(6)) + sd_rep(C(1)). H3 is checked replicate by replicate from rep_spread_<dataset>.json.

python combined_check.py -> results/combined_check.json
"""
import json

from common import RESULTS

out = {}
for ds in ("reddit", "stackexchange"):
    m = json.load(open(RESULTS / f"results_{ds}.json"))["models"]["M1"]
    st = m["staleness"]
    N = json.load(open(RESULTS / f"results_{ds}.json"))["meta"]["N"]
    g = str(max(int(k) for k in st["forward"] if k.isdigit()))
    def lower(est, ci, sd):
        se_b = (ci[1] - ci[0]) / (2 * 1.96)
        se = (se_b ** 2 + sd ** 2 / 5) ** 0.5
        return round(est - 1.96 * se, 5)
    c1 = st["forward"]["1"]
    d = st["forward_gmax_minus_g1"]
    sd_d = st["forward"][g]["replicate_sd"] + c1["replicate_sd"]
    rs = json.load(open(RESULTS / f"rep_spread_{ds}.json"))["models"]["M1"][g]["n_star_by_replicate"]
    out[ds] = {"H1_lower_combined": lower(c1["cost"], c1["ci"], c1["replicate_sd"]),
               "H2_lower_combined": lower(d["diff"], d["ci"], sd_d),
               "H3_min_replicate_n_star": min(rs), "half_N": N / 2}
json.dump(out, open(RESULTS / "combined_check.json", "w"), indent=1)
print(out)
