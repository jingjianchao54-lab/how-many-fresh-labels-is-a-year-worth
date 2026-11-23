"""R-A5: descriptive drift of corpus composition (Reviewer 2, comments 6 and 8). No model is involved.
Per year: natural class shares among eligible items (data gate counts), JS divergence of class shares to the first year,
item length (registered tokens), share of items reaching the 256-token cap, share above 128 whitespace words,
Stack Exchange post share. python a5_descriptives.py -> revision/out/descriptives.json"""
from __future__ import annotations
import json, pickle
import numpy as np
from common_rev import ROOT

def js(p: np.ndarray, q: np.ndarray) -> float:
    m = (p + q) / 2
    f = lambda a, b: float(np.sum(np.where(a > 0, a * np.log2(a / b), 0.0)))
    return 0.5 * f(p, m) + 0.5 * f(q, m)

out = {}
gate = json.load(open(ROOT / "results" / "gate.json"))
for ds in ("reddit", "stackexchange"):
    dec = gate[ds]["decision"]; tab = gate[ds]["table"]
    Y = dec["years"]
    share = {y: np.array([sum(tab[c][str(y)]) for c in dec["classes"]], float) for y in Y}
    share = {y: v / v.sum() for y, v in share.items()}
    s = pickle.load(open(ROOT / "revision" / "full" / f"sample_{ds}.pkl", "rb"))
    ntok = np.array([len(t.split()) for t in s["tok"]]); nws = np.array([len(r.split()) for r in s["raw"]])
    yr = s["year"]; kind = np.array(s["kind"])
    per = {}
    for y in Y:
        m = yr == y
        per[str(y)] = {"class_share_max": round(float(share[y].max()), 4), "class_share_min": round(float(share[y].min()), 4),
                       "js_class_shares_vs_first_year_bits": round(js(share[y], share[Y[0]]), 4),
                       "tokens_median": float(np.median(ntok[m])), "tokens_mean": round(float(ntok[m].mean()), 1),
                       "share_at_256_cap": round(float((ntok[m] >= 256).mean()), 4),
                       "share_over_128_words": round(float((nws[m] > 128).mean()), 4),
                       "post_share": round(float((kind[m] == 0).mean()), 4) if ds == "stackexchange" else None}
    out[ds] = {"classes": dec["classes"], "class_share_by_year": {str(y): [round(float(x), 4) for x in share[y]] for y in Y}, "per_year": per,
               "overall": {"share_at_256_cap": round(float((ntok >= 256).mean()), 4), "share_over_128_words": round(float((nws > 128).mean()), 4)}}
json.dump(out, open(ROOT / "revision" / "out" / "descriptives.json", "w"), indent=1)
for ds in out: print(ds, out[ds]["overall"], {y: (v["js_class_shares_vs_first_year_bits"], v["tokens_median"], v["post_share"]) for y, v in out[ds]["per_year"].items()})
