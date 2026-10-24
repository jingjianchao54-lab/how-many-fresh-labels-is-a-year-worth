"""Counts-only data gate (plan 3.4). Applies the pre-specified ladder rule and writes results/gate.json."""
import json
from collections import defaultdict

import numpy as np

from common import LADDER, RESULTS, SUBREDDITS, WORK


def eligible_cell(cnt, size):
    n, v, t = size
    return cnt[0] >= 2 * n and cnt[1] >= v and cnt[2] >= t


def runs(years_ok):
    """Maximal runs of consecutive years from a sorted list."""
    out, cur = [], []
    for y in years_ok:
        if cur and y != cur[-1] + 1:
            out.append(cur)
            cur = []
        cur.append(y)
    if cur:
        out.append(cur)
    return out


def reddit_gate(SUBREDDITS=SUBREDDITS, prefix="", max_year=None):
    """D1 rule of plan 3.4; also used unchanged for Stack Exchange (amendment A-01) with prefix "se_"."""
    counts = {}                                   # class -> year -> [train, val, test]
    for s in SUBREDDITS:
        z = np.load(WORK / "eligible" / f"{prefix}{s}.npz")
        d = defaultdict(lambda: [0, 0, 0])
        for y, sp in zip(z["year"], z["split"]):
            if max_year is None or int(y) <= max_year:
                d[int(y)][int(sp)] += 1
        counts[s] = dict(d)
    years = sorted({y for d in counts.values() for y in d})
    table = {s: {y: counts[s].get(y, [0, 0, 0]) for y in years} for s in SUBREDDITS}
    decision = None
    for k_min in (10, 8):
        for size in LADDER["reddit"]:
            best = None
            # for k_min 8: choose class subsets implicitly by eligibility over a run
            ok_by_year = {y: [s for s in SUBREDDITS if eligible_cell(table[s][y], size)] for y in years}
            for run in runs([y for y in years if len(ok_by_year[y]) >= k_min]):
                # shrink to sub-runs where a common class set of size >= k_min exists
                for a in range(len(run)):
                    for b in range(len(run), a, -1):
                        common = set(SUBREDDITS)
                        for y in run[a:b]:
                            common &= set(ok_by_year[y])
                        if len(common) >= k_min and b - a >= 6:
                            cand = (b - a, run[b - 1], sorted(common, key=SUBREDDITS.index), run[a:b])
                            if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                                best = cand
                            break
            if best:
                decision = {"size": size, "k_min": k_min, "years": best[3], "classes": best[2]}
                break
        if decision:
            break
    return {"table": {s: {str(y): v for y, v in table[s].items()} for s in SUBREDDITS}, "decision": decision}


def huffpost_gate():
    z = np.load(WORK / "eligible" / "huffpost.npz")
    names = json.load(open(WORK / "eligible" / "huffpost.counts.json"))["categories"]
    d = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for y, sp, c in zip(z["year"], z["split"], z["cat"]):
        d[names[int(c)]][int(y)][int(sp)] += 1
    years = sorted({y for c in d for y in d[c]})
    table = {c: {y: d[c].get(y, [0, 0, 0]) for y in years} for c in names}
    decision = None
    for size in LADDER["huffpost"]:
        best = None
        ok = {y: {c for c in names if eligible_cell(table[c][y], size)} for y in years}
        for a in range(len(years)):
            for b in range(a + 5, len(years) + 1):
                if years[b - 1] - years[a] != b - a - 1:
                    continue
                common = set(names)
                for y in years[a:b]:
                    common &= ok[y]
                if len(common) >= 5:
                    cand = (b - a, len(common), years[b - 1], years[a:b], common)
                    if best is None or cand[:3] > best[:3]:
                        best = cand
        if best:
            classes = sorted(best[4], key=lambda c: -min(table[c][y][0] for y in best[3]))[:10]
            decision = {"size": size, "years": best[3], "classes": sorted(classes), "n_eligible_classes": len(best[4])}
            break
    return {"table": {c: {str(y): v for y, v in table[c].items()} for c in names}, "decision": decision}


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    import sys
    path = RESULTS / "gate.json"
    if len(sys.argv) > 1 and sys.argv[1] == "stackexchange":       # added to the existing gate file; D1 entry untouched
        from build_eligible_se import SE_SITES
        out = json.load(open(path))
        out = {"stackexchange": reddit_gate(SE_SITES, "se_", max_year=2023)}
        full = json.load(open(path)); full.update(out)
        json.dump(full, open(path, "w"), indent=1)
    else:
        out = {"reddit": reddit_gate(), "huffpost": huffpost_gate()}
        json.dump(out, open(path, "w"), indent=1)
    for k in out:
        print(k, json.dumps(out[k]["decision"]))
