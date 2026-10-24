"""Draw the fixed test/validation sets and training pools (plan 3.5) and extract their texts.

Output work/sample_<dataset>.pkl: dict with
  cls (int8), year (int16), part (int8: 0 training pool, 1 validation, 2 test), line (int32),
  tok (processed tokens joined by space; D1 name-masked), tok_raw (D1 only: without name masking),
  nat (natural text for the sentence encoder: URLs removed, D1 names removed), classes, years, size.
"""
import json
import pickle
import re
import sys
import zipfile

import numpy as np

from common import DATA, NAME_TOKENS, RESULTS, SUBREDDITS, URL_RE, WORK, tokens

NAME_RE = re.compile(r"\b(?:r/)?(?:" + "|".join(sorted(NAME_TOKENS, key=len, reverse=True)) + r")\b", re.IGNORECASE)


def draw(idx_by_part, need, key):
    out = {}
    for part, n in need.items():
        pool = idx_by_part[part]
        rng = np.random.default_rng([0, part] + key)
        out[part] = np.sort(rng.choice(pool, size=n, replace=False))
    return out


def reddit(dec):
    n, v, t = dec["size"]
    years = dec["years"]
    rec = {k: [] for k in ("cls", "year", "part", "line", "tok", "tok_raw", "nat")}
    for ci, sub in enumerate(dec["classes"]):
        z = np.load(WORK / "eligible" / f"{sub}.npz")
        want = {}
        for y in years:
            idx = {p: z["line"][(z["year"] == y) & (z["split"] == p)] for p in (0, 1, 2)}
            d = draw(idx, {0: 2 * n, 1: v, 2: t}, [ci, y])
            for p, lines in d.items():
                for ln in lines:
                    want[int(ln)] = (y, p)
        zf = zipfile.ZipFile(DATA / "reddit" / f"{sub}.corpus.zip")
        name = [x for x in zf.namelist() if x.endswith("utterances.jsonl")][0]
        with zf.open(name) as f:
            for line_no, raw in enumerate(f):
                if line_no in want:
                    y, p = want[line_no]
                    text = json.loads(raw)["text"]
                    rec["cls"].append(ci); rec["year"].append(y); rec["part"].append(p); rec["line"].append(line_no)
                    rec["tok"].append(" ".join(tokens(text, mask_names=True)))
                    rec["tok_raw"].append(" ".join(tokens(text, mask_names=False)))
                    rec["nat"].append(" ".join(NAME_RE.sub(" ", URL_RE.sub(" ", text)).split()))
        print(sub, len(want), flush=True)
    return rec


def stackexchange(dec):
    from build_eligible_se import clean_html, rows
    import html
    n, v, t = dec["size"]
    rec = {k: [] for k in ("cls", "year", "part", "line", "tok", "nat")}
    for ci, site in enumerate(dec["classes"]):
        z = np.load(WORK / "eligible" / f"se_{site}.npz")
        key = z["id"] * 2 + z["kind"]                       # unique item key within a site
        want = {}
        for y in dec["years"]:
            idx = {p: key[(z["year"] == y) & (z["split"] == p)] for p in (0, 1, 2)}
            for p, ks in draw(idx, {0: 2 * n, 1: v, 2: t}, [ci, y]).items():
                for k in ks:
                    want[int(k)] = (y, p)
        got = {}
        for kind, uid, user, text, date in rows(site):
            if not uid:
                continue
            k = int(uid) * 2 + kind
            if k in want and k not in got and text:
                got[k] = clean_html(text) if kind == 0 else html.unescape(text)
        assert len(got) == len(want), (site, len(got), len(want))
        for k in sorted(want):
            y, p = want[k]
            rec["cls"].append(ci); rec["year"].append(y); rec["part"].append(p); rec["line"].append(k)
            rec["tok"].append(" ".join(tokens(got[k])))
            rec["nat"].append(" ".join(URL_RE.sub(" ", got[k]).split()))
        print(site, len(want), flush=True)
    return rec


def main(dataset):
    gate = json.load(open(RESULTS / "gate.json"))[dataset]["decision"]
    rec = reddit(gate) if dataset == "reddit" else stackexchange(gate)
    out = {k: (np.array(v, dtype=np.int64 if k == "line" else None) if k in ("cls", "year", "part", "line") else v) for k, v in rec.items()}
    out.update(classes=gate["classes"], years=gate["years"], size=gate["size"])
    pickle.dump(out, open(WORK / f"sample_{dataset}.pkl", "wb"), protocol=4)
    print(dataset, len(out["cls"]))


if __name__ == "__main__":
    main(sys.argv[1])
