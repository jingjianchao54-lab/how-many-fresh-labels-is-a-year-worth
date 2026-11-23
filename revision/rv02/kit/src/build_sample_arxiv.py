"""RV-02: draw the fixed test/validation sets and training pools for arXiv-10 (plan 3.5, same draw() as
build_sample.py) and extract their texts. Writes work/sample_arxiv.pkl with the fields of the other sample files
(cls, year, part, line, tok, nat, classes, years, size) plus `author` (salted SHA-256 of the submitter key, 16 hex)
and `pid` (arXiv identifier, public metadata; used only for the primary-category check).

python build_sample_arxiv.py <snapshot_dir>
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from build_eligible_arxiv import iter_records, nat_text, submitter_key
from build_sample import draw
from common import RESULTS, URL_RE, WORK, tokens


def main(snapshot: Path) -> None:
    dec = json.load(open(RESULTS / "gate.json"))["arxiv"]["decision"]
    n, v, t = dec["size"]
    want: dict[int, tuple] = {}
    for ci, cat in enumerate(dec["classes"]):
        z = np.load(WORK / "eligible" / f"ax_{cat}.npz")
        for y in dec["years"]:
            idx = {p: z["line"][(z["year"] == y) & (z["split"] == p)] for p in (0, 1, 2)}
            for p, ks in draw(idx, {0: 2 * n, 1: v, 2: t}, [ci, y]).items():
                for k in ks:
                    want[int(k)] = (ci, y, p)
    got: dict[int, tuple] = {}
    for fi, row, d in iter_records(snapshot):
        line = fi * 10_000_000 + row
        if line in want:
            got[line] = (nat_text(d["title"], d["abstract"]), submitter_key(d["submitter"]), str(d["id"]))
    assert len(got) == len(want), (len(got), len(want))
    rec = {k: [] for k in ("cls", "year", "part", "line", "tok", "nat", "author", "pid")}
    for line in sorted(want, key=lambda k: (want[k][0], k)):
        ci, y, p = want[line]
        text, key, pid = got[line]
        rec["pid"].append(pid)
        rec["cls"].append(ci); rec["year"].append(y); rec["part"].append(p); rec["line"].append(line)
        rec["tok"].append(" ".join(tokens(text)))
        rec["nat"].append(" ".join(URL_RE.sub(" ", text).split()))
        rec["author"].append(hashlib.sha256(("paper5-author|arxiv:" + key).encode("utf-8")).hexdigest()[:16])
    out = {k: (np.array(vv, dtype=np.int64 if k == "line" else (np.int8 if k in ("cls", "part") else np.int16))
               if k in ("cls", "year", "part", "line") else vv) for k, vv in rec.items()}
    out.update(classes=dec["classes"], years=dec["years"], size=dec["size"])
    pickle.dump(out, open(WORK / "sample_arxiv.pkl", "wb"), protocol=4)
    exp = len(dec["classes"]) * len(dec["years"]) * (2 * n + v + t)
    print("arxiv", len(out["cls"]), "expected", exp)
    assert len(out["cls"]) == exp


if __name__ == "__main__":
    main(Path(sys.argv[1]))
