"""Rebuild the full sample file (texts, author keys) from the public sample index and the raw corpora.

python extract_from_index.py <data_dir> <out_dir> [reddit|stackexchange ...]

Text processing is identical to src/build_sample.py; the row order is that of the sample index, which equals the
original sample order. Adds three fields used only by the revision analyses:
  author   SHA-256 of the author key (Reddit user name; Stack Exchange "site:UserId"), never the key itself
  raw      original text after the registered URL removal only (kept in memory of the analysis machine; not released)
  kind     Stack Exchange item type (0 post, 1 comment); Reddit: 0 for every item
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import html
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np

import os
ROOT = Path(os.environ.get("P5_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
from common import URL_RE, tokens  # noqa: E402
from build_sample import NAME_RE  # noqa: E402


def akey(s: str) -> str:
    return hashlib.sha256(("paper5-author|" + s).encode("utf-8")).hexdigest()[:16]


def load_index(dataset: str) -> list[list[str]]:
    return list(csv.reader(gzip.open(ROOT / "results" / f"sample_index_{dataset}.csv.gz", "rt")))[1:]


def reddit(data: Path) -> dict:
    dec = json.load(open(ROOT / "results" / "gate.json"))["reddit"]["decision"]
    rows = load_index("reddit")
    want: dict[str, dict[int, int]] = {}
    for i, r in enumerate(rows):
        want.setdefault(r[0], {})[int(r[2])] = i
    n = len(rows)
    rec = {k: [None] * n for k in ("tok", "tok_raw", "nat", "raw", "author")}
    for sub in dec["classes"]:
        zf = zipfile.ZipFile(data / "reddit" / f"{sub}.corpus.zip")
        name = [x for x in zf.namelist() if x.endswith("utterances.jsonl")][0]
        need = want[sub]
        with zf.open(name) as f:
            for line_no, raw in enumerate(f):
                i = need.get(line_no)
                if i is None:
                    continue
                d = json.loads(raw)
                assert d["id"] == rows[i][1], (sub, line_no)
                text = d["text"]
                rec["tok"][i] = " ".join(tokens(text, mask_names=True))
                rec["tok_raw"][i] = " ".join(tokens(text, mask_names=False))
                rec["nat"][i] = " ".join(NAME_RE.sub(" ", URL_RE.sub(" ", text)).split())
                rec["raw"][i] = text
                rec["author"][i] = akey(d["user"])
        print(sub, len(need), flush=True)
    assert all(v is not None for v in rec["tok"])
    rec["kind"] = [0] * n
    return rec, rows, dec


def stackexchange(data: Path) -> dict:
    import common
    common.DATA = data                                  # build_eligible_se reads DATA / "stackexchange"
    import build_eligible_se as bse
    bse.DATA = data
    dec = json.load(open(ROOT / "results" / "gate.json"))["stackexchange"]["decision"]
    rows = load_index("stackexchange")
    want: dict[str, dict[tuple[int, int], int]] = {}
    for i, r in enumerate(rows):
        want.setdefault(r[0], {})[(0 if r[1] == "post" else 1, int(r[2]))] = i
    n = len(rows)
    rec = {k: [None] * n for k in ("tok", "nat", "raw", "author", "kind")}
    for site in dec["classes"]:
        need = want[site]
        for kind, uid, user, text, date in bse.rows(site):
            if not uid:
                continue
            i = need.get((kind, int(uid)))
            if i is None or rec["tok"][i] is not None or not text:
                continue
            t = bse.clean_html(text) if kind == 0 else html.unescape(text)
            rec["tok"][i] = " ".join(tokens(t))
            rec["nat"][i] = " ".join(URL_RE.sub(" ", t).split())
            rec["raw"][i] = t
            rec["author"][i] = akey(f"{site}:{user}")
            rec["kind"][i] = kind
        print(site, len(need), flush=True)
    assert all(v is not None for v in rec["tok"])
    return rec, rows, dec


def main() -> None:
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for ds in sys.argv[3:] or ["stackexchange", "reddit"]:
        rec, rows, dec = (reddit if ds == "reddit" else stackexchange)(data)
        cls_of = {c: i for i, c in enumerate(dec["classes"])}
        s = {k: v for k, v in rec.items()}
        s["cls"] = np.array([cls_of[r[0]] for r in rows], np.int8)
        s["year"] = np.array([int(r[-2]) for r in rows], np.int16)
        s["part"] = np.array([int(r[-1]) for r in rows], np.int8)
        s["line"] = np.array([int(r[2]) if ds == "reddit" else int(r[2]) * 2 + (0 if r[1] == "post" else 1) for r in rows],
                             np.int64)
        s.update(classes=dec["classes"], years=dec["years"], size=dec["size"])
        import gzip as _gz
        pickle.dump(s, _gz.open(out / f"sample_{ds}_full.pkl.gz", "wb", compresslevel=6), protocol=4)
        print(ds, "written", len(rows), flush=True)


if __name__ == "__main__":
    main()
