"""RV-02 (third domain, arXiv-10): counts-only eligibility pass over the arXiv metadata snapshot.

python build_eligible_arxiv.py <snapshot_dir>
  <snapshot_dir> holds the Parquet files of the pinned Hugging Face revision (data/train-*.parquet) or, for the
  offline self-test only, a file records.jsonl with the same fields.

Rules (AMENDMENT_RV-02.md, Section 3; frozen before this script was run):
  unit      one paper; label = primary category (first entry of `categories`); year = calendar year (UTC) of the
            v1 `created` date; author key = normalised `submitter`.
  exclude   missing id/submitter/title/abstract/categories/v1 date; withdrawn papers (abstract or comments say
            "withdrawn"); papers cross-listed to another candidate category; exact duplicates of
            (submitter, normalised title) (first id kept); fewer than 10 tokens; years after MAX_YEAR.
  cap/split at most 5 papers per (submitter, category, year), the earliest by v1 date then id; author split by the
            registered hash (common.split_of) on key "arxiv:" + submitter.
Writes work/eligible/ax_<category>.npz (line, year, split, ts) and ax_<category>.counts.json, plus
work/eligible/ax_index.json mapping `line` to (file, row). No text is written; no model is fitted.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np

from common import AUTHOR_CAP, WORK, split_of, tokens

CANDIDATES_A = ["astro-ph.GA", "cond-mat.mtrl-sci", "cs.CL", "cs.CV", "hep-ph", "hep-th", "math.AP", "math.CO",
                "quant-ph", "physics.optics"]
CANDIDATES_B = ["astro-ph.SR", "cond-mat.str-el", "cond-mat.stat-mech", "gr-qc", "math.PR", "math.NA", "math.OC",
                "math.NT", "cs.LG", "stat.ME"]
CANDIDATES = CANDIDATES_A + CANDIDATES_B
MAX_YEAR = 2025
MIN_TOKENS_AX = 10
WITHDRAWN_RE = re.compile(r"\bwithdrawn\b", re.IGNORECASE)
FIELDS = ["id", "submitter", "title", "abstract", "categories", "versions", "comments"]


def norm_space(s: str) -> str:
    return " ".join(str(s).split())


def submitter_key(s: str) -> str:
    return norm_space(s).casefold()


def nat_text(title: str, abstract: str) -> str:
    return norm_space(title) + ". " + norm_space(abstract)


def v1_datetime(versions):
    if not versions:
        return None
    v1 = [v for v in versions if (v.get("version") if isinstance(v, dict) else None) == "v1"]
    v = v1[0] if v1 else versions[0]
    try:
        return parsedate_to_datetime(v["created"])
    except Exception:
        return None


def iter_records(snapshot: Path):
    """Yield (file_index, row_index, record dict) in a fixed order (files sorted by name)."""
    files = sorted(snapshot.glob("**/*.parquet"))
    if files:
        import pyarrow.parquet as pq
        for fi, f in enumerate(files):
            pf = pq.ParquetFile(f)
            row = 0
            cols = [c for c in FIELDS if c in pf.schema_arrow.names]
            for batch in pf.iter_batches(batch_size=20000, columns=cols):
                for rec in batch.to_pylist():
                    yield fi, row, rec
                    row += 1
        return
    jl = snapshot / "records.jsonl"
    if not jl.exists():
        raise SystemExit(f"no parquet files and no records.jsonl in {snapshot}")
    with open(jl, encoding="utf-8") as fh:
        for row, line in enumerate(fh):
            yield 0, row, json.loads(line)


def file_list(snapshot: Path) -> list[str]:
    files = sorted(snapshot.glob("**/*.parquet"))
    return [str(f.relative_to(snapshot)) for f in files] if files else ["records.jsonl"]


def main(snapshot: Path) -> None:
    t0 = time.time()
    c = Counter()
    cand = set(CANDIDATES)
    seen_title: set[bytes] = set()
    per_key: dict[tuple, list] = defaultdict(list)          # (category, submitter, year) -> [(ts, id, line)]
    index: dict[int, list[int]] = {}
    for fi, row, d in iter_records(snapshot):
        c["read"] += 1
        pid, sub, title, abstract, cats = (d.get("id"), d.get("submitter"), d.get("title"), d.get("abstract"),
                                           d.get("categories"))
        dt = v1_datetime(d.get("versions"))
        if not pid or not sub or not title or not abstract or not cats or dt is None:
            c["x_missing"] += 1
            continue
        clist = str(cats).split()
        primary = clist[0]
        if primary not in cand:
            c["x_not_candidate"] += 1
            continue
        if WITHDRAWN_RE.search(str(abstract)[:300]) or WITHDRAWN_RE.search(str(d.get("comments") or "")):
            c["x_withdrawn"] += 1
            continue
        if any(x in cand for x in clist[1:] if x != primary):
            c["x_crosslisted_candidate"] += 1
            continue
        key = submitter_key(sub)
        h = hashlib.blake2b((key + "\x00" + norm_space(title).casefold()).encode("utf-8"), digest_size=8).digest()
        if h in seen_title:
            c["x_duplicate"] += 1
            continue
        seen_title.add(h)
        year = dt.year if dt.tzinfo is None else dt.astimezone(__import__("datetime").timezone.utc).year
        if year > MAX_YEAR:
            c["x_after_max_year"] += 1
            continue
        if len(tokens(nat_text(title, abstract))) < MIN_TOKENS_AX:
            c["x_short"] += 1
            continue
        line = fi * 10_000_000 + row
        index[line] = [fi, row]
        per_key[(primary, key, year)].append((int(dt.timestamp()), str(pid), line))
        c["eligible_before_cap"] += 1
    rows_by_cat: dict[str, list] = defaultdict(list)
    split_cache: dict[str, int] = {}
    for (cat, key, year), items in per_key.items():
        sp = split_cache.get(key)
        if sp is None:
            sp = split_cache[key] = split_of("arxiv:" + key)
        items.sort()
        for ts, pid, line in items[:AUTHOR_CAP]:
            rows_by_cat[cat].append((line, year, sp, ts))
        c["x_over_author_cap"] += max(0, len(items) - AUTHOR_CAP)
    out = WORK / "eligible"
    out.mkdir(parents=True, exist_ok=True)
    for cat in CANDIDATES:
        rows = sorted(rows_by_cat.get(cat, []))
        arr = np.array(rows, dtype=np.int64).reshape(-1, 4)
        np.savez_compressed(out / f"ax_{cat}.npz", line=arr[:, 0], year=arr[:, 1].astype(np.int16),
                            split=arr[:, 2].astype(np.int8), ts=arr[:, 3])
        cell = Counter((int(y), int(s)) for _, y, s, _ in rows)
        json.dump({"category": cat, "kept_after_cap": len(rows),
                   "cells": {f"{y}|{s}": n for (y, s), n in sorted(cell.items())}},
                  open(out / f"ax_{cat}.counts.json", "w"), indent=1)
    keep = {int(r[0]) for v in rows_by_cat.values() for r in v}
    json.dump({"files": file_list(snapshot), "index": {str(k): index[k] for k in sorted(keep)}},
              open(out / "ax_index.json", "w"))
    c["kept_after_cap"] = len(keep)
    c["submitters"] = len(split_cache)
    json.dump({"counts": dict(c), "seconds": round(time.time() - t0, 1), "max_year": MAX_YEAR,
               "candidates_A": CANDIDATES_A, "candidates_B": CANDIDATES_B},
              open(out / "ax_counts_total.json", "w"), indent=1)
    print(dict(c), round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
