"""Pass over one Reddit corpus: apply exclusions (plan 3.1, 3.3), the per-author cap and the author split.

Writes work/eligible/<sub>.npz with one row per eligible utterance kept after the cap:
  line (int32 line number in utterances.jsonl), year (int16), split (int8), ts (int64)
and work/eligible/<sub>.counts.json with exclusion counts. No text is written; no model is fitted.
"""
import hashlib
import json
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

from common import AUTHOR_CAP, DATA, MIN_TOKENS, WORK, is_excluded_author, split_of, tokens


def main(sub):
    t0 = time.time()
    zf = zipfile.ZipFile(DATA / "reddit" / f"{sub}.corpus.zip")
    name = [n for n in zf.namelist() if n.endswith("utterances.jsonl")][0]
    c = Counter()
    seen = set()
    per_author = defaultdict(list)          # (author, year) -> [(ts, id, line)]
    split_cache = {}
    with zf.open(name) as f:
        for line_no, raw in enumerate(f):
            c["read"] += 1
            try:
                d = json.loads(raw)
            except Exception:
                c["x_malformed"] += 1
                continue
            uid, author, text, ts = d.get("id"), d.get("user"), d.get("text"), d.get("timestamp")
            if not uid or not author or text is None or not isinstance(ts, (int, float)) or ts <= 0:
                c["x_missing"] += 1
                continue
            st = text.strip()
            if st == "" or st in ("[deleted]", "[removed]"):
                c["x_empty_deleted_text"] += 1
                continue
            if is_excluded_author(author):
                c["x_author_deleted_or_bot"] += 1
                continue
            key = hashlib.blake2b((author + "\x00" + text).encode("utf-8"), digest_size=8).digest()
            if key in seen:
                c["x_duplicate"] += 1
                continue
            seen.add(key)
            if len(tokens(text, mask_names=True)) < MIN_TOKENS["reddit"]:
                c["x_short"] += 1
                continue
            year = datetime.fromtimestamp(int(ts), tz=timezone.utc).year
            per_author[(author, year)].append((int(ts), uid, line_no))
            c["eligible_before_cap"] += 1
    rows = []
    for (author, year), items in per_author.items():
        sp = split_cache.get(author)
        if sp is None:
            sp = split_cache[author] = split_of(author)
        items.sort()
        for ts, uid, line_no in items[:AUTHOR_CAP]:
            rows.append((line_no, year, sp, ts))
        c["x_over_author_cap"] += max(0, len(items) - AUTHOR_CAP)
    rows.sort()
    arr = np.array(rows, dtype=np.int64)
    out = WORK / "eligible"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{sub}.npz", line=arr[:, 0].astype(np.int32), year=arr[:, 1].astype(np.int16),
                        split=arr[:, 2].astype(np.int8), ts=arr[:, 3])
    c["kept_after_cap"] = len(rows)
    c["authors"] = len(split_cache)
    cell = Counter((int(y), int(s)) for _, y, s, _ in rows)
    json.dump({"subreddit": sub, "counts": dict(c), "seconds": round(time.time() - t0, 1),
               "cells": {f"{y}|{s}": n for (y, s), n in sorted(cell.items())}},
              open(out / f"{sub}.counts.json", "w"), indent=1)
    print(sub, dict(c), round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
