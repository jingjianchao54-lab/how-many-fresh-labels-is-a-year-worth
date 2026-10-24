"""Stack Exchange eligibility pass for one site (amendment A-01). Writes work/eligible/se_<site>.npz and counts.

Rows kept after exclusions and the per-user cap: kind (0 post, 1 comment), id, year, split, ts. No model is fitted.
"""
import hashlib
import html
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

from common import AUTHOR_CAP, DATA, MIN_TOKENS, WORK, split_of, tokens
from sevenzip import iter_lines

TAG_RE = re.compile(r"<[^>]+>")
SE_SITES = ["philosophy", "photo", "music", "history", "cooking", "movies", "bicycles", "boardgames", "gardening", "chess"]


def clean_html(body):
    return html.unescape(TAG_RE.sub(" ", body))


def rows(site):
    """Yield (kind, id, user, text, timestamp) for questions, answers and comments."""
    path = DATA / "stackexchange" / f"{site}.stackexchange.com.7z"
    for name, line in iter_lines(path, {"Posts.xml", "Comments.xml"}):
        s = line.strip()
        if not s.startswith(b"<row "):
            continue
        a = ET.fromstring(s).attrib
        if name == "Posts.xml":
            if a.get("PostTypeId") not in ("1", "2"):
                continue
            yield 0, a.get("Id"), a.get("OwnerUserId"), a.get("Body"), a.get("CreationDate")
        else:
            yield 1, a.get("Id"), a.get("UserId"), a.get("Text"), a.get("CreationDate")


def main(site):
    t0 = time.time()
    c = Counter()
    seen = set()
    per_user = defaultdict(list)
    for kind, uid, user, text, date in rows(site):
        c["read"] += 1
        if not uid or not user or not text or not date:
            c["x_missing"] += 1
            continue
        if user == "-1":
            c["x_community_account"] += 1
            continue
        text = clean_html(text) if kind == 0 else html.unescape(text)
        key = hashlib.blake2b((user + "\x00" + text).encode("utf-8"), digest_size=8).digest()
        if key in seen:
            c["x_duplicate"] += 1
            continue
        seen.add(key)
        if len(tokens(text)) < MIN_TOKENS["reddit"]:
            c["x_short"] += 1
            continue
        dt = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
        per_user[(user, dt.year)].append((int(dt.timestamp()), kind, int(uid)))
        c["eligible_before_cap"] += 1
    out_rows = []
    for (user, year), items in per_user.items():
        sp = split_of(f"{site}:{user}")
        items.sort()
        for ts, kind, uid in items[:AUTHOR_CAP]:
            out_rows.append((kind, uid, year, sp, ts))
        c["x_over_user_cap"] += max(0, len(items) - AUTHOR_CAP)
    out_rows.sort()
    arr = np.array(out_rows, dtype=np.int64)
    out = WORK / "eligible"
    np.savez_compressed(out / f"se_{site}.npz", kind=arr[:, 0].astype(np.int8), id=arr[:, 1], year=arr[:, 2].astype(np.int16),
                        split=arr[:, 3].astype(np.int8), ts=arr[:, 4])
    c["kept_after_cap"] = len(out_rows)
    c["users"] = len({u for u, _ in per_user})
    json.dump({"site": site, "counts": dict(c), "seconds": round(time.time() - t0, 1)},
              open(out / f"se_{site}.counts.json", "w"), indent=1)
    print(site, dict(c), round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
