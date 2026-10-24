"""HuffPost eligibility pass (plan 3.2, 3.3). Writes work/eligible/huffpost.npz and counts. No model is fitted."""
import json
from collections import Counter

import numpy as np

from common import DATA, MIN_TOKENS, WORK, split_of, tokens


def main():
    c = Counter()
    seen = set()
    rows, cats = [], {}
    with open(DATA / "huffpost" / "data.json", encoding="utf-8") as f:
        for line_no, raw in enumerate(f):
            c["read"] += 1
            d = json.loads(raw)
            head, date, cat = (d.get("headline") or "").strip(), d.get("date"), d.get("category")
            if not head or not date or not cat:
                c["x_missing"] += 1
                continue
            text = head + " " + (d.get("short_description") or "").strip()
            if text in seen:
                c["x_duplicate"] += 1
                continue
            seen.add(text)
            if len(tokens(text)) < MIN_TOKENS["huffpost"]:
                c["x_short"] += 1
                continue
            key = d.get("link") or text
            rows.append((line_no, int(str(date)[:4]), split_of(key), cats.setdefault(cat, len(cats))))
    arr = np.array(rows, dtype=np.int64)
    out = WORK / "eligible"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "huffpost.npz", line=arr[:, 0].astype(np.int32), year=arr[:, 1].astype(np.int16),
                        split=arr[:, 2].astype(np.int8), cat=arr[:, 3].astype(np.int16))
    c["kept"] = len(rows)
    names = [k for k, _ in sorted(cats.items(), key=lambda kv: kv[1])]
    json.dump({"counts": dict(c), "categories": names}, open(out / "huffpost.counts.json", "w"), indent=1)
    print(dict(c), len(names))


if __name__ == "__main__":
    main()
