"""R-B4: language identification of every sampled item with langid.py (Lui and Baldwin 2012), restricted to English and
the languages it ranks next. Writes revision/out/lang_<dataset>.json (shares by year/class) and mask_english_<dataset>.npy.
The registered natural text (URLs removed, names masked on Reddit) is classified; items with fewer than 3 letters-words
are treated as English (too short to classify)."""
from __future__ import annotations
import json, pickle, sys
from collections import Counter
from pathlib import Path
import numpy as np
import os
sys.path.insert(0, os.environ.get("LANGID_DIR", "langid-1.1.6"))
import langid  # noqa: E402
from common_rev import ROOT
OUT = ROOT / "revision" / "out"
for ds in sys.argv[1:] or ["stackexchange", "reddit"]:
    s = pickle.load(open(ROOT / "revision" / "full" / f"sample_{ds}.pkl", "rb"))
    part_all = s["part"]
    sel = np.ones(len(part_all), bool) if ds == "stackexchange" else (part_all == 2)   # Reddit: test items only (time)
    lang = np.array(["en"] * len(part_all), dtype=object)
    for i in np.flatnonzero(sel):
        t, nat = s["tok"][i], s["nat"][i]
        lang[i] = "en" if len(t.split()) < 3 else langid.classify(nat)[0]
    eng = lang == "en"
    np.save(OUT / f"mask_english_{ds}.npy", eng)
    yr, cls, part = s["year"], s["cls"], s["part"]
    res = {"scope": "all sampled items" if ds == "stackexchange" else "test items only",
           "share_non_english_all": round(float(1 - eng[sel].mean()), 4),
           "share_non_english_test": round(float(1 - eng[part == 2].mean()), 4),
           "by_year": {str(y): round(float(1 - eng[sel & (yr == y)].mean()), 4) for y in s["years"]},
           "by_class": {str(c): round(float(1 - eng[sel & (cls == i)].mean()), 4) for i, c in enumerate(s["classes"])},
           "top_other_languages": Counter(lang[~eng]).most_common(8)}
    json.dump(res, open(OUT / f"lang_{ds}.json", "w"), indent=1)
    print(ds, json.dumps(res), flush=True)
