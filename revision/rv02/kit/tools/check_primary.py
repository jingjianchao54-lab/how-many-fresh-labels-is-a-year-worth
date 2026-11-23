"""RV-02 check of the primary-category assumption, step 2 (LOCAL Windows machine with internet; standard library only):
python check_primary.py RV02_primary_check_ids.tsv -> prints agreement and writes RV02_primary_check_result.tsv
Queries the public arXiv API (export.arxiv.org) in batches of 50 with a 3-second pause, as its terms of use ask."""
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
out, agree = [], 0
for i in range(0, len(rows), 50):
    batch = rows[i:i + 50]
    url = "https://export.arxiv.org/api/query?max_results=50&id_list=" + ",".join(r[0] for r in batch)
    xml = urllib.request.urlopen(url, timeout=60).read()
    prim = {}
    for e in ET.fromstring(xml).findall("a:entry", NS):
        eid = re.sub(r"v\d+$", "", e.find("a:id", NS).text.rsplit("/abs/", 1)[-1])
        pc = e.find("x:primary_category", NS)
        prim[eid] = pc.get("term") if pc is not None else ""
    for pid, cls in batch:
        p = prim.get(re.sub(r"v\d+$", "", pid), "NOT_FOUND")
        agree += p == cls
        out.append(f"{pid}\t{cls}\t{p}\t{int(p == cls)}")
    time.sleep(3)
open("RV02_primary_check_result.tsv", "w", encoding="utf-8").write("\n".join(out) + "\n")
print(f"agreement {agree}/{len(rows)} = {agree / len(rows):.3f}  (acceptance: >= 0.95)")
