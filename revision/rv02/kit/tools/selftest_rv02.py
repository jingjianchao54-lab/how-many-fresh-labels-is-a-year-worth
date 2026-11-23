"""RV-02 synthetic self-test. Reads NO real data and writes only inside <project tmp dir>/rv02_selftest_<timestamp>/.

python selftest_rv02.py <repo_exp_dir> <kit_dir> <tmp_dir>
  <repo_exp_dir>  the registered repository root (folder holding src/, results/, work/)
  <kit_dir>       the unpacked RV-02 kit (holding src/ and tools/)
  <tmp_dir>       scratch folder inside the project, e.g. .../Paper5_LabelRecency_Algorithms/tmp
Builds a synthetic arXiv-like corpus, then runs eligibility -> gate -> sample -> 12 M1 select runs -> verify, in an
isolated copy of the code. Prints SELFTEST PASS or raises.
"""
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

repo, kit, tmp = map(Path, sys.argv[1:4])
import time
root = tmp / f"rv02_selftest_{time.strftime('%Y%m%d_%H%M%S')}" / "exp"     # new folder each time; nothing is deleted
(root / "results").mkdir(parents=True)
shutil.copytree(repo / "src", root / "src")
for f in (kit / "src").glob("*.py"):
    shutil.copy(f, root / "src" / f.name)
shutil.copy(repo / "results" / "gate.json", root / "results" / "gate.json")
snap = root.parent / "snap"
snap.mkdir()
random.seed(2026)
A = ["astro-ph.GA", "cond-mat.mtrl-sci", "cs.CL", "cs.CV", "hep-ph", "hep-th", "math.AP", "math.CO", "quant-ph", "physics.optics"]
words = {c: [f"{c.replace('.', '').replace('-', '')}w{i}" for i in range(50)] for c in A}
common_w = [f"the{i}" for i in range(200)]
pid = 0
with open(snap / "records.jsonl", "w") as f:
    for y in range(2018, 2027):
        for c in A:
            for _ in range(1900):
                pid += 1
                txt = " ".join(random.choice(words[c] if random.random() < 0.3 else common_w) for _ in range(40))
                f.write(json.dumps({"id": f"{y % 100:02d}{pid:07d}", "submitter": f"user{random.randint(0, 40000)}",
                                    "title": f"Title {pid}", "abstract": txt, "categories": c,
                                    "versions": [{"version": "v1", "created": f"Mon, 3 Mar {y} 10:00:00 GMT"}],
                                    "comments": ""}) + "\n")
src = root / "src"
run = lambda *a: subprocess.run([sys.executable, *a], cwd=src, check=True)
run("build_eligible_arxiv.py", str(snap))
run("gate_arxiv.py")
run("build_sample_arxiv.py", str(snap))
run("run.py", "arxiv", "select", "M1", "0", "60")
dec = json.load(open(root / "results" / "gate.json"))["arxiv"]["decision"]
assert dec and dec["years"][-1] == 2025 and len(dec["classes"]) == 10, dec
assert len(list((root / "work" / "runs" / "arxiv" / "select" / "M1").glob("*.npz"))) > 0
print("SELFTEST PASS", json.dumps(dec))
