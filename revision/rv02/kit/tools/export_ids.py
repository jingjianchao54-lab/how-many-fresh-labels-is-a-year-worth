"""RV-02 check of the primary-category assumption, step 1 (server, from exp/src):
python $KIT/tools/export_ids.py -> ../results/RV02_primary_check_ids.tsv (300 random sampled papers: id <TAB> assigned class)"""
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
from common import RESULTS, WORK
s = pickle.load(open(WORK / "sample_arxiv.pkl", "rb"))
rng = np.random.default_rng(2026)
idx = rng.choice(len(s["pid"]), size=300, replace=False)
with open(RESULTS / "RV02_primary_check_ids.tsv", "w") as f:
    for i in sorted(idx):
        f.write(f"{s['pid'][i]}\t{s['classes'][int(s['cls'][i])]}\n")
print("written 300 ids")
