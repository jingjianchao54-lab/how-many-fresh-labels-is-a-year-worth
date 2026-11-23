"""Write work/sample_arxiv_slim.pkl (no text: cls, year, part, line, author, pid, classes, years, size) for return.
Run from exp/src:  python $KIT/tools/slim_sample.py"""
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from common import WORK
s = pickle.load(open(WORK / "sample_arxiv.pkl", "rb"))
keep = {k: s[k] for k in ("cls", "year", "part", "line", "author", "pid", "classes", "years", "size")}
pickle.dump(keep, open(WORK / "sample_arxiv_slim.pkl", "wb"), protocol=4)
print("slim sample", len(keep["cls"]))
