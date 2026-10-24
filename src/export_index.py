"""Export the fixed samples as identifier lists (no text) for the public repository.

python export_index.py -> results/sample_index_<dataset>.csv.gz
Reddit: subreddit, ConvoKit utterance id, line number in utterances.jsonl, year, part.
Stack Exchange: site, item type (post/comment), post or comment Id, year, part.
part: 0 training pool, 1 validation, 2 test. Replicate draws follow from run.Data.train_rows.
"""
import csv
import gzip
import json
import pickle
import zipfile

from common import DATA, RESULTS, WORK


def reddit():
    s = pickle.load(open(WORK / "sample_reddit.pkl", "rb"))
    want = {}
    for i, (c, ln) in enumerate(zip(s["cls"], s["line"])):
        want.setdefault(int(c), {})[int(ln)] = i
    ids = {}
    for ci, sub in enumerate(s["classes"]):
        zf = zipfile.ZipFile(DATA / "reddit" / f"{sub}.corpus.zip")
        name = [x for x in zf.namelist() if x.endswith("utterances.jsonl")][0]
        need = want[ci]
        with zf.open(name) as f:
            for line_no, raw in enumerate(f):
                if line_no in need:
                    ids[need[line_no]] = json.loads(raw)["id"]
    with gzip.open(RESULTS / "sample_index_reddit.csv.gz", "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subreddit", "utterance_id", "line_in_utterances_jsonl", "year", "part"])
        for i in range(len(s["cls"])):
            w.writerow([s["classes"][s["cls"][i]], ids[i], int(s["line"][i]), int(s["year"][i]), int(s["part"][i])])
    print("reddit", len(s["cls"]))


def stackexchange():
    s = pickle.load(open(WORK / "sample_stackexchange.pkl", "rb"))
    with gzip.open(RESULTS / "sample_index_stackexchange.csv.gz", "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["site", "item_type", "item_id", "year", "part"])
        for i in range(len(s["cls"])):
            k = int(s["line"][i])
            w.writerow([s["classes"][s["cls"][i]], "post" if k % 2 == 0 else "comment", k // 2, int(s["year"][i]), int(s["part"][i])])
    print("stackexchange", len(s["cls"]))


if __name__ == "__main__":
    stackexchange()
    reddit()
