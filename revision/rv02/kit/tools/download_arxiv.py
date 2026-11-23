"""RV-02 step B1: fetch the arXiv metadata snapshot at one pinned commit and record hashes.

Run on the server inside the project .venv with:
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python download_arxiv.py <data_dir>/arxiv <results_dir>
Writes <data_dir>/arxiv/data/*.parquet and <results_dir>/RV02_arxiv_snapshot.json (repo, commit, file sizes, SHA-256).
Refuses to run twice into a folder that already holds Parquet files (re-use the recorded commit instead).
"""
import hashlib
import json
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = "librarian-bots/arxiv-metadata-snapshot"
out, res = Path(sys.argv[1]), Path(sys.argv[2])
if list(out.glob("**/*.parquet")):
    raise SystemExit(f"Parquet files already in {out}; not downloading again")
try:
    commit = HfApi().dataset_info(REPO).sha
except Exception as e:  # noqa: BLE001  mirror without the info API: download "main" and read the commit it served
    print("dataset_info failed:", repr(e)[:200], "-> falling back to revision main", flush=True)
    commit = None
print("pinned commit", commit, flush=True)
snapshot_download(repo_id=REPO, repo_type="dataset", revision=commit or "main", local_dir=str(out),
                  allow_patterns=["data/*.parquet", "README.md"])
if commit is None:                                   # huggingface_hub writes <file>.metadata with the commit hash first
    metas = sorted((out / ".cache" / "huggingface" / "download").glob("**/*.metadata"))
    commits = {m.read_text().splitlines()[0].strip() for m in metas}
    assert len(commits) == 1, f"could not determine a single commit: {commits}"
    commit = commits.pop()
    print("commit served for main:", commit, flush=True)
files = sorted(out.glob("**/*.parquet"))
assert files, "no parquet files downloaded"
man = {"repo": REPO, "commit": commit, "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "files": {}}
for f in files:
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    man["files"][str(f.relative_to(out))] = {"bytes": f.stat().st_size, "sha256": h.hexdigest()}
json.dump(man, open(res / "RV02_arxiv_snapshot.json", "w"), indent=1)
print(json.dumps({k: man[k] for k in ("repo", "commit", "fetched_at")}), len(files), "files")
