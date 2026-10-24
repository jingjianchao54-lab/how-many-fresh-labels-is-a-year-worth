"""Frozen all-MiniLM-L6-v2 sentence vectors for every sampled text (plan Section 4, M5). Resumable in chunks.

python encode_minilm.py check            -> reference-sentence check, results/minilm_reference_check.json
python encode_minilm.py <dataset>        -> work/minilm_<dataset>.npy (written once all chunks exist)
Only accuracy is studied with these vectors; no timing is recorded or reported.
"""
import json
import pickle
import sys

import numpy as np

from common import DATA, RESULTS, WORK
from minilm_np import MiniLM

CHUNK = 4000


def check():
    sent = ["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."]
    ref = np.array([[1.0000, 0.6660, 0.1046], [0.6660, 1.0000, 0.1411], [0.1046, 0.1411, 1.0000]])
    e = MiniLM(DATA / "minilm", max_len=256).encode(sent, batch_size=3)
    s = e @ e.T
    diff = float(np.abs(np.round(s, 4) - ref).max())
    out = {"sentences": sent, "reference": ref.tolist(), "computed": np.round(s, 6).tolist(),
           "max_abs_diff_after_rounding_to_4dp": diff, "pass": diff <= 1e-4,
           "source": "sentence-transformers documentation, quickstart example"}
    json.dump(out, open(RESULTS / "minilm_reference_check.json", "w"), indent=1)
    print(json.dumps(out))
    assert out["pass"]


def encode(dataset):
    """A-02: encode validation, test and replicate-1 training items only, at 128 word pieces."""
    from run import Data
    s = pickle.load(open(WORK / f"sample_{dataset}.pkl", "rb"))
    texts = s["nat"]
    d = Data(dataset)
    need = np.zeros(len(texts), dtype=bool)
    need[d.part > 0] = True
    for t in d.years:
        need[d.train_rows(t, 1, d.N)] = True
    idx = np.flatnonzero(need)
    out_dir = WORK / f"minilm_{dataset}_chunks"
    out_dir.mkdir(exist_ok=True)
    model = MiniLM(DATA / "minilm", max_len=128)
    for a in range(0, len(idx), CHUNK):
        path = out_dir / f"{a:07d}.npy"
        if path.exists():
            continue
        np.save(path, model.encode([texts[i] for i in idx[a:a + CHUNK]], batch_size=32))
        print(a, len(idx), flush=True)
    full = np.full((len(texts), 384), np.nan, dtype=np.float32)
    full[idx] = np.concatenate([np.load(out_dir / f"{a:07d}.npy") for a in range(0, len(idx), CHUNK)])
    np.save(WORK / f"minilm_{dataset}.npy", full)
    print("done", len(idx), flush=True)


if __name__ == "__main__":
    check() if sys.argv[1] == "check" else encode(sys.argv[1])
