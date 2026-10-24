"""GPU models added by amendment A-03. Requires torch and transformers (see requirements_gpu.txt).

  python gpu_models.py check                         MiniLM reference-sentence check with the transformers implementation
  python gpu_models.py encode <dataset>              M5F vectors: work/minilm_<dataset>_full.npy (every sampled item, 256 word pieces)
  python gpu_models.py bert <dataset> <stage> [wi nw] [--smoke]
        M6, fine-tuned bert-base-uncased. stage "select" (replicate 1, validation sets) or "test" (needs LOCK_<dataset>_M6.json).
        Writes the same run files as run.py: work/runs/<dataset>/<stage>/M6/<kind>_<r>_<t>_<n>_<extra>.npz
        --smoke: two tiny jobs on VALIDATION data into work/smoke/ (never reads test rows); for checking the installation only.

Fixed recipe for M6 (A-03), identical for every job:
  input   natural text (field "nat" of the sample file), truncated to 128 word pieces;
  model   bert-base-uncased + linear classification head, all weights trained;
  optimiser AdamW, weight decay 0.01, batch 32, linear warm-up over the first 10% of steps then linear decay;
  epochs  max(3, ceil(150 / steps per epoch)), at most 30 (so that small training sets still get >= 150 updates);
  precision bf16 autocast if the GPU supports it, otherwise fp16 with a gradient scaler;
  seed    derived from (replicate, training year, size, extra) as below; no early stopping; last epoch is used.
  grid    learning rate in GRID_M6, chosen on same-year validation data exactly as for the other models.
"""
import hashlib
import json
import math
import os
import pickle
import sys
import time

import numpy as np

from common import RESULTS, ROOT, WORK

GRID_M6 = [2e-5, 5e-5]
MAXLEN_M6 = 128
BATCH = 32


def model_dir(name):
    """Local model folders: <repo>/../models/<name> (override with env MODELS_DIR)."""
    from pathlib import Path
    base = Path(os.environ.get("MODELS_DIR", ROOT.parent / "models"))
    return base / name


def seed_of(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def device():
    import torch
    assert torch.cuda.is_available(), "no CUDA device"
    return torch.device("cuda")


# ---------------------------------------------------------------- MiniLM, full design (M5F)
def minilm(texts, max_len=256, batch=256):
    import torch
    from transformers import AutoModel, AutoTokenizer
    path = model_dir("all-MiniLM-L6-v2")
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModel.from_pretrained(path).to(device()).eval()
    out = np.zeros((len(texts), 384), dtype=np.float32)
    with torch.no_grad():
        for a in range(0, len(texts), batch):
            enc = tok(texts[a:a + batch], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device())
            h = mdl(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out[a:a + len(v)] = torch.nn.functional.normalize(v, dim=1).float().cpu().numpy()
    return out


def check():
    sent = ["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."]
    ref = np.array([[1.0000, 0.6660, 0.1046], [0.6660, 1.0000, 0.1411], [0.1046, 0.1411, 1.0000]])
    e = minilm(sent)
    diff = float(np.abs(np.round(e @ e.T, 4) - ref).max())
    out = {"sentences": sent, "max_abs_diff_after_rounding_to_4dp": diff, "pass": diff <= 1e-4, "implementation": "transformers"}
    json.dump(out, open(RESULTS / "minilm_reference_check_transformers.json", "w"), indent=1)
    print(out)
    assert out["pass"]


def encode(dataset):
    s = pickle.load(open(WORK / f"sample_{dataset}.pkl", "rb"))
    path = WORK / f"minilm_{dataset}_full.npy"
    if path.exists():
        sys.exit(f"{path.name} exists")
    t0 = time.time()
    np.save(path, minilm(list(s["nat"]), max_len=256))
    print(dataset, len(s["nat"]), "texts encoded in", round(time.time() - t0), "s")


# ---------------------------------------------------------------- fine-tuned BERT (M6)
class Bert:
    def __init__(self, dataset):
        from transformers import AutoTokenizer
        from run import Data
        self.d = Data(dataset)
        self.texts = pickle.load(open(WORK / f"sample_{dataset}.pkl", "rb"))["nat"]
        self.tok = AutoTokenizer.from_pretrained(model_dir("bert-base-uncased"))
        self._cache = {}

    def enc(self, rows):
        key = (int(rows[0]), int(rows[-1]), len(rows))
        if key not in self._cache:
            e = self.tok([self.texts[i] for i in rows], truncation=True, max_length=MAXLEN_M6, padding=False)
            self._cache[key] = e["input_ids"]
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[key]

    def batches(self, ids, labels, order):
        import torch
        pad = self.tok.pad_token_id
        for a in range(0, len(order), BATCH):
            idx = order[a:a + BATCH]
            L = max(len(ids[i]) for i in idx)
            x = torch.full((len(idx), L), pad, dtype=torch.long)
            for j, i in enumerate(idx):
                x[j, :len(ids[i])] = torch.tensor(ids[i])
            yield x, (x != pad).long(), (None if labels is None else torch.tensor(labels[idx], dtype=torch.long))

    def fit_predict(self, tr, lr, seed, eval_sets):
        import torch
        from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
        dev = device()
        torch.manual_seed(seed); np.random.seed(seed % 2**32)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        mdl = AutoModelForSequenceClassification.from_pretrained(model_dir("bert-base-uncased"), num_labels=self.d.K).to(dev)
        ids, y = self.enc(tr), self.d.cls[tr].astype(np.int64)
        spe = math.ceil(len(tr) / BATCH)
        epochs = min(30, max(3, math.ceil(150 / spe)))
        steps = spe * epochs
        opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
        sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        bf16 = torch.cuda.is_bf16_supported()
        scaler = None if bf16 else torch.cuda.amp.GradScaler()
        rng = np.random.default_rng(seed)
        mdl.train()
        for _ in range(epochs):
            for x, m, lab in self.batches(ids, y, rng.permutation(len(tr))):
                with torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float16):
                    loss = mdl(input_ids=x.to(dev), attention_mask=m.to(dev), labels=lab.to(dev)).loss
                opt.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else:
                    loss.backward(); opt.step()
                sch.step()
        mdl.eval()
        preds = []
        with torch.no_grad():
            for rows in eval_sets:
                eids = self.enc(rows)
                order = np.argsort([len(t) for t in eids])          # length-sorted for speed; restored below
                p = np.zeros(len(rows), dtype=np.int8)
                pos = 0
                for x, m, _ in self.batches(eids, None, order):
                    with torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float16):
                        lo = mdl(input_ids=x.to(dev), attention_mask=m.to(dev)).logits
                    p[order[pos:pos + len(x)]] = lo.argmax(-1).cpu().numpy().astype(np.int8)
                    pos += len(x)
                preds.append(p)
        del mdl; torch.cuda.empty_cache()
        return preds, bool(np.isfinite(float(loss))), epochs


def bert(dataset, stage, wi=0, nw=1, smoke=False):
    from sklearn.metrics import f1_score
    from run import test_jobs
    b = Bert(dataset)
    d = b.d
    if smoke:
        out = WORK / "smoke" / dataset
        jobs = [("sel", 1, d.years[0], d.sizes[0], GRID_M6[0]), ("sel", 1, d.years[-1], d.sizes[1], GRID_M6[1])]
    elif stage == "select":
        out = WORK / "runs" / dataset / "select" / "M6"
        jobs = [("sel", 1, t, n, h) for t in d.years for n in d.sizes for h in GRID_M6]
    else:
        out = WORK / "runs" / dataset / "test" / "M6"
        sel = json.load(open(RESULTS / f"LOCK_{dataset}_M6.json"))["selected"]["M6"]
        jobs = [j for j in test_jobs(d) if not (j[0] == "e3" and j[1] != 1)]     # A-03: E3 on replicate 1 only
    out.mkdir(parents=True, exist_ok=True)
    for j, job in enumerate(jobs):
        if j % nw != wi:
            continue
        kind, r, t, n, extra = job
        path = out / ("_".join(str(v) for v in job) + ".npz")
        if path.exists():
            continue
        t0 = time.time()
        if kind == "sel":
            rows = d.rows(1, t)
            (pred,), ok, ep = b.fit_predict(d.train_rows(t, r, n), extra, seed_of(dataset, "sel", r, t, n, extra), [rows])
            np.savez(path, f1=f1_score(d.cls[rows], pred, average="macro"), acc=np.mean(d.cls[rows] == pred), ok=ok,
                     epochs=ep, sec=time.time() - t0)
        else:
            if kind == "e1":
                tr, h, evals = d.train_rows(t, r, n), sel[str(n)], [d.rows(2, u) for u in d.years]
            elif kind == "e2":
                tr, h, evals = d.train_rows(t, r, n), sel[str(n)], [d.rows(2, t)]
            else:
                tr = np.sort(np.concatenate([d.train_rows(t - extra, r, d.N), d.train_rows(t, r, n)]))
                h, evals = sel[str(d.N)], [d.rows(2, t)]
            preds, ok, ep = b.fit_predict(tr, float(h), seed_of(dataset, kind, r, t, n, extra), evals)
            np.savez(path, pred=np.stack(preds), ok=ok, epochs=ep, sec=time.time() - t0)
        print("M6", job, round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "check":
        check()
    elif what == "encode":
        encode(sys.argv[2])
    elif what == "bert":
        args = [a for a in sys.argv[2:] if not a.startswith("--")]
        ds, stage = args[0], args[1]
        wi, nw = (int(args[2]), int(args[3])) if len(args) > 3 else (0, 1)
        bert(ds, stage, wi, nw, smoke="--smoke" in sys.argv)
