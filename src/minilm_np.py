"""Frozen all-MiniLM-L6-v2 sentence encoder in numpy (protocol B9, deviation D-07).

BERT-style encoder read straight from model.safetensors, WordPiece tokeniser from vocab.txt,
mean pooling over real tokens and L2 normalisation (the model's published pooling configuration).
No training happens here; the weights are never changed.
"""
import json
import struct
import unicodedata
from pathlib import Path

import numpy as np
from scipy.special import erf

DTYPES = {"F32": np.float32, "I64": np.int64, "F16": np.float16, "F64": np.float64}


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in header.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            f.seek(base + a)
            out[k] = np.frombuffer(f.read(b - a), dtype=DTYPES[v["dtype"]]).reshape(v["shape"])
    return out


# ------------------------------------------------------------------ tokeniser (BERT uncased)
def _is_punct(ch):
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp):
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F
            or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


class WordPiece:
    def __init__(self, vocab_path, max_len=128):
        words = Path(vocab_path).read_text(encoding="utf-8").split("\n")
        if words and words[-1] == "":
            words = words[:-1]
        self.vocab = {w: i for i, w in enumerate(words)}
        self.unk, self.cls, self.sep, self.pad = (self.vocab[t] for t in ("[UNK]", "[CLS]", "[SEP]", "[PAD]"))
        self.max_len = max_len
        self._cache = {}

    def basic(self, text):
        out = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or (unicodedata.category(ch) in ("Cc", "Cf") and ch not in "\t\n\r"):
                continue
            if ch in "\t\n\r" or unicodedata.category(ch) == "Zs":
                out.append(" ")
            elif _is_cjk(cp):
                out.append(f" {ch} ")
            else:
                out.append(ch)
        text = "".join(out).lower()
        text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
        toks = []
        for word in text.split():
            cur = []
            for ch in word:
                if _is_punct(ch):
                    if cur:
                        toks.append("".join(cur))
                        cur = []
                    toks.append(ch)
                else:
                    cur.append(ch)
            if cur:
                toks.append("".join(cur))
        return toks

    def wordpiece(self, word):
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        if len(word) > 100:
            ids = [self.unk]
        else:
            ids, start = [], 0
            while start < len(word):
                end = len(word)
                cur = None
                while start < end:
                    sub = word[start:end] if start == 0 else "##" + word[start:end]
                    if sub in self.vocab:
                        cur = self.vocab[sub]
                        break
                    end -= 1
                if cur is None:
                    ids = [self.unk]
                    break
                ids.append(cur)
                start = end
        self._cache[word] = ids
        return ids

    def encode(self, text):
        ids = [self.cls]
        budget = self.max_len - 2
        for w in self.basic(text):
            ids.extend(self.wordpiece(w))
            if len(ids) - 1 >= budget:
                break
        ids = ids[:budget + 1]
        ids.append(self.sep)
        return ids


# ------------------------------------------------------------------ encoder
def _layer_norm(x, w, b, eps=1e-12):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0, dtype=np.float32)))


class MiniLM:
    def __init__(self, model_dir, max_len=128):
        model_dir = Path(model_dir)
        self.cfg = json.loads((model_dir / "config.json").read_text())
        self.w = load_safetensors(model_dir / "model.safetensors")
        self.tok = WordPiece(model_dir / "vocab.txt", max_len=max_len)
        self.heads = self.cfg["num_attention_heads"]
        self.layers = self.cfg["num_hidden_layers"]
        self.hidden = self.cfg["hidden_size"]
        # transposed weight matrices, contiguous, so every linear layer is a single matrix product
        self._wt = {k[:-len(".weight")]: np.ascontiguousarray(v.T) for k, v in self.w.items()
                    if k.startswith("encoder.layer.") and k.endswith("dense.weight") or ".self." in k and k.endswith(".weight")}

    def forward(self, ids, mask):
        """ids, mask: (B, n) -> token states (B, n, H)."""
        w = self.w
        B, n = ids.shape
        H, h = self.hidden, self.heads
        d = H // h
        x = (w["embeddings.word_embeddings.weight"][ids] + w["embeddings.position_embeddings.weight"][:n]
             + w["embeddings.token_type_embeddings.weight"][0])
        x = _layer_norm(x, w["embeddings.LayerNorm.weight"], w["embeddings.LayerNorm.bias"]).reshape(B * n, H)
        bias = ((1.0 - mask) * -1e4).astype(np.float32)[:, None, None, :]
        scale = np.float32(1.0 / np.sqrt(d))
        for i in range(self.layers):
            p = f"encoder.layer.{i}."

            def lin(name, v):                       # one BLAS call on the flattened (B*n, .) matrix
                return v @ self._wt[p + name] + w[p + name + ".bias"]
            q = lin("attention.self.query", x).reshape(B, n, h, d).transpose(0, 2, 1, 3)
            k = lin("attention.self.key", x).reshape(B, n, h, d).transpose(0, 2, 3, 1)
            v = lin("attention.self.value", x).reshape(B, n, h, d).transpose(0, 2, 1, 3)
            s = (q @ k) * scale + bias
            s -= s.max(-1, keepdims=True)
            np.exp(s, out=s)
            s /= s.sum(-1, keepdims=True)
            ctx = (s @ v).transpose(0, 2, 1, 3).reshape(B * n, H)
            x = _layer_norm(x + lin("attention.output.dense", ctx),
                            w[p + "attention.output.LayerNorm.weight"], w[p + "attention.output.LayerNorm.bias"])
            ff = lin("output.dense", _gelu(lin("intermediate.dense", x)))
            x = _layer_norm(x + ff, w[p + "output.LayerNorm.weight"], w[p + "output.LayerNorm.bias"])
        return x.reshape(B, n, H)

    def encode(self, texts, batch_size=64):
        """Mean-pooled, L2-normalised sentence vectors, (len(texts), H) float32."""
        enc = [self.tok.encode(t) for t in texts]
        order = np.argsort([len(e) for e in enc], kind="stable")
        out = np.zeros((len(texts), self.hidden), np.float32)
        for a in range(0, len(order), batch_size):
            idx = order[a:a + batch_size]
            n = max(len(enc[j]) for j in idx)
            ids = np.full((len(idx), n), self.tok.pad, np.int64)
            mask = np.zeros((len(idx), n), np.float32)
            for r, j in enumerate(idx):
                ids[r, :len(enc[j])] = enc[j]
                mask[r, :len(enc[j])] = 1.0
            x = self.forward(ids, mask)
            pooled = (x * mask[:, :, None]).sum(1) / mask.sum(1, keepdims=True)
            pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
            out[idx] = pooled
        return out
