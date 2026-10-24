"""Feature construction. TF-IDF statistics come from the training rows only (plan Section 4).

A global 1-2-gram count matrix is built once; for each training set the vocabulary (document frequency
>= 2 in the training rows, at most 200,000 terms by training term count), the IDF weights and the L2
normalisation are computed from the training rows alone, which equals fitting a vectoriser on them.
"""
import pickle

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from common import DATA, WORK

MAX_FEATURES = 200_000


def global_counts(dataset, field="tok"):
    path = WORK / f"counts_{dataset}_{field}.npz"
    if path.exists():
        return sp.load_npz(path)
    s = pickle.load(open(WORK / f"sample_{dataset}.pkl", "rb"))
    cv = CountVectorizer(ngram_range=(1, 2), token_pattern=r"\S+", lowercase=False, min_df=2, dtype=np.int32)
    x = cv.fit_transform(s[field]).tocsr()
    x.sort_indices()
    sp.save_npz(path, x)
    return x


class TrainVocab:
    """Vocabulary and IDF learned from training rows of the global count matrix."""

    def __init__(self, x, train_rows, tfidf=True, max_features=MAX_FEATURES):
        xt = x[train_rows]
        n = xt.shape[0]
        df = np.bincount(xt.indices, minlength=x.shape[1])
        keep = df >= 2
        if max_features and keep.sum() > max_features:
            tf = np.asarray(xt.sum(axis=0)).ravel()
            cand = np.flatnonzero(keep)
            order = cand[np.argsort(-tf[cand], kind="stable")[:max_features]]
            keep = np.zeros_like(keep)
            keep[order] = True
        self.remap = np.full(x.shape[1], -1, dtype=np.int64)
        self.remap[keep] = np.arange(int(keep.sum()))
        self.k = int(keep.sum())
        self.tfidf = tfidf
        self.idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
        self.x = x

    def transform(self, rows):
        xr = self.x[rows].tocsr()
        new = self.remap[xr.indices]
        m = new >= 0
        if self.tfidf:
            data = (1.0 + np.log(xr.data[m].astype(np.float64))) * self.idf[xr.indices[m]]
        else:
            data = xr.data[m].astype(np.float64)
        row_of = np.repeat(np.arange(xr.shape[0]), np.diff(xr.indptr))[m]
        counts = np.bincount(row_of, minlength=xr.shape[0])
        indptr = np.concatenate([[0], np.cumsum(counts)])
        out = sp.csr_matrix((data, new[m], indptr), shape=(xr.shape[0], self.k))
        if self.tfidf:
            norm = np.sqrt(np.asarray(out.multiply(out).sum(axis=1)).ravel())
            norm[norm == 0] = 1.0
            out = sp.diags(1.0 / norm) @ out
        return out.tocsr()


def glove_means(dataset, field="tok"):
    path = WORK / f"glove_{dataset}_{field}.npy"
    if path.exists():
        return np.load(path)
    s = pickle.load(open(WORK / f"sample_{dataset}.pkl", "rb"))
    need = set()
    for t in s[field]:
        need.update(t.split())
    vec = {}
    with open(DATA / "glove.6B.100d.txt", encoding="utf-8") as f:
        for line in f:
            w, rest = line.split(" ", 1)
            if w in need:
                vec[w] = np.asarray(rest.split(), dtype=np.float32)
    out = np.zeros((len(s[field]), 100), dtype=np.float32)
    for i, t in enumerate(s[field]):
        vs = [vec[w] for w in t.split() if w in vec]
        if vs:
            out[i] = np.mean(vs, axis=0)
    np.save(path, out)
    return out
