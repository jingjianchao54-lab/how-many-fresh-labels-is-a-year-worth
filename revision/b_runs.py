"""Revision sensitivity refits for the primary model M1 (Reviewer 2, comments 1-3, 6-8; Reviewer 1, comment 9).

python b_runs.py <dataset> <variant> <stage> [worker n_workers]
variants (each writes revision/rwork/<variant>/runs/<dataset>/test/M1/*.npz with the registered file names):
  check      registered text; replicate 1 only; used to verify that refitting reproduces the stored predictions
  pii        text with e-mail addresses, phone numbers, user handles and IP addresses replaced by placeholders (B3)
  english    registered text; analysis restricted afterwards to items identified as English (B4; no refit needed)
  emoji      tokeniser that also keeps emoji and common text emoticons as tokens (B5)
  sitemask   Stack Exchange with site-name tokens removed, as on Reddit (B7)
  fixedC     E2 with C fixed at the value selected for N at every size (B2)
  widegrid   selection over C in {0.01, 0.1, 1, 10, 100, 1000}, then E2 with the new selection (B2)
  mixw       E3 with old and current items weighted so that both sets carry equal total weight (B6)
  mixtune    E3 with C re-selected for the combined set on same-year validation data, replicate 1 (B6)
stage: "select" (only widegrid, mixtune) or "test".
Every variant keeps the registered splits, replicate draws and evaluation sets; only the stated element changes.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import common  # noqa: E402
import features  # noqa: E402
import run  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

FULL = ROOT / "revision" / "full"
RW = ROOT / "revision" / "rwork"
WIDE = [0.01, 0.1, 1, 10, 100, 1000]

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
HANDLE_RE = re.compile(r"(?<![\w/])(?:/?u/[\w-]{3,}|@[A-Za-z0-9_]{2,})")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]")
EMOTICON_RE = re.compile(r"(?:[:;=xX8][-o^']?[)(\]\[dDpP/\\|*3]|<3|\^_?\^|[)(][-']?[:;=])(?=\s|$|[.,!?])")
SE_NAMES = {"philosophy", "photo", "photography", "music", "history", "cooking", "movies", "movie", "bicycles",
            "bicycle", "boardgames", "boardgame", "gardening", "chess"}


def pii_scrub(text: str) -> tuple[str, dict[str, int]]:
    counts = {}
    for name, rx, ph in (("email", EMAIL_RE, " emailaddr "), ("ip", IP_RE, " ipaddr "), ("phone", PHONE_RE, " phonenum "),
                         ("handle", HANDLE_RE, " userhandle ")):
        text, k = rx.subn(ph, text)
        counts[name] = k
    return text, counts


def tokens_emoji(text: str, mask: set[str] | None = None) -> list[str]:
    t = common.URL_RE.sub(" ", text)
    out = []
    pat = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|" + EMOJI_RE.pattern + "|" + EMOTICON_RE.pattern)
    for m in pat.finditer(t.lower() if True else t):
        tok = m.group(0)
        if mask and tok in mask:
            continue
        out.append(tok)
    return out[: common.MAX_TOKENS]


def build_variant(ds: str, variant: str) -> Path:
    """Write revision/rwork/<variant>/sample_<ds>.pkl with the variant's 'tok' field; return the work dir."""
    wd = RW / variant
    wd.mkdir(parents=True, exist_ok=True)
    path = wd / f"sample_{ds}.pkl"
    if path.exists():
        return wd
    s = pickle.load(open(FULL / f"sample_{ds}.pkl", "rb"))
    mask = common.NAME_TOKENS if ds == "reddit" else None
    stats = {}
    if variant == "pii":
        tok, tot, items = [], {}, 0
        for raw in s["raw"]:
            t, c = pii_scrub(raw)
            items += int(sum(c.values()) > 0)
            for k, v in c.items():
                tot[k] = tot.get(k, 0) + v
            tok.append(" ".join(common.tokens(t, mask_names=(ds == "reddit"))))
        s["tok"] = tok
        stats = {"replacements": tot, "items_with_any": items, "items": len(tok)}
    elif variant == "emoji":
        tok, items = [], 0
        for raw in s["raw"]:
            tt = tokens_emoji(raw, mask)
            items += int(any(EMOJI_RE.fullmatch(x) or EMOTICON_RE.fullmatch(x) for x in tt))
            tok.append(" ".join(tt))
        s["tok"] = tok
        stats = {"items_with_emoji_or_emoticon": items, "items": len(tok)}
    elif variant == "sitemask":
        assert ds == "stackexchange"
        s["tok"] = [" ".join(w for w in t.split() if w not in SE_NAMES) for t in s["tok"]]
    keep = {k: s[k] for k in ("cls", "year", "part", "line", "tok", "classes", "years", "size")}
    pickle.dump(keep, open(path, "wb"), protocol=4)
    json.dump(stats, open(wd / f"stats_{ds}.json", "w"), indent=1)
    return wd


def point_to(wd: Path) -> None:
    for mod in (common, features, run):
        mod.WORK = wd


def lock(ds: str) -> dict[str, float]:
    return json.load(open(ROOT / "results" / f"LOCK_{ds}.json"))["selected"]["M1"]


def fit(d, h, tr, evals, weight=None):
    tv = features.TrainVocab(d.x, tr, tfidf=True, max_features=features.MAX_FEATURES)
    clf = LogisticRegression(C=h, solver="lbfgs", max_iter=1000).fit(tv.transform(tr), d.cls[tr], sample_weight=weight)
    return [clf.predict(tv.transform(r)).astype(np.int8) for r in evals], bool(np.all(clf.n_iter_ < 1000))


def main() -> None:
    ds, variant, stage = sys.argv[1:4]
    wi, nw = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)
    base = "registered" if variant in ("check", "english", "fixedC", "widegrid", "mixw", "mixtune") else variant
    wd = build_variant(ds, base) if base != "registered" else RW / "registered"
    if base == "registered":
        wd.mkdir(parents=True, exist_ok=True)
        src = FULL / f"sample_{ds}.pkl"
        if not (wd / f"sample_{ds}.pkl").exists():
            s = pickle.load(open(src, "rb"))
            pickle.dump({k: s[k] for k in ("cls", "year", "part", "line", "tok", "classes", "years", "size")},
                        open(wd / f"sample_{ds}.pkl", "wb"), protocol=4)
    point_to(wd)
    d = run.Data(ds)
    out = RW / variant / "runs" / ds / stage / "M1"
    out.mkdir(parents=True, exist_ok=True)
    sel = lock(ds)
    gmax = d.years[-1] - d.years[0]
    jobs = []
    if stage == "select":
        if variant == "widegrid":
            jobs = [("sel", 1, t, n, h) for t in d.years for n in d.sizes for h in WIDE]
        elif variant == "mixtune":
            jobs = [("msel", 1, t, n, (g, h)) for g in (gmax,) for t in d.years if t - g >= d.years[0]
                    for n in d.sizes[:-1] for h in run.GRID["M1"]]
    else:
        reps = [1] if variant == "check" else [1, 2, 3, 4, 5]
        if variant in ("check", "pii", "emoji", "sitemask"):
            jobs = [j for j in run.test_jobs(d) if j[1] in reps and j[0] != "e3"]
            if variant == "check":
                jobs = [j for j in jobs if j[2] in (d.years[0], d.years[-1])]
        elif variant in ("fixedC", "widegrid"):
            jobs = [j for j in run.test_jobs(d) if j[0] == "e2"]
            if variant == "widegrid":
                jobs += [j for j in run.test_jobs(d) if j[0] == "e1"]
                lock_sel = lock(ds)
                new_sel = json.load(open(RW / "widegrid" / f"selection_{ds}.json"))
                # E2 runs whose selected value is unchanged are identical to the registered ones (linked, not refitted)
                jobs = [j for j in jobs if j[0] == "e1" or float(new_sel[str(j[3])]) != float(lock_sel[str(j[3])])]
                sel = json.load(open(RW / "widegrid" / f"selection_{ds}.json"))
        elif variant in ("mixw", "mixtune"):
            jobs = [j for j in run.test_jobs(d) if j[0] == "e3" and j[4] == gmax]
            if variant == "mixtune":
                msel = json.load(open(RW / "mixtune" / f"selection_{ds}.json"))
    for j, job in enumerate(jobs):
        if j % nw != wi:
            continue
        kind, r, t, n, extra = job
        name = "_".join(str(v) for v in job) if kind != "msel" else f"msel_{t}_{n}_{extra[0]}_{extra[1]}"
        path = out / (name + ".npz")
        if path.exists():
            continue
        t0 = time.time()
        if kind == "sel":
            rows = d.rows(1, t)
            (pred,), ok = fit(d, extra, d.train_rows(t, r, n), [rows])
            np.savez(path, f1=f1_score(d.cls[rows], pred, average="macro"), ok=ok)
        elif kind == "msel":
            g, h = extra
            rows = d.rows(1, t)
            tr = np.sort(np.concatenate([d.train_rows(t - g, 1, d.N), d.train_rows(t, 1, n)]))
            (pred,), ok = fit(d, h, tr, [rows])
            np.savez(path, f1=f1_score(d.cls[rows], pred, average="macro"), ok=ok)
        elif kind == "e1":
            preds, ok = fit(d, sel[str(n)], d.train_rows(t, r, n), [d.rows(2, u) for u in d.years])
            np.savez(path, pred=np.stack(preds), ok=ok)
        elif kind == "e2":
            h = sel[str(d.N)] if variant == "fixedC" else sel[str(n)]
            preds, ok = fit(d, h, d.train_rows(t, r, n), [d.rows(2, t)])
            np.savez(path, pred=np.stack(preds), ok=ok)
        else:                                                 # e3
            old, new = d.train_rows(t - extra, r, d.N), d.train_rows(t, r, n)
            tr = np.concatenate([old, new])
            order = np.argsort(tr)
            w = None
            if variant == "mixw":
                w = np.concatenate([np.full(len(old), len(new) / len(old)), np.ones(len(new))])[order]
                w = w * len(tr) / w.sum()
            h = msel[f"{extra}_{n}"] if variant == "mixtune" else sel[str(d.N)]
            preds, ok = fit(d, h, tr[order], [d.rows(2, t)], weight=w)
            np.savez(path, pred=np.stack(preds), ok=ok)
        print(variant, job, round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
