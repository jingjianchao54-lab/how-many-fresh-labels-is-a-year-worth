"""Shared definitions for the label-recency study (analysis plan v1.0, Sections 3.1-3.5)."""
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data"
WORK = ROOT / "work"
RESULTS = ROOT / "results"

SUBREDDITS = ["chess", "Coffee", "Cooking", "gardening", "history", "investing",
              "languagelearning", "philosophy", "running", "woodworking"]
NAME_TOKENS = {s.lower() for s in SUBREDDITS}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
BOT_RE = re.compile(r"bot\d*$|^auto[-_]?mod", re.IGNORECASE)
MAX_TOKENS = 256
MIN_TOKENS = {"reddit": 10, "huffpost": 5}
AUTHOR_CAP = 5

LADDER = {"reddit": [(2000, 500, 1000), (1000, 250, 500), (500, 150, 300)],
          "huffpost": [(1000, 200, 500), (500, 100, 300), (300, 100, 200)]}
SIZES = [25, 50, 100, 200, 500, 1000, 2000]


def tokens(text, mask_names=False):
    """Lower-case, strip URLs, tokenise, optionally drop subreddit-name tokens, truncate to 256."""
    toks = TOKEN_RE.findall(URL_RE.sub(" ", text.lower()))
    if mask_names:
        toks = [t for t in toks if t not in NAME_TOKENS]
    return toks[:MAX_TOKENS]


def split_u(key):
    """Deterministic uniform value in [0, 1) from a string key (author name or record link)."""
    h = hashlib.sha256(("paper5|" + key).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


def split_of(key):
    u = split_u(key)
    return 0 if u < 0.60 else (1 if u < 0.70 else 2)   # 0 train, 1 validation, 2 test


def is_excluded_author(name):
    return name == "[deleted]" or name == "AutoModerator" or BOT_RE.search(name) is not None
