"""RV-02: apply the registered D1 gate rule (plan 3.4, unchanged code: gate.reddit_gate) to arXiv-10.

python gate_arxiv.py -> adds key "arxiv" to results/gate.json (other keys untouched; refuses to overwrite "arxiv").
Candidate list A is gated first; list B only if A yields no decision (order fixed in AMENDMENT_RV-02.md).
"""
import json

from build_eligible_arxiv import CANDIDATES_A, CANDIDATES_B, MAX_YEAR
from common import RESULTS
from gate import reddit_gate


def main() -> None:
    path = RESULTS / "gate.json"
    full = json.load(open(path))
    if "arxiv" in full:
        raise SystemExit("gate.json already has an arxiv entry; not overwritten")
    used = "A"
    res = reddit_gate(CANDIDATES_A, "ax_", max_year=MAX_YEAR)
    if res["decision"] is None:
        used = "B"
        res = reddit_gate(CANDIDATES_B, "ax_", max_year=MAX_YEAR)
    res["candidate_list_used"] = used
    full["arxiv"] = res
    json.dump(full, open(path, "w"), indent=1)
    print("arxiv", used, json.dumps(res["decision"]))
    if res["decision"] is None:
        raise SystemExit("STOP: arXiv-10 failed the data gate with both registered candidate lists")


if __name__ == "__main__":
    main()
