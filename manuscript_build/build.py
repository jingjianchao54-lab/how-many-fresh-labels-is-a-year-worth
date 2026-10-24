"""Fill the manuscript template with values and tables from the result files, number the citations,
append the reference list, and write manuscript.md, manuscript.txt and manuscript.docx.

python build.py [template.md]
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import values
from refs import REFS

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "exp" / "figures"
V = values.build()
INFO = {  # filled once the author supplies them
    "AUTHOR_BLOCK": "[Author name, affiliation, address, e-mail and ORCID to be supplied]",
    "AUTHOR_CONTRIB": "Conceptualization, J.J.; methodology, J.J.; software, J.J.; validation, J.J.; formal analysis, J.J.; investigation, J.J.; data curation, J.J.; writing—original draft preparation, J.J.; writing—review and editing, J.J.; visualization, J.J. The author has read and agreed to the published version of the manuscript.",
    "FUNDING": "[To be completed by the author]",
    "REPO_URL": "[repository URL]",
    "ZENODO_DOI": "[Zenodo DOI]",
    "AI_DISCLOSURE": "During the preparation of this manuscript, the authors used Claude (Anthropic) to assist with language refinement and the preliminary synthesis of selected materials. All AI-assisted content was critically reviewed and revised by the authors. The authors take full responsibility for the accuracy, originality, integrity, and final content of the manuscript.",
}


def ds_models(tag):
    return values.models(values.DS[tag])


def table1():
    return """**Table 1.** Datasets and design. *N*, *V* and *T* are the training, validation and test items per (class, year) cell; the training pool holds 2*N* items per cell.

| | Reddit-10 | Stack Exchange-10 |
|:--|:--|:--|
| Source | ConvoKit subreddit corpora | Stack Exchange data dump, April 2024 |
| Unit | Post or comment | Question, answer or comment |
| Classes | 10 subreddits | 10 sites |
| Years | 2012–2018 | 2015–2021 |
| *N* / *V* / *T* per cell | «rd_N» / 500 / 1,000 | «se_N» / 150 / 300 |
| Training sizes *n* per class | 25, 50, 100, 200, 500, 1,000, 2,000 | 25, 50, 100, 200, 500 |
| Items sampled | 385,000 | 101,500 |
| Size level passed at the gate | Largest of three | Smallest of three |
"""


def table2():
    import json
    sel = {d: json.load(open(values.RES / f"LOCK_{d}.json"))["selected"] for d in ("reddit", "stackexchange")}
    for d in sel:
        for k in ("M5", "M5F", "M6"):
            sel[d][k] = json.load(open(values.RES / f"LOCK_{d}_{k}.json"))["selected"][k]
    rows = [
        ("M1 (primary)", "TF-IDF, word 1–2-grams", "Logistic regression", "*C* ∈ {1, 10, 100}"),
        ("M2", "TF-IDF, word 1–2-grams", "Linear SVM", "*C* ∈ {0.1, 1, 10}"),
        ("M3", "Counts, word 1–2-grams", "Complement naive Bayes", "*α* ∈ {0.1, 0.5, 1.0}"),
        ("M4", "Mean GloVe 6B 100d vector", "Logistic regression (standardised inputs)", "*C* ∈ {0.1, 1, 10}"),
        ("M5", "Frozen all-MiniLM-L6-v2, mean pooled; 128 word pieces, replicate 1 only", "Logistic regression", "*C* ∈ {0.1, 1, 10}"),
        ("M5F (added)", "Frozen all-MiniLM-L6-v2, mean pooled; 256 word pieces", "Logistic regression", "*C* ∈ {0.1, 1, 10}"),
        ("M6 (added)", "bert-base-uncased, fine-tuned end to end; 128 word pieces", "Linear layer on [CLS]", "Learning rate ∈ {2 × 10^−5^, 5 × 10^−5^}"),
    ]
    out = ["**Table 2.** Models, hyper-parameter grids and the value selected on validation data at the full cell size *N* (Reddit-10 / Stack Exchange-10). M5F and M6 were added after the main results were known (amendment A-03; Table 3, D8). TF-IDF, term frequency–inverse document frequency; SVM, support vector machine; GloVe, Global Vectors; BERT, Bidirectional Encoder Representations from Transformers.", "",
           "| Model | Representation | Classifier | Grid | Selected at *N* |", "|:--|:--|:--|:--|:--|"]
    for (mid, rep, clf, grid) in rows:
        k = mid.split()[0]
        a = sel["reddit"][k][str(2000)]; b = sel["stackexchange"][k][str(500)]
        a, b = (f"{float(x) * 1e5:g} × 10^−5^" if k == "M6" else x for x in (a, b))
        out.append(f"| {mid} | {rep} | {clf} | {grid} | {a} / {b} |")
    return "\n".join(out) + "\n"


def table3():
    return """**Table 3.** Deviations from the analysis plan, with the amendment-log entries that record them. D1–D5 were recorded before any score they affect was computed; D6 was decided when the half-size run could not use 250 labels, before its result was scored; D7 changed only how two printed values were displayed; D8 was registered before any score of the two added models was computed, but after the results of M1–M5 were known.

| Id | Deviation | Reason | Effect on results |
|:--|:--|:--|:--|
| D1 | Second dataset HuffPost replaced by Stack Exchange (A-01) | HuffPost failed the data gate (R-01) | Registered before any Stack Exchange file was downloaded; same rules as Reddit-10 |
| D2 | Stack Exchange-10 analysed at the smallest size, *N* = «se_N» (R-02) | Larger sizes failed the gate | Training sizes up to «se_N» per class |
| D3 | Implementation choices the plan left open (N-01): E3 uses the value selected at size *N*; replicate draws are seeded by (replicate, class, year) and the fixed samples by (part, class, year) | The plan named no value for the combined set and one seed per replicate | None on the comparisons; recorded before validation or test scoring |
| D4 | MiniLM on replicate 1 only, input truncated at 128 word pieces (A-02) | Encoding without a GPU was too slow for the full design | M5 has no replicate spread; the full design was run later as M5F (D8) |
| D5 | MiniLM selected and locked separately (X-03) | Its vectors did not exist when the main locks were written | Same selection rule as the other models |
| D6 | Half-size sensitivity on Stack Exchange-10 uses 200 labels per class; no headline-only or name-masking analysis there (A-01, X-04) | 250 is not on the size grid; posts have no headline and site names are not used as references | Sensitivity analyses only |
| D7 | One censored upper bound and its ratio printed as censored instead of as numbers (X-02) | Interpolation with the censoring sentinel printed a meaningless value | Re-run with the same seed; only those two fields changed |
| D8 | Two models added: MiniLM with the planned design (M5F) and a fine-tuned BERT (M6); E3 for M6 on replicate 1 only (A-03) | A GPU became available after the main analysis | Added analysis, reported descriptively; H1–H3 remain defined for M1 only; M1–M5 results unchanged |
"""


def tableA1():
    import json
    out = ["**Table A1.** Hyper-parameter values selected on validation data for each training size *n* (labels per class). For M6 the value is the learning rate.", "",
           "| Dataset | Model | " + " | ".join(f"*n* = {n:,}" for n in (25, 50, 100, 200, 500, 1000, 2000)) + " |",
           "|:--|:--|" + ":--|" * 7]
    for d, name in (("reddit", "Reddit-10"), ("stackexchange", "Stack Exchange-10")):
        sel = json.load(open(values.RES / f"LOCK_{d}.json"))["selected"]
        for k in ("M5", "M5F", "M6"):
            sel[k] = json.load(open(values.RES / f"LOCK_{d}_{k}.json"))["selected"][k]
        for k in values.MODELS:
            fmt = (lambda x: f"{float(x) * 1e5:g} × 10^−5^") if k == "M6" else str
            out.append(f"| {name} | {k} | " + " | ".join(fmt(sel[k][str(n)]) if str(n) in sel[k] else "–" for n in (25, 50, 100, 200, 500, 1000, 2000)) + " |")
    return "\n".join(out) + "\n"


NAMES = {"M1": "M1 TF-IDF + LR", "M2": "M2 TF-IDF + SVM", "M3": "M3 Counts + CNB", "M4": "M4 GloVe + LR",
         "M5": "M5 MiniLM + LR (reduced)", "M5F": "M5F MiniLM + LR", "M6": "M6 Fine-tuned BERT"}


def tableA2():
    import json
    out = ["**Table A2.** Label-equivalence ratio computed separately for each pair of years (u − g → u), by gap *g*: smallest and largest value over the pairs, with the number of pairs in brackets. Censored pairs are counted in the bracket but not in the range; all of them had a negative cost, i.e. the old labels scored higher than the current ones. Point estimates from the stored test predictions, averaged over five replicates, without bootstrap; added after the results were known (amendment log X-11). Gap 6 has one pair and equals the value in Table 4.", "",
           "| Dataset | Model | *g* = 1 | *g* = 2 | *g* = 3 | *g* = 4 | *g* = 5 |", "|:--|:--|:--|:--|:--|:--|:--|"]
    for d, name in (("reddit", "Reddit-10"), ("stackexchange", "Stack Exchange-10")):
        yp = json.load(open(values.RES / f"year_pairs_{d}.json"))["models"]
        for k in ("M1", "M2", "M3", "M4", "M5F", "M6"):
            cells = []
            for g in ("1", "2", "3", "4", "5"):
                x = yp[k][g]
                c = f" ({x['n_pairs']}; {x['n_censored']} censored)" if x["n_censored"] else f" ({x['n_pairs']})"
                cells.append(f"{values.r2(x['rho_min'])}–{values.r2(x['rho_max'])}{c}")
            out.append(f"| {name} | {NAMES[k]} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def fmt_row(tag, k, v):
    p = f"{tag}_{k}"
    return (f"| {NAMES[k]} | {V[p+'_F1']} | {V[p+'_C1']} {V[p+'_C1ci']} | {V[p+'_C6']} {V[p+'_C6ci']} | "
            f"{V[p+'_n1']} {V[p+'_n1ci']} | {V[p+'_n6']} {V[p+'_n6ci']} | {V[p+'_rho1']} | {V[p+'_rho6']} |")


def table4():
    out = ["**Table 4.** Main results. Same-year Macro-F1 at size *N*; staleness cost *C*(*g*) in Macro-F1 points; break-even size *n*^\\*^(*g*) in labels per class; label-equivalence ratio *ρ*(*g*). Brackets give 95% stratified paired bootstrap intervals. M1–M4, M5F and M6 average five replicates; M5 uses replicate 1 only. M5F and M6 were added after the main results were known (Table 3, D8) and are reported descriptively. LR, logistic regression; SVM, support vector machine; CNB, complement naive Bayes; GloVe, Global Vectors; BERT, Bidirectional Encoder Representations from Transformers.", ""]
    for tag, name in (("rd", "Reddit-10"), ("se", "Stack Exchange-10")):
        out += [f"*{name} (N = {V[tag + '_N']})*", "",
                "| Model | Same-year F1 (0–1) | *C*(1) (points) | *C*(6) (points) | *n*^\\*^(1) | *n*^\\*^(6) | *ρ*(1) | *ρ*(6) |",
                "|:--|:--|:--|:--|:--|:--|:--|:--|"]
        out += [fmt_row(tag, k, None) for k in values.MODELS]
        out.append("")
    return "\n".join(out)


def table5():
    out = ["**Table 5.** Training on *N* six-year-old labels plus *n* current labels per class, compared with the old labels alone and with the current labels alone. Differences in Macro-F1 points with 95% bootstrap intervals. M1 and M5F average five replicates; M6 uses replicate 1 only (Table 3, D8). Results for one-year-old labels and for M2–M5 are in the repository.", "",
           "| Dataset | Model | *n* | Combined − old only | Combined − current only |", "|:--|:--|:--|:--|:--|"]
    for tag, name, ns in (("rd", "Reddit-10", (25, 100, 500, 1000)), ("se", "Stack Exchange-10", (25, 100, 200))):
        for k in ("M1", "M5F", "M6"):
            for g in (6,):
                for n in ns:
                    p = f"{tag}_{k}"
                    out.append(f"| {name} | {NAMES[k]} | {n:,} | {V[f'{p}_mixS_g{g}_n{n}']} {V[f'{p}_mixSci_g{g}_n{n}']} | {V[f'{p}_mixF_g{g}_n{n}']} {V[f'{p}_mixFci_g{g}_n{n}']} |")
    return "\n".join(out) + "\n"


def table6():
    out = ["**Table 6.** Sensitivity analyses for the primary model. Costs in Macro-F1 points; break-even sizes in labels per class; 95% stratified paired bootstrap intervals in brackets.", "",
           "| Setting | Old labels per class | Same-year F1 (0–1) | *C*(1) (points) | *C*(6) (points) | *n*^\\*^(1) | *n*^\\*^(6) | *ρ*(1) | *ρ*(6) |", "|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    def row(label, size, f1, p):
        return (f"| {label} | {size} | {f1} | {V[p+'C1']} {V[p+'C1ci']} | {V[p+'C6']} {V[p+'C6ci']} | "
                f"{V[p+'n1']} {V[p+'n1ci']} | {V[p+'n6']} {V[p+'n6ci']} | {V[p+'rho1']} | {V[p+'rho6']} |")
    out.append(row("Reddit-10, main analysis", V['rd_N'], V['rd_M1_F1'], "rd_M1_"))
    out.append(row("Reddit-10, half size", V['rd_half_size'], V['rd_half_F1'], "rd_half_"))
    out.append(row("Reddit-10, names not masked", V['rd_N'], V['raw_F1'], "raw_"))
    out.append(row("Stack Exchange-10, main analysis", V['se_N'], V['se_M1_F1'], "se_M1_"))
    out.append(row("Stack Exchange-10, reduced size", V['se_half_size'], V['se_half_F1'], "se_half_"))
    return "\n".join(out) + "\n"


def fill(s):
    tables = {"TABLE1": table1(), "TABLE2": table2(), "TABLE3": table3(), "TABLE4": table4(), "TABLE5": table5(), "TABLE6": table6(), "TABLEA1": tableA1(), "TABLEA2": tableA2()}
    for k, t in tables.items():
        s = s.replace(f"«{k}»", t)
    for _ in range(2):
        s = re.sub(r"«([A-Za-z0-9_]+)»", lambda m: str(INFO.get(m.group(1), V.get(m.group(1), m.group(0)))), s)
    left = re.findall(r"«[^»]+»", s)
    if left:
        sys.exit(f"unfilled placeholders: {sorted(set(left))}")
    return s


def number_citations(s):
    order = []
    def keys(block):
        return [k.strip().lstrip("@") for k in block.split(";")]
    for m in re.finditer(r"\[(@[^\]]+)\]", s):
        for k in keys(m.group(1)):
            if k not in order:
                order.append(k)
    missing = [k for k in order if k not in REFS]
    if missing:
        sys.exit(f"keys without reference entries: {missing}")
    idx = {k: i + 1 for i, k in enumerate(order)}
    def compress(nums):
        nums = sorted(set(nums)); out = []; i = 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ",".join(str(x) for x in nums[i:j + 1]))
            i = j + 1
        return "[" + ",".join(out) + "]"
    s = re.sub(r"\[(@[^\]]+)\]", lambda m: compress([idx[k] for k in keys(m.group(1))]), s)
    refs = "\n".join(f"{i}. {REFS[k]}" for i, k in enumerate(order, 1))
    return s + "\n" + refs + "\n", order


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "manuscript_template.md"
    s = fill(open(src, encoding="utf-8").read())
    s, order = number_citations(s)
    for f in FIG.glob("fig*.png"):
        shutil.copy(f, HERE / f.name)
    (HERE / "manuscript.md").write_text(s, encoding="utf-8")
    (HERE / "cited_keys.txt").write_text("\n".join(order) + "\n")
    subprocess.run(["pandoc", "manuscript.md", "-o", "manuscript.docx"], cwd=HERE, check=True)
    subprocess.run(["pandoc", "manuscript.md", "-t", "plain", "--wrap=none", "-o", "manuscript.txt"], cwd=HERE, check=True)
    words = len(re.findall(r"\w+", s.split("\n# References")[0]))
    print("references:", len(order), "words (to references):", words)


if __name__ == "__main__":
    main()
