"""Named values and tables for amendment RV-02 (Section 4.7, wider BERT grid and longer input; Section 4.8, arXiv-8).

Read only from revision/rv02/: server_results/ (registered server scoring, LOCK files, gate, verification) and
scored/ (local scoring of the returned M6G/M6L predictions with the registered evaluate.py; author-clustered
bootstrap of arXiv-8 with revision/b1_cluster.py). Formatting as in values.py / values_rev.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import values_rev as vr

RV = Path(__file__).resolve().parents[1] / "revision" / "rv02"
SRV, SCO = RV / "server_results", RV / "scored"
NAMES = {"M1": "M1 TF-IDF + LR", "M2": "M2 TF-IDF + SVM", "M3": "M3 Counts + CNB", "M4": "M4 GloVe + LR",
         "M5F": "M5F MiniLM + LR", "M6": "M6 Fine-tuned BERT", "M6G": "M6G BERT, six-value grid",
         "M6L": "M6L BERT, 256 word pieces"}
CAT = {"astro-ph.GA": "astrophysics of galaxies", "cond-mat.mtrl-sci": "materials science", "cs.CL": "computation and language",
       "cs.CV": "computer vision", "hep-ph": "high-energy phenomenology", "math.AP": "analysis of PDEs",
       "math.CO": "combinatorics", "quant-ph": "quantum physics", "hep-th": "high-energy theory", "physics.optics": "optics"}


def J(p: Path) -> dict:
    return json.load(open(p))


def lr(x: float) -> str:
    m, e = f"{x:.0e}".split("e")
    return f"{int(m)} × 10^−{int(e[1:])}^"


def ratio(e: dict, n_: int) -> str:
    """ρ with interval from an exchange record of evaluate.py (rho, rho_ci; None = censored)."""
    def f(v, side):
        if v is None:
            return "> 1" if side else "< " + vr.r2(25 / n_)
        return vr.r2(v)
    r = e.get("rho")
    head = "> 1" if r is None and str(e["n_star"]).startswith(">") else f(r, 1)
    return f"{head} [{f(e['rho_ci'][0], 0)}, {f(e['rho_ci'][1], 1)}]"


def cost(c: dict) -> str:
    return f"{vr.pts(c['cost'] * 100)} {vr.ci_p([v * 100 for v in c['ci']])}"


def arxiv_models() -> dict:
    m = dict(J(SRV / "results_arxiv.json")["models"])
    a03 = J(SRV / "results_arxiv_A03.json")["models"]
    assert a03["M1"] == m["M1"]
    m["M5F"], m["M6"] = a03["M5F"], a03["M6"]
    return m


def bert(ds: str) -> dict:
    return J(SCO / f"results_{ds}_RV02.json")["models"]


def build() -> dict[str, str]:
    V: dict[str, str] = {}
    # ---------------- Part A
    for tag, ds in (("rd", "reddit"), ("se", "stackexchange")):
        b = bert(ds)
        lk = J(SRV / f"LOCK_{ds}_M6G.json")
        sel = lk["selected"]["M6G"]
        V[f"Q_{tag}_sel"] = "; ".join(f"{int(n):,}: {lr(v)}" for n, v in sel.items())
        V[f"Q_{tag}_sel_lo"], V[f"Q_{tag}_sel_hi"] = lr(min(sel.values())), lr(max(sel.values()))
        V[f"Q_{tag}_sel_first"], V[f"Q_{tag}_sel_last"] = lr(sel[min(sel, key=int)]), lr(sel[max(sel, key=int)])
        # spread of validation F1 over the learning rates 3e-5 ... 1e-4, largest over sizes (points)
        tab = lk["validation_macro_f1"]
        V[f"Q_{tag}_valspread"] = vr.pts(100 * max(max(v for h, v in t.items() if 3e-5 <= float(h) <= 1e-4)
                                                   - min(v for h, v in t.items() if 3e-5 <= float(h) <= 1e-4) for t in tab.values()))
        for m in ("M6", "M6G", "M6L"):
            x = b[m]
            V[f"Q_{tag}_{m}_F1"] = f"{x['same_year_macro_f1']:.3f}"
            V[f"Q_{tag}_{m}_C1"] = cost(x["staleness"]["forward"]["1"])
            V[f"Q_{tag}_{m}_C6"] = cost(x["staleness"]["forward"]["6"])
            V[f"Q_{tag}_{m}_C6pt"] = vr.pts(100 * x["staleness"]["forward"]["6"]["cost"])
            V[f"Q_{tag}_{m}_rho6"] = ratio(x["exchange"]["6"], vr.N[tag])
            V[f"Q_{tag}_{m}_rho6pt"] = vr.r2(x["exchange"]["6"]["rho"])
            V[f"Q_{tag}_{m}_n6"] = f"{vr.nst(x['exchange']['6']['n_star'])} {vr.ci_n(x['exchange']['6']['n_star_ci'])}"
        V[f"Q_{tag}_dF1_G"] = vr.pts(100 * abs(b["M6G"]["same_year_macro_f1"] - b["M6"]["same_year_macro_f1"]))
    V["Q_L_dF1_max"] = vr.pts(max(100 * abs(bert(ds)["M6L"]["same_year_macro_f1"] - bert(ds)["M6"]["same_year_macro_f1"])
                                  for ds in ("reddit", "stackexchange")))
    V["Q_L_drho_max"] = vr.r2(max(abs(bert(ds)["M6L"]["exchange"]["6"]["rho"] - bert(ds)["M6"]["exchange"]["6"]["rho"])
                                  for ds in ("reddit", "stackexchange")))
    ver = (SRV / "RV02_verify.txt").read_text()
    for tag, ds in (("rd", "reddit"), ("se", "stackexchange")):
        mm = re.search(rf"reproduce M6 select {ds}: (\d+) shared runs, max \|dF1\| = ([\d.]+), mean = ([\d.]+)", ver)
        V[f"Q_{tag}_rep_n"], V[f"Q_{tag}_rep_max"], V[f"Q_{tag}_rep_mean"] = mm.group(1), mm.group(2), mm.group(3)
    # ---------------- Part B
    g = J(SRV / "gate.json")["arxiv"]
    dec = g["decision"]
    cnt = J(SRV / "ax_counts_total.json")
    V["Q_ax_years"] = f"{dec['years'][0]}–{dec['years'][-1]}"
    V["Q_ax_K"] = str(len(dec["classes"]))
    V["Q_ax_cls"] = ", ".join(f"{CAT[c]} ({c})" for c in dec["classes"])
    V["Q_ax_dropped"] = " and ".join(f"{CAT[c]} ({c})" for c in cnt["candidates_A"] if c not in dec["classes"])
    V["Q_ax_N"], V["Q_ax_V"], V["Q_ax_T"] = (f"{v:,}" for v in dec["size"])
    V["Q_ax_items"] = f"{len(dec['classes']) * len(dec['years']) * (2 * dec['size'][0] + dec['size'][1] + dec['size'][2]):,}"
    c = cnt["counts"]
    V["Q_ax_read"], V["Q_ax_eligible"], V["Q_ax_xcross"] = f"{c['read']:,}", f"{c['kept_after_cap']:,}", f"{c['x_crosslisted_candidate']:,}"
    snap = J(SRV / "RV02_arxiv_snapshot.json")
    V["Q_ax_commit"] = snap["commit"][:12]
    V["Q_ax_fetched"] = snap["fetched_at"][:10]
    st = (SRV / "RV02_primary_check_status.txt").read_text()
    mm = re.search(r"agreement (\d+)/(\d+)", st)
    V["Q_ax_prim"] = f"{mm.group(1)} of {mm.group(2)}"
    m = arxiv_models()
    gm = len(dec["years"]) - 1
    V["Q_ax_gmax"] = str(gm)
    x = m["M1"]
    V["Q_ax_M1_F1"] = f"{x['same_year_macro_f1']:.3f}"
    V["Q_ax_M1_C1"] = cost(x["staleness"]["forward"]["1"])
    V["Q_ax_M1_Cg"] = cost(x["staleness"]["forward"][str(gm)])
    V["Q_ax_M1_Cgpt"] = vr.pts(100 * x["staleness"]["forward"][str(gm)]["cost"])
    e = x["exchange"][str(gm)]
    V["Q_ax_M1_n6only"] = vr.nst(e["n_star"])
    V["Q_ax_M1_ng"] = f"{vr.nst(e['n_star'])} {vr.ci_n(e['n_star_ci'])}"
    V["Q_ax_M1_rhog"] = ratio(e, 500)
    V["Q_ax_M1_rho1"] = vr.r2(x["exchange"]["1"]["rho"])
    V["Q_ax_M1_rhogpt"] = vr.r2(e["rho"])
    h = x["hypotheses"]["holm_p"]
    V["Q_ax_holm_H3"] = f"{h['H3']:.3f}"
    V["Q_ax_holm_H1"] = f"{h['H1']:.2f}"
    fc = x["fresh_curve"]
    V["Q_ax_gain"] = vr.pts(100 * (fc["500"] - fc["200"]))
    se = J(vr.REV.parents[1] / "results" / "results_stackexchange.json")["models"]["M1"]["fresh_curve"]
    V["Q_se_gain"] = vr.pts(100 * (se["500"] - se["200"]))
    cl = J(SCO / "cluster_arxiv.json")
    V["Q_ax_cl_ng"] = vr.ci_n(cl["models"]["M1"]["cluster"]["exchange"][str(gm)]["n_star_ci"])
    V["Q_ax_cl_single"] = vr.pct(cl["test_cells"]["share_single_item_authors"])
    rhos = {k: m[k]["exchange"][str(gm)]["rho"] for k in ("M2", "M3", "M4")}
    V["Q_ax_M234_lo"], V["Q_ax_M234_hi"] = vr.r2(min(rhos.values())), vr.r2(max(rhos.values()))
    V["Q_ax_M5F_rhog"] = ratio(m["M5F"]["exchange"][str(gm)], 500)
    V["Q_ax_M6_Cg"] = cost(m["M6"]["staleness"]["forward"][str(gm)])
    V["Q_ax_M6_rhog"] = ratio(m["M6"]["exchange"][str(gm)], 500)
    V["Q_ax_F1_min"] = f"{min(m[k]['same_year_macro_f1'] for k in m if k in NAMES):.3f}"
    V["Q_ax_F1_max"] = f"{max(m[k]['same_year_macro_f1'] for k in m if k in NAMES):.3f}"
    return V


def decide_ax(x: dict, gm: int) -> str:
    """Registered rules (Section 3.6) applied at the largest gap of arXiv-8."""
    s = x["staleness"]
    h = x["hypotheses"]["holm_p"]
    h1 = "S" if s["forward"]["1"]["ci"][0] > 0 and h["H1"] < 0.05 else "NS"
    h2 = "S" if s["forward_gmax_minus_g1"]["ci"][0] > 0 and h["H2"] < 0.05 else "NS"
    lo, hi = x["exchange"][str(gm)]["n_star_ci"]
    num = lambda v: float("inf") if str(v).startswith(">") else 0.0 if str(v).startswith("<") else float(v)
    h3 = "S" if num(hi) < 250 else ("NS" if num(lo) >= 250 else "I")
    return f"{h1} / {h2} / {h3}"


def table11() -> str:
    out = ["**Table 11.** ⟦+Fine-tuned BERT with the registered two-value learning-rate grid (M6), a six-value grid from 1 × 10^−5^ to 1 × 10^−4^ with the registered one-step extension (M6G) and input of 256 instead of 128 word pieces with the M6G learning rates (M6L), amendment RV-02. Costs in Macro-F1 points; break-even size *n*^\\*^(6) in labels per class and *ρ*(6) = *n*^\\*^(6)/*N*, with 95% stratified paired bootstrap intervals (item level, registered seed). Descriptive.+⟧", "",
           "| Dataset | Model | Learning rate by size (labels per class: rate) | Same-year F1 | *C*(1) | *C*(6) | *n*^\\*^(6) | *ρ*(6) |",
           "|:--|:--|:--|:--|:--|:--|:--|:--|"]
    V = build()
    for tag, ds, name in (("rd", "reddit", "Reddit-10"), ("se", "stackexchange", "Stack Exchange-10")):
        for m in ("M6", "M6G", "M6L"):
            rate = "5 × 10^−5^ at every size" if m == "M6" else (V[f"Q_{tag}_sel"] if m == "M6G" else "as M6G")
            out.append(f"| {name} | {NAMES[m]} | {rate} | {V[f'Q_{tag}_{m}_F1']} | {V[f'Q_{tag}_{m}_C1']} | "
                       f"{V[f'Q_{tag}_{m}_C6']} | {V[f'Q_{tag}_{m}_n6']} | {V[f'Q_{tag}_{m}_rho6']} |")
    return "\n".join(out) + "\n"


def table12() -> str:
    m = arxiv_models()
    dec = J(SRV / "gate.json")["arxiv"]["decision"]
    gm = len(dec["years"]) - 1
    cl = J(SCO / "cluster_arxiv.json")["models"]
    out = [f"**Table 12.** ⟦+arXiv-8 ({dec['years'][0]}–{dec['years'][-1]}, *N* = {dec['size'][0]}, {dec['size'][2]} test items per class and year; amendment RV-02). Costs in Macro-F1 points and ratios *ρ* = *n*^\\*^/*N*, with 95% stratified paired bootstrap intervals (item level, registered seed); *n*^\\*^({gm}) with the author-clustered interval in the last column. H1 / H2 / H3: decision under the registered rules for the primary model at *g*~max~ = {gm} (S, supported; NS, not supported; I, inconclusive). M5F and M6 are descriptive.+⟧", "",
           f"| Model | Same-year F1 | *C*(1) | *C*({gm}) | *ρ*(1) | *ρ*({gm}) | *n*^\\*^({gm}), author-clustered interval | H1 / H2 / H3 |",
           "|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for k in ("M1", "M2", "M3", "M4", "M5F", "M6"):
        x = m[k]
        e = cl[k]["cluster"]["exchange"][str(gm)]
        out.append(f"| {NAMES[k]} | {x['same_year_macro_f1']:.3f} | {cost(x['staleness']['forward']['1'])} | "
                   f"{cost(x['staleness']['forward'][str(gm)])} | {ratio(x['exchange']['1'], 500)} | "
                   f"{ratio(x['exchange'][str(gm)], 500)} | {vr.nst(e['n_star'])} {vr.ci_n(e['n_star_ci'])} | "
                   f"{decide_ax(x, gm) if k == 'M1' else '–'} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    for k, v in build().items():
        print(k, "=", v)
    print(table11())
    print(table12())
