"""Convert the filled manuscript (ms/manuscript.md) into a Word file laid out with the MDPI Algorithms template.

python to_mdpi.py -> Paper5_Algorithms_MDPI.docx
Only layout changes: every word, number and citation comes from manuscript.md unchanged.
"""
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MS = HERE.parent / "ms"
INFO = {
    "authors": "Jianzhao Jing ^1,2,\\*^, Chia Sien Lim ^2^, Muhammad Ehsan Rana ^3^, Chunhui Wang ^1^, Xianjuan Shang ^1^, Mei Liu ^1^ and Mengxia Wang ^1,\\*^",
    "affil": ["^1^\tSchool of Computer Science, Shandong Xiehe University, Jinan 250109, China; wangchunhui@sdxiehe.edu.cn (C.W.); shangxianjuan@sdxiehe.edu.cn (X.S.); liumei@sdxiehe.edu.cn (M.L.)",
              "^2^\tGraduate School of Technology, Asia Pacific University of Technology and Innovation, Technology Park Malaysia, Kuala Lumpur 57000, Malaysia; lim.chiasien@apu.edu.my",
              "^3^\tSchool of Computing, Asia Pacific University of Technology and Innovation, Technology Park Malaysia, Kuala Lumpur 57000, Malaysia; muhd_ehsanrana@apu.edu.my",
              "\\*\tCorrespondence: TP079489@mail.apu.edu.my (J.J.); wangmengxia@sdxiehe.edu.cn (M.W.)"],
}


def div(style, text):
    return f'::: {{custom-style="{style}"}}\n{text}\n:::\n'


def convert(md):
    title = re.search(r'title: "(.*)"', md).group(1)
    body = md.split("---\n", 2)[2]
    blocks = [b for b in re.split(r"\n\s*\n", body) if b.strip()]
    out = [div("MDPI_1.1_article_type", "Article"), div("MDPI_1.2_title", title), div("MDPI_1.3_authornames", INFO["authors"])]
    out += [div("MDPI_1.6_affiliation", a) for a in INFO["affil"]]
    fig = 0
    i = 0
    mode = "body"
    while i < len(blocks):
        b = blocks[i].strip()
        if b.startswith("**Author:**"):
            i += 1; continue
        if b == "# Abstract":
            out.append(div("MDPI_1.7_abstract", "**Abstract:** " + blocks[i + 1].strip()))
            i += 2; continue
        if b.startswith("**Keywords:**"):
            out.append(div("MDPI_1.8_keywords", b)); i += 1; continue
        if b == "# Back Matter":
            mode = "back"; i += 1; continue
        if b == "# References":
            mode = "refs"; out.append(div("MDPI_2.1_heading1", "References")); i += 1; continue
        if b.startswith("# "):
            out.append(div("MDPI_2.1_heading1", re.sub(r"^(\d+)\. ", r"\1\\. ", b[2:]))); i += 1
            if b.startswith("# Appendix"):
                mode = "app"
            continue
        if b.startswith("## "):
            out.append(div("MDPI_2.2_heading2", b[3:])); i += 1; continue
        if b.startswith("!["):
            m = re.match(r"!\[(.*)\]\((.*?)\)$", b, re.S)
            fig += 1
            out.append(div("MDPI_5.2_figure", f"![]({m.group(2)}){{width=16.5cm}}"))
            out.append(div("MDPI_5.1_figure_caption", f"**Figure {fig}.** {m.group(1)}"))
            i += 1; continue
        if b.startswith("$$"):
            m = re.search(r"\\qquad \((\d+)\)\$\$$", b)
            eq = re.sub(r"\s*\\qquad \(\d+\)\$\$$", "$$", b)
            out.append(div("MDPI_3.9_equation", eq))
            if m:
                out.append(div("MDPI_3.a_equation_number", f"EQNUM({m.group(1)})"))
            i += 1; continue
        if b.startswith("**Table"):
            out.append(div("MDPI_4.1_table_caption", b)); i += 1; continue
        if b.startswith("|"):
            out.append(b + "\n"); i += 1; continue
        if re.match(r"\*[^*].*\(N = [\d,]+\)\*$", b):          # sub-caption inside Table 4
            out.append(div("MDPI_4.1_table_caption", b)); i += 1; continue
        if mode == "refs":
            for line in b.split("\n"):
                out.append(div("MDPI_8.1_references", re.sub(r"^(\d+)\. ", r"\1\\. ", line.strip())))
            i += 1; continue
        if mode == "back":
            out.append(div("MDPI_6.2_back_matter", b)); i += 1; continue
        if re.match(r"\d+\. ", b):                             # numbered contribution list
            for item in re.split(r"\n(?=\d+\. )", b):
                out.append(div("MDPI_3.7_itemize", re.sub(r"^(\d+)\. ", r"\1\\. ", item.strip().replace("\n", " "))))
            i += 1; continue
        out.append(div("MDPI_3.1_text", b)); i += 1
    return "\n".join(out)


def fix_xml(path):
    x = path.read_text(encoding="utf-8")
    # literal numbers are kept in references and the contribution list: switch off the styles' automatic numbering
    for sid in ("MDPI81references", "MDPI37itemize"):
        ind = '<w:ind w:left="2891" w:hanging="283" />' if sid == "MDPI37itemize" else '<w:ind w:left="340" w:hanging="340" />'
        x = x.replace(f'<w:pStyle w:val="{sid}" />', f'<w:pStyle w:val="{sid}" /><w:numPr><w:ilvl w:val="0" /><w:numId w:val="0" /></w:numPr>{ind}')
    # tables: three-line style, full text-block width, table-body paragraph style
    def table(m):
        t = m.group(0)
        ncols = t.count("<w:gridCol")
        wide = ncols >= 4
        width, ind = (10466, 0) if wide else (7857, 2608)
        tblpr = (f'<w:tblPr><w:bidiVisual w:val="0" /><w:tblW w:w="{width}" w:type="dxa" /><w:tblInd w:w="{ind}" w:type="dxa" />'
                 '<w:tblBorders><w:top w:val="single" w:sz="8" w:space="0" w:color="000000" />'
                 '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000" /></w:tblBorders>'
                 '<w:tblLayout w:type="autofit" /><w:tblCellMar><w:left w:w="57" w:type="dxa" /><w:right w:w="57" w:type="dxa" /></w:tblCellMar></w:tblPr>')
        t = re.sub(r"<w:tblPr>.*?</w:tblPr>", tblpr, t, count=1, flags=re.S)
        t = re.sub(r'<w:pStyle w:val="Compact" />', '<w:pStyle w:val="MDPI42tablebody" />', t)
        # header row: bottom rule and bold text
        def hdr(r):
            row = r.group(0)
            row = row.replace("<w:tcPr />", '<w:tcPr><w:tcBorders><w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000" /></w:tcBorders></w:tcPr>')
            row = re.sub(r"<w:r>(?!<w:rPr>)", "<w:r><w:rPr><w:b /></w:rPr>", row)
            row = row.replace("<w:r><w:rPr>", "<w:r><w:rPr><w:b />").replace("<w:b /><w:b />", "<w:b />")
            return row
        t = re.sub(r"<w:tr><w:trPr><w:tblHeader[^>]*/></w:trPr>.*?</w:tr>", hdr, t, count=1, flags=re.S)
        return t
    x = re.sub(r"<w:tbl>.*?</w:tbl>", table, x, flags=re.S)
    x = re.sub(r"<w:bidi\s*/>", "", x)
    # numbered display equations: MDPI layout, equation in a wide cell and the number right-aligned in a narrow one
    def eqtable(m):
        eq, num = m.group(1), m.group(2)
        cell = lambda w, p: (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa" /><w:vAlign w:val="center" /></w:tcPr>{p}</w:tc>')
        nump = (f'<w:p><w:pPr><w:pStyle w:val="MDPI3aequationnumber" /><w:jc w:val="right" /></w:pPr>'
                f'<w:r><w:t>({num})</w:t></w:r></w:p>')
        return ('<w:tbl><w:tblPr><w:bidiVisual w:val="0" /><w:tblW w:w="7857" w:type="dxa" /><w:tblInd w:w="2608" w:type="dxa" />'
                '<w:tblBorders><w:top w:val="nil" /><w:left w:val="nil" /><w:bottom w:val="nil" /><w:right w:val="nil" />'
                '<w:insideH w:val="nil" /><w:insideV w:val="nil" /></w:tblBorders><w:tblLayout w:type="fixed" /></w:tblPr>'
                '<w:tblGrid><w:gridCol w:w="7140" /><w:gridCol w:w="717" /></w:tblGrid><w:tr>'
                + cell(7140, eq) + cell(717, nump) + '</w:tr></w:tbl>')
    x = re.sub(r'(<w:p><w:pPr><w:pStyle w:val="MDPI39equation" /></w:pPr>(?:(?!</w:p>).)*</w:p>)'
               r'<w:p><w:pPr><w:pStyle w:val="MDPI3aequationnumber" /></w:pPr><w:r><w:t xml:space="preserve">EQNUM\((\d+)\)</w:t></w:r></w:p>',
               eqtable, x, flags=re.S)
    assert "EQNUM" not in x, "equation number marker left"
    # schema order inside m:dPr is begChr, sepChr, endChr
    x = re.sub(r'(<m:begChr [^>]*/>)(<m:endChr [^>]*/>)(<m:sepChr [^>]*/>)', r'\1\3\2', x)
    path.write_text(x, encoding="utf-8")


def fix_styles(path):
    s = path.read_text(encoding="utf-8")
    m = re.search(r'(<w:style [^>]*w:styleId="MDPI42tablebody".*?</w:style>)', s, re.S)
    st = m.group(1).replace('<w:sz w:val="20"/>', '<w:sz w:val="16"/>').replace('<w:szCs w:val="20"/>', '<w:szCs w:val="16"/>')
    path.write_text(s.replace(m.group(1), st), encoding="utf-8")


def main():
    md = (MS / "manuscript.md").read_text(encoding="utf-8")
    (HERE / "manuscript_mdpi.md").write_text(convert(md), encoding="utf-8")
    for f in MS.glob("fig*.png"):
        shutil.copy(f, HERE / f.name)
    raw = HERE / "raw.docx"
    subprocess.run(["pandoc", "manuscript_mdpi.md", "--reference-doc=tpl.docx", "-o", str(raw)], cwd=HERE, check=True)
    tmp = HERE / "unz"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(raw) as z:
        z.extractall(tmp)
    fix_xml(tmp / "word" / "document.xml")
    fix_styles(tmp / "word" / "styles.xml")
    # drop media files that no relationship part references (template leftovers)
    used = set()
    for rel in tmp.rglob("*.rels"):
        used |= set(re.findall(r'Target="media/([^"]+)"', rel.read_text(encoding="utf-8")))
    for f in (tmp / "word" / "media").glob("*"):
        if f.name not in used:
            f.unlink()
    out = HERE / "Paper5_Algorithms_MDPI.docx"
    out.unlink(missing_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(tmp.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(tmp).as_posix())
    print(out)


if __name__ == "__main__":
    main()
