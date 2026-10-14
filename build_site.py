#!/usr/bin/env python3
"""Build English GitHub Pages HTML from the essay markdown."""

from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

PAGE = {
    "src": SRC / "diffusion_is_matching.md",
    "out": ROOT / "index.html",
    "title": "Your Diffusion Model is Just a Distribution Matching Model",
    "lang": "en",
    "date": "6 October 2026",
}


def cite(n: str) -> str:
    return f'<a class="cite" href="#ref-{int(n)}">{n}</a>'


def format_inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(
        r"\[\[(\d+)\]\]\((https?://[^)]+)\)",
        lambda m: f'<a class="cite" href="#ref-{int(m.group(1))}" data-ext="{m.group(2)}">{m.group(1)}</a>',
        text,
    )
    text = re.sub(r"\[(\d+)\]–\[(\d+)\]", lambda m: cite(m.group(1)) + "–" + cite(m.group(2)), text)
    text = re.sub(r"\[(\d+)\]", lambda m: cite(m.group(1)), text)
    text = text.replace("</a><a class=\"cite\"", "</a> <a class=\"cite\"")
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def slugify(title: str) -> str:
    title = re.sub(r"\\[()]", "", title)
    return re.sub(r"[^\w\u4e00-\u9fff]+", "-", title.lower()).strip("-") or "section"


TOC_LABEL = {
    "what-the-papers-have-been-fitting": "Prior work",
    "what-the-loss-is-averaging": "The loss",
    "four-places-to-look": "Experiments",
    "what-this-reading-leaves-open": "What it leaves open",
    "conclusion": "Conclusion",
    "acknowledgments": "Acknowledgments",
    "references": "References",
    "cite": "Cite",
}


def parse_table(lines: list[str], start: int) -> tuple[str, int]:
    """Parse a GitHub-style markdown table starting at lines[start]."""
    rows: list[list[str]] = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        raw = lines[i].strip().strip("|")
        cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", raw)]
        # skip separator row |---|---|
        if all(re.match(r"^:?-+:?$", c or "") for c in cells):
            i += 1
            continue
        rows.append(cells)
        i += 1
    if not rows:
        return "", start + 1
    head, *body = rows
    thead = "<tr>" + "".join(f"<th>{format_inline(c)}</th>" for c in head) + "</tr>"
    tbodies = []
    for r in body:
        # pad/truncate to head width
        while len(r) < len(head):
            r.append("")
        r = r[: len(head)]
        tbodies.append("<tr>" + "".join(f"<td>{format_inline(c)}</td>" for c in r) + "</tr>")
    html_tbl = (
        '<div class="table-wrap"><table class="md-table">'
        f"<thead>{thead}</thead><tbody>{''.join(tbodies)}</tbody>"
        "</table></div>"
    )
    return html_tbl, i


def toc_html(body: str) -> str:
    items = [("top", "Introduction")]
    for sid, raw in re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', body, flags=re.S):
        label = TOC_LABEL.get(sid) or re.sub(r"<[^>]+>", "", raw)
        items.append((sid, label))
    if not any(sid == "cite" for sid, _ in items):
        items.append(("cite", "Cite"))
    lis = "\n".join(
        f'<li><a href="#{html.escape(sid)}">{html.escape(label)}</a></li>'
        for sid, label in items
    )
    return f'<nav class="toc" aria-label="Contents"><p class="toc-k">Contents</p><ol>{lis}</ol></nav>'


def _caption_html(cap: str) -> str:
    """Escape caption; emphasize Cambrian Left:/Right: panel cues."""
    t = html.escape(cap)
    t = re.sub(
        r"\b(Left|Right|Top|Bottom|Center|Middle):",
        r"<strong>\1:</strong>",
        t,
    )
    return t


def _figure_html(fig_n: int, cap: str, src: str, kind: str) -> str:
    """Cambrian / paper-style figure block."""
    cls = "fig fig-img"
    if kind == "wide":
        cls += " wide"
    elif kind == "compact":
        cls += " compact"
    elif kind in ("board", "paper"):
        # arxiv / Cambrian breakout (no card chrome)
        cls += " paper"
    # else: default card figure
    return (
        f'<figure class="{cls}">'
        f'<img src="{html.escape(src)}" alt="{html.escape(cap)}" loading="eager">'
        f'<figcaption><span class="fig-k">Figure {fig_n}.</span> '
        f"{_caption_html(cap)}</figcaption></figure>"
    )


def fig_claim() -> str:
    rows = (
        "<tr><th>Object</th><th>Looks like</th><th>What the loss demands</th></tr>"
        "<tr><td>Diffusion / score matching</td><td>Learn to denoise and generate</td>"
        "<td class=\"flat\">Fit the score of noised data; pull back to training mass</td></tr>"
        "<tr><td>Text / PE conditioning</td><td>Understand prompt semantics</td>"
        "<td class=\"flat\">Route seen condition keys to seen visual modes</td></tr>"
        "<tr><td>Generalization / composition</td><td>Paraphrase, extrapolate, recombine fields</td>"
        "<td class=\"up\">Not rewarded by the objective</td></tr>"
    )
    return f"""<figure class="fig fig-exp">
  <div class="matrix-wrap"><table class="matrix">{rows}</table></div>
</figure>"""


def md_to_html(md: str) -> str:
    lines = md.splitlines()
    chunks: list[str] = []
    i = 0
    in_code = False
    code: list[str] = []
    refs = False
    ref_items: list[str] = []
    inserted_claim = False
    fig_n = 0
    dek_done = False

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            if in_code:
                body = "\n".join(code)
                chunks.append("<pre><code>" + html.escape(body) + "</code></pre>")
                code = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code.append(line)
            i += 1
            continue
        if line.strip() in {"---", ""}:
            i += 1
            continue
        if line.startswith("# ") and not line.startswith("## "):
            i += 1
            continue
        if line.strip() == "## References":
            refs = True
            chunks.append('<h2 id="references">References</h2>')
            i += 1
            continue
        if refs:
            m = re.match(r"^(\d+)\.\s+(.*)$", line)
            if m:
                n, rest = m.group(1), m.group(2)
                rest = html.escape(rest)
                rest = re.sub(r"(https?://\S+)", r'<a href="\1">\1</a>', rest)
                rest = re.sub(r"\*(.+?)\*", r"<em>\1</em>", rest)
                ref_items.append(f'<li id="ref-{n}">{rest}</li>')
            elif line.startswith("    ") and ref_items:
                extra = html.escape(line.strip())
                extra = re.sub(r"(https?://\S+)", r'<a href="\1">\1</a>', extra)
                extra = re.sub(r"\*(.+?)\*", r"<em>\1</em>", extra)
                ref_items[-1] = ref_items[-1][:-5] + "<br>" + extra + "</li>"
            i += 1
            continue
        if line.startswith("### "):
            title = line[4:].strip()
            chunks.append(f'<h3 id="{slugify(title)}">{format_inline(title)}</h3>')
            i += 1
            continue
        if line.startswith("## "):
            title = line[3:].strip()
            chunks.append(f'<h2 id="{slugify(title)}">{format_inline(title)}</h2>')
            i += 1
            continue
        if line.strip().startswith("|") and "|" in line.strip()[1:]:
            tbl, i = parse_table(lines, i)
            if tbl:
                chunks.append(tbl)
            continue
        if line.startswith("> "):
            box = []
            while i < len(lines) and lines[i].startswith("> "):
                box.append(lines[i][2:].strip())
                i += 1
            head = box[0]
            body = " ".join(box[1:]) if len(box) > 1 else ""
            km = re.match(r"\*\*(.+?)\*\*\.?\s*(.*)", head)
            if km:
                label, rest = km.group(1), km.group(2)
                body = (rest + " " + body).strip()
                chunks.append(
                    f'<aside class="callout"><span class="k">{html.escape(label)}</span>'
                    f"{format_inline(body)}</aside>"
                )
            else:
                chunks.append(f'<aside class="callout">{format_inline(" ".join(box))}</aside>')
            continue
        if line.startswith("![") and "](figures/" in line:
            m = re.match(
                r"!\[([^\]]*)\]\((figures/[^)]+)\)(?:\s*\{(wide|compact|board|paper|pair)\})?",
                line.strip(),
            )
            if not m:
                i += 1
                continue
            fig_n += 1
            cap, src, kind = m.group(1), m.group(2), m.group(3) or ""
            if kind == "pair":
                items = [(fig_n, cap, src)]
                j = i + 1
                while j < len(lines) and len(items) < 2:
                    m2 = re.match(
                        r"!\[([^\]]*)\]\((figures/[^)]+)\)(?:\s*\{(wide|compact|board|paper|pair)\})?",
                        lines[j].strip(),
                    )
                    if not m2 or (m2.group(3) or "") != "pair":
                        break
                    fig_n += 1
                    items.append((fig_n, m2.group(1), m2.group(2)))
                    j += 1
                if len(items) == 2:
                    chunks.append(
                        '<div class="fig-pair">'
                        + _figure_html(items[0][0], items[0][1], items[0][2], "paper")
                        + _figure_html(items[1][0], items[1][1], items[1][2], "paper")
                        + "</div>"
                    )
                    i = j
                    continue
            chunks.append(_figure_html(fig_n, cap, src, kind))
            i += 1
            continue
        if line.startswith("\\[") or line.strip() == "\\[":
            block = [line]
            if line.strip() != "\\]" and not line.strip().endswith("\\]"):
                i += 1
                while i < len(lines) and lines[i].strip() != "\\]":
                    block.append(lines[i])
                    i += 1
                if i < len(lines):
                    block.append(lines[i])
            chunks.append('<div class="math-block">' + html.escape("\n".join(block)) + "</div>")
            i += 1
            continue
        para = [line]
        i += 1
        while (
            i < len(lines)
            and lines[i].strip()
            and not lines[i].startswith("#")
            and not lines[i].startswith("```")
            and not lines[i].startswith("![")
            and not lines[i].startswith("> ")
            and not lines[i].strip().startswith("|")
            and lines[i].strip() != "---"
            and not lines[i].startswith("\\[")
        ):
            para.append(lines[i])
            i += 1
        text = " ".join(p.strip() for p in para)
        if not dek_done:
            chunks.append('<p class="dek">' + format_inline(text) + "</p>")
            dek_done = True
        else:
            chunks.append("<p>" + format_inline(text) + "</p>")
        if not inserted_claim and text.startswith("Put differently"):
            chunks.append(fig_claim())
            inserted_claim = True

    if ref_items:
        chunks.append('<div class="refs"><ol>' + "\n".join(ref_items) + "</ol></div>")
    return "\n".join(chunks)


def authors_html() -> str:
    authors = (
        ("Weiyang Jin", "The University of Hong Kong"),
    )
    items = [
        f'<li><span class="name">{html.escape(n)}</span>'
        f'<span class="aff">{html.escape(a)}</span></li>'
        for n, a in authors
    ]
    return '<ul class="authors">' + "".join(items) + "</ul>"


def page_html(cfg: dict, body: str) -> str:
    bibtex = """@misc{jin2026diffusionmatching,
  author       = {Jin, Weiyang},
  title        = {Your Diffusion Model is Just a Distribution Matching Model},
  year         = {2026},
  month        = oct,
  howpublished = {Blog post},
  url          = {https://xiaoxiaoyang2003.github.io/diffusion-is-matching/},
}"""
    credit = (
        'Figure style after '
        '<a href="https://cambrian-mllm.github.io/">Cambrian</a>'
        ' and <a href="https://waynejin0918.github.io/how-far-native-ar-video/">a previous note</a>.'
    )
    return f"""<!DOCTYPE html>
<html lang="{cfg["lang"]}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(cfg["title"])}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://cdn.jsdelivr.net">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
  <link rel="stylesheet" href="css/blog.css?v=23">
</head>
<body>
  <div class="shell">
    {toc_html(body)}
    <main class="page">
    <header class="mast">
      <p class="date">{html.escape(cfg["date"])}<span class="views" data-views="hit" data-views-label="views">— views</span></p>
    </header>
    <article>
      <h1 id="top">{html.escape(cfg["title"])}</h1>
      {authors_html()}
      {body}
      <h2 id="cite">Cite</h2>
      <div class="bib">
        <button type="button" class="bib-copy" data-copy>Copy</button>
        <pre><code>{html.escape(bibtex)}</code></pre>
      </div>
    </article>
    <footer class="credit">{credit}</footer>
    </main>
  </div>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
  <script defer src="js/math.js"></script>
  <script defer src="js/ui.js"></script>
</body>
</html>
"""


def main() -> None:
    (ROOT / ".nojekyll").write_text("")
    body = md_to_html(PAGE["src"].read_text(encoding="utf-8"))
    PAGE["out"].write_text(page_html(PAGE, body), encoding="utf-8")
    print("wrote", PAGE["out"])
    zh = ROOT / "zh.html"
    if zh.exists():
        zh.unlink()
        print("removed", zh)


if __name__ == "__main__":
    main()
