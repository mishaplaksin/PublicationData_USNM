#!/usr/bin/env python3
"""Extract the full text of a .docx manuscript into readable Markdown.

Headings become ATX headings, tables become pipe tables, and list-ish styles
(Compact / Normal inside numbered blocks) become bullets. The output is meant
to be read by a human (or a model) while writing a narration script, so nothing
is truncated and the paragraph order of the source is preserved.

Usage:
    python3 extract_docx.py INPUT.docx [-o OUTPUT.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

HEADING_LEVELS = {
    "Title": 1,
    "Heading 1": 1,
    "Heading 2": 2,
    "Heading 3": 3,
    "Heading 4": 4,
    "Heading 5": 5,
}

BULLET_STYLES = {"Compact", "List Paragraph", "List Bullet", "List Number"}


def iter_block_items(parent: DocxDocument):
    """Yield Paragraph and Table objects in document order."""
    body = parent.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def render_table(table: Table) -> str:
    rows = []
    for row in table.rows:
        cells = [" ".join(c.text.split()) for c in row.cells]
        rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |",
           "| " + " | ".join(["---"] * width) + " |"]
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def extract(path: Path) -> str:
    doc = Document(str(path))
    lines: list[str] = []
    for block in iter_block_items(doc):
        if isinstance(block, Table):
            rendered = render_table(block)
            if rendered:
                lines.extend(["", rendered, ""])
            continue

        text = block.text.strip()
        if not text:
            continue
        style = block.style.name if block.style is not None else "Normal"

        if style in HEADING_LEVELS:
            lines.extend(["", "#" * HEADING_LEVELS[style] + " " + text, ""])
        elif style in BULLET_STYLES:
            lines.append(f"- {text}")
        else:
            lines.extend(["", text])

    # Collapse runs of blank lines.
    cleaned: list[str] = []
    for line in lines:
        if not line.strip() and cleaned and not cleaned[-1].strip():
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip() + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="source .docx file")
    ap.add_argument("-o", "--output", type=Path, help="destination .md (default: stdout)")
    args = ap.parse_args(argv)

    if not args.input.is_file():
        print(f"error: no such file: {args.input}", file=sys.stderr)
        return 1

    markdown = extract(args.input)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        words = len(markdown.split())
        print(f"wrote {args.output} ({words:,} words)")
    else:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
