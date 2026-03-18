"""
Extract structured text from a .docx file.
Usage: python extract_docx.py <file_path>
"""

import sys
import os


def extract(path: str) -> str:
    try:
        from docx import Document
        from docx.oxml.ns import qn
    except ImportError:
        return "ERROR: python-docx not installed. Run: pip install python-docx"

    if not os.path.exists(path):
        return f"ERROR: File not found: {path}"
    if not path.lower().endswith(".docx"):
        return f"ERROR: Not a .docx file: {path}"

    doc = Document(path)
    lines = []

    # --- Core properties ---
    props = doc.core_properties
    meta = []
    if props.title:
        meta.append(f"Title: {props.title}")
    if props.author:
        meta.append(f"Author: {props.author}")
    if props.created:
        meta.append(f"Created: {props.created}")
    if meta:
        lines.append("=== Document Metadata ===")
        lines.extend(meta)
        lines.append("")

    # --- Body: paragraphs & tables in document order ---
    lines.append("=== Document Content ===")

    for block in doc.element.body:
        tag = block.tag.split("}")[-1] if "}" in block.tag else block.tag

        if tag == "p":
            # Paragraph
            from docx.text.paragraph import Paragraph
            para = Paragraph(block, doc)
            text = para.text.strip()
            if not text:
                continue
            style = para.style.name if para.style else ""
            if style.startswith("Heading"):
                level = style.replace("Heading", "").strip()
                prefix = "#" * int(level) if level.isdigit() else "##"
                lines.append(f"\n{prefix} {text}")
            else:
                lines.append(text)

        elif tag == "tbl":
            # Table
            from docx.table import Table
            table = Table(block, doc)
            lines.append("\n[TABLE]")
            for row in table.rows:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                lines.append(" | ".join(cells))
            lines.append("[/TABLE]")

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_docx.py <file_path>")
        sys.exit(1)
    print(extract(sys.argv[1]))
