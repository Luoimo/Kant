---
name: read-docx
description: Read and analyze a .docx Word document. Use this skill when the user wants to open, read, understand, or analyze the content of a Word document (.docx file).
argument-hint: <path/to/file.docx>
allowed-tools: Bash, Read
---

The user wants to read and analyze the following document: **$ARGUMENTS**

## Extracted Content

!`python "${CLAUDE_SKILL_DIR}/extract_docx.py" "$ARGUMENTS"`

---

Based on the extracted content above, perform a thorough analysis:

1. **Document Overview** — What is this document about? What is its purpose?
2. **Structure** — Summarize the main sections and their organization.
3. **Key Points** — Extract the most important ideas, arguments, or data.
4. **Notable Details** — Tables, lists, or specific data worth highlighting.
5. **Project Relevance** — How does this document relate to the current project context?

If the extraction step returned an ERROR, diagnose the issue and tell the user exactly how to fix it (e.g., install missing dependencies). Do not proceed with analysis if no content was extracted.
