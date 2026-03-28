from __future__ import annotations
from pathlib import Path

from backend.utils.text import tokenize



class _LocalMarkdownStorage:
    """Base: stores content as .md files under a root directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _save(self, content: str, file_id: str) -> str:
        path = self.root / f"{file_id}.md"
        path.write_text(content, encoding="utf-8")
        return str(path)

    def load(self, storage_path: str) -> str:
        return Path(storage_path).read_text(encoding="utf-8")

    def list(self, prefix: str = "") -> list[str]:
        return sorted(str(p) for p in self.root.glob(f"{prefix}*.md"))

    def delete(self, storage_path: str) -> None:
        Path(storage_path).unlink(missing_ok=True)

    def update(self, storage_path: str, content: str) -> None:
        Path(storage_path).write_text(content, encoding="utf-8")

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, str]]:
        """全文搜索本地 .md 文件，返回 (storage_path, snippet) 列表，按相关度降序。"""
        tokens = tokenize(query)
        terms = [t for t in tokens if len(t.strip()) > 1]
        if not terms:
            return []

        scored: list[tuple[str, str, int]] = []
        for path in sorted(self.root.glob("*.md")):
            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                continue
            content_lower = content.lower()
            score = sum(content_lower.count(t.lower()) for t in terms)
            if score == 0:
                continue
            # 找第一个匹配词的位置，截取上下文片段
            snippet_start = 0
            for t in terms:
                pos = content_lower.find(t.lower())
                if pos >= 0:
                    snippet_start = max(0, pos - 40)
                    break
            snippet = content[snippet_start: snippet_start + 160].replace("\n", " ").strip()
            scored.append((str(path), snippet, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        return [(p, s) for p, s, _ in scored[:top_k]]


class LocalNoteStorage(_LocalMarkdownStorage):
    """Stores notes as .md files under a configurable root directory."""

    def save(self, content: str, note_id: str) -> str:
        return self._save(content, note_id)


__all__ = ["LocalNoteStorage"]
