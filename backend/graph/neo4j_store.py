from __future__ import annotations

import logging
import re
from collections import Counter
from itertools import combinations
from typing import Any

logger = logging.getLogger(__name__)

_CHAPTER_PREFIX_RE = re.compile(r"^\s*(chapter|ch\.?|part|section)\s*\d+[:.\-\s]*", re.IGNORECASE)
_EN_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")
_EN_TITLE_PHRASE_RE = re.compile(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z0-9]+){0,3})\b")
_ZH_PHRASE_RE = re.compile(r"[\u4e00-\u9fff]{2,12}")
_ZH_QUOTED_RE = re.compile(r"[《“\"']([^》”\"']{2,24})[》”\"']")
_SYSTEM_TERM_RE = re.compile(r"系统\s*([一二三四五六七八九十0-9]+)")
_MIN_RELATED_WEIGHT = 2

_EN_STOP_TERMS = {
    "the", "this", "that", "these", "those", "chapter", "part", "section",
    "copyright", "contents", "gutenberg", "ebook", "license",
}
_ZH_STOP_TERMS = {
    "这个", "那个", "这里", "那里", "什么", "为何", "为什么", "如何", "怎么",
    "哪些", "哪个", "作者", "书里", "书中", "内容", "部分", "问题", "区别", "关系",
    "意思", "解释", "概念", "章节", "目录", "前言", "导言", "版权", "声明",
}
_NOISE_TITLE_PATTERNS = (
    re.compile(r"gutenberg|project gutenberg|copyright|all rights reserved", re.IGNORECASE),
    re.compile(r"table of contents|contents|index|preface|introduction", re.IGNORECASE),
    re.compile(r"目录|版权|声明|前言|致谢|附录|参考文献"),
)


class Neo4jStore:
    """
    Neo4j 元数据写入封装。

    - 未配置连接信息时自动禁用；
    - 依赖缺失或运行时报错时降级，不影响主流程。
    """

    def __init__(self) -> None:
        from config import get_settings

        settings = get_settings()
        self._database = settings.neo4j_database
        self._enabled = False
        self._driver = None

        uri = settings.neo4j_uri.strip()
        user = settings.neo4j_user.strip()
        password = settings.neo4j_password.strip()
        if not uri or not user or not password:
            logger.info("Neo4j 未配置（NEO4J_URI/USER/PASSWORD），已跳过图数据库写入")
            return

        try:
            from neo4j import GraphDatabase
        except ImportError:
            logger.warning("neo4j 驱动未安装，已跳过图数据库写入（pip install neo4j）")
            return

        try:
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            self._driver.verify_connectivity()
            self._enabled = True
            logger.info("Neo4j 连接成功（uri=%s, database=%s）", uri, self._database)
        except Exception as exc:
            logger.warning("Neo4j 初始化失败（%s），已跳过图数据库写入", exc)
            self._driver = None
            self._enabled = False

    def upsert_book(
        self,
        *,
        book_id: str,
        title: str,
        author: str,
        source: str,
        total_chunks: int,
        cover_path: str = "",
        status: str = "unread",
        progress: float = 0.0,
    ) -> None:
        """在 Neo4j 中创建/更新 Book 节点和 Author 关系。"""
        if not self._enabled or self._driver is None:
            return

        book_payload: dict[str, Any] = {
            "book_id": book_id,
            "title": title,
            "author": author,
            "source": source,
            "total_chunks": total_chunks,
            "cover_path": cover_path,
            "status": status,
            "progress": progress,
        }
        query = """
        MERGE (b:Book {book_id: $book_id})
        SET b.title = $title,
            b.author = $author,
            b.source = $source,
            b.total_chunks = $total_chunks,
            b.cover_path = $cover_path,
            b.status = $status,
            b.progress = $progress,
            b.updated_at = datetime()
        RETURN b.book_id AS book_id
        """
        try:
            with self._driver.session(database=self._database) as session:
                session.run(query, book_payload).consume()
                if author.strip():
                    session.run(
                        """
                        MATCH (b:Book {book_id: $book_id})
                        MERGE (a:Author {name: $author})
                        MERGE (a)-[:WROTE]->(b)
                        """,
                        {"book_id": book_id, "author": author},
                    ).consume()
        except Exception as exc:
            logger.warning("Neo4j upsert_book 失败（book_id=%s, err=%s）", book_id, exc)

    def upsert_book_graph(
        self,
        *,
        book_id: str,
        documents: list[Any],
        max_concepts_per_chapter: int = 12,
        max_concepts_per_chunk: int = 8,
    ) -> None:
        """
        从 chunk 文档构建知识图谱骨架：
        - Book -[:HAS_CHAPTER]-> Chapter
        - Chapter -[:HAS_CHUNK]-> Chunk
        - Chapter -[:COVERS]-> Concept
        - Chunk -[:MENTIONS]-> Concept
        - Concept -[:RELATED_TO]-> Concept
        """
        if not self._enabled or self._driver is None:
            return

        chapters, chunks = self._build_graph_payloads(
            documents,
            max_concepts_per_chapter=max_concepts_per_chapter,
            max_concepts_per_chunk=max_concepts_per_chunk,
        )
        related_pairs = self._build_related_pairs(chapters)

        try:
            with self._driver.session(database=self._database) as session:
                session.run(
                    """
                    MATCH (b:Book {book_id: $book_id})-[:HAS_CHAPTER]->(c:Chapter)
                    DETACH DELETE c
                    """,
                    {"book_id": book_id},
                ).consume()
                session.run(
                    """
                    MATCH (c1:Concept)-[r:RELATED_TO {source_book_id: $book_id}]->(c2:Concept)
                    DELETE r
                    """,
                    {"book_id": book_id},
                ).consume()

                if chapters:
                    session.run(
                        """
                        MATCH (b:Book {book_id: $book_id})
                        UNWIND $chapters AS chapter
                        MERGE (ch:Chapter {book_id: $book_id, title: chapter.title})
                        SET ch.order = chapter.order,
                            ch.updated_at = datetime()
                        MERGE (b)-[:HAS_CHAPTER]->(ch)
                        WITH ch, chapter
                        UNWIND chapter.concepts AS concept_name
                        MERGE (c:Concept {name: concept_name})
                        MERGE (ch)-[:COVERS]->(c)
                        """,
                        {"book_id": book_id, "chapters": chapters},
                    ).consume()

                if chunks:
                    session.run(
                        """
                        UNWIND $chunks AS chunk
                        MATCH (ch:Chapter {book_id: $book_id, title: chunk.chapter_title})
                        MERGE (ck:Chunk {book_id: $book_id, chunk_id: chunk.chunk_id})
                        SET ck.chunk_index = chunk.chunk_index,
                            ck.text = chunk.text,
                            ck.updated_at = datetime()
                        MERGE (ch)-[:HAS_CHUNK]->(ck)
                        WITH ck, chunk
                        UNWIND chunk.concepts AS concept_name
                        MERGE (c:Concept {name: concept_name})
                        MERGE (ck)-[:MENTIONS]->(c)
                        """,
                        {"book_id": book_id, "chunks": chunks},
                    ).consume()

                if related_pairs:
                    session.run(
                        """
                        UNWIND $pairs AS pair
                        MERGE (c1:Concept {name: pair.left})
                        MERGE (c2:Concept {name: pair.right})
                        MERGE (c1)-[r:RELATED_TO {source_book_id: $book_id}]->(c2)
                        SET r.relation_type = 'co_occurs',
                            r.weight = pair.weight,
                            r.updated_at = datetime()
                        """,
                        {"book_id": book_id, "pairs": related_pairs},
                    ).consume()
        except Exception as exc:
            logger.warning("Neo4j upsert_book_graph 失败（book_id=%s, err=%s）", book_id, exc)

    def related_books(self, *, book_id: str, limit: int = 5) -> list[dict[str, Any]]:
        """
        查询同作者的相关书籍（排除自身）。

        返回字段：
        - book_id
        - title
        - author
        - total_chunks
        """
        if not self._enabled or self._driver is None:
            return []

        query = """
        MATCH (b:Book {book_id: $book_id})<-[:WROTE]-(a:Author)-[:WROTE]->(other:Book)
        WHERE other.book_id <> $book_id
        RETURN other.book_id AS book_id,
               other.title AS title,
               other.author AS author,
               other.total_chunks AS total_chunks
        ORDER BY coalesce(other.updated_at, datetime({epochSeconds: 0})) DESC
        LIMIT $limit
        """
        try:
            with self._driver.session(database=self._database) as session:
                rows = session.run(query, {"book_id": book_id, "limit": limit})
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("Neo4j related_books 失败（book_id=%s, err=%s）", book_id, exc)
            return []

    def router_context(
        self,
        *,
        book_id: str,
        query_terms: list[str],
        recent_concepts: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        为 RouterAgent 提供图谱上下文：
        - 命中概念
        - 相关章节
        - 邻接概念
        - 与最近轮次连续性的命中
        """
        if not self._enabled or self._driver is None:
            return {
                "matched_concepts": [],
                "chapters": [],
                "related_concepts": [],
                "continuity_hits": [],
            }
        terms = [t.strip().lower() for t in (query_terms or []) if t and t.strip()]
        if not terms:
            return {
                "matched_concepts": [],
                "chapters": [],
                "related_concepts": [],
                "continuity_hits": [],
            }

        try:
            with self._driver.session(database=self._database) as session:
                if book_id:
                    matched_rows = session.run(
                        """
                        UNWIND $terms AS term
                        MATCH (:Book {book_id: $book_id})-[:HAS_CHAPTER]->(:Chapter)-[:COVERS]->(c:Concept)
                        WHERE toLower(c.name) CONTAINS term
                        RETURN DISTINCT c.name AS concept
                        LIMIT 12
                        """,
                        {"book_id": book_id, "terms": terms},
                    )
                else:
                    matched_rows = session.run(
                        """
                        UNWIND $terms AS term
                        MATCH (c:Concept)
                        WHERE toLower(c.name) CONTAINS term
                        RETURN DISTINCT c.name AS concept
                        LIMIT 12
                        """,
                        {"terms": terms},
                    )
                matched = [r["concept"] for r in matched_rows if r.get("concept")]

                if matched and book_id:
                    chapter_rows = session.run(
                        """
                        MATCH (:Book {book_id: $book_id})-[:HAS_CHAPTER]->(ch:Chapter)-[:COVERS]->(c:Concept)
                        WHERE c.name IN $matched
                        RETURN ch.title AS chapter, collect(DISTINCT c.name)[0..6] AS concepts
                        ORDER BY ch.order ASC
                        LIMIT 6
                        """,
                        {"book_id": book_id, "matched": matched},
                    )
                    chapters = [dict(r) for r in chapter_rows]
                elif matched:
                    chapter_rows = session.run(
                        """
                        MATCH (ch:Chapter)-[:COVERS]->(c:Concept)
                        WHERE c.name IN $matched
                        RETURN ch.title AS chapter, collect(DISTINCT c.name)[0..6] AS concepts
                        LIMIT 6
                        """,
                        {"matched": matched},
                    )
                    chapters = [dict(r) for r in chapter_rows]
                else:
                    chapters = []

                if matched:
                    related_rows = session.run(
                        """
                        MATCH (c:Concept)-[:RELATED_TO]->(rc:Concept)
                        WHERE c.name IN $matched
                        RETURN DISTINCT rc.name AS concept
                        LIMIT 12
                        """,
                        {"matched": matched},
                    )
                    related = [r["concept"] for r in related_rows if r.get("concept")]
                else:
                    related = []

                recent = recent_concepts or []
                recent_lower = {x.lower() for x in recent if x}
                continuity_hits = [
                    c for c in matched + related
                    if c and c.lower() in recent_lower
                ][:10]

                return {
                    "matched_concepts": matched,
                    "chapters": chapters,
                    "related_concepts": related,
                    "continuity_hits": continuity_hits,
                }
        except Exception as exc:
            logger.warning("Neo4j router_context 失败（book_id=%s, err=%s）", book_id, exc)
            return {
                "matched_concepts": [],
                "chapters": [],
                "related_concepts": [],
                "continuity_hits": [],
            }

    @staticmethod
    def _normalize_title(raw: str) -> str:
        title = (raw or "").strip()
        if not title:
            return ""
        title = _CHAPTER_PREFIX_RE.sub("", title).strip()
        if Neo4jStore._is_noise_text(title):
            return ""
        return title

    @staticmethod
    def _extract_concepts_from_text(text: str) -> list[str]:
        raw = (text or "").strip()
        if not raw:
            return []

        candidates: list[str] = []

        for m in _ZH_QUOTED_RE.findall(raw):
            c = " ".join(m.split()).strip()
            if c:
                candidates.append(c)

        for num in _SYSTEM_TERM_RE.findall(raw):
            candidates.append(f"系统{num}")
            if num.isdigit():
                candidates.append(f"System {num}")

        for m in _EN_TITLE_PHRASE_RE.findall(raw):
            c = " ".join(m.split()).strip()
            if c:
                candidates.append(c)
        for m in _EN_TERM_RE.findall(raw):
            c = m.strip()
            if c:
                candidates.append(c)
        for m in _ZH_PHRASE_RE.findall(raw):
            c = m.strip()
            if c:
                candidates.append(c)

        out: list[str] = []
        seen: set[str] = set()
        for concept in candidates:
            c = " ".join(concept.split()).strip()
            if not c or Neo4jStore._is_noise_text(c):
                continue
            if len(c) < 2 or len(c) > 48:
                continue
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    @staticmethod
    def _is_noise_text(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return True
        lower = t.lower()
        if lower in _EN_STOP_TERMS or t in _ZH_STOP_TERMS:
            return True
        if lower.startswith(("chapter ", "part ", "section ")):
            return True
        if t.endswith(("的吗", "吗", "呢", "呀")):
            return True
        for p in _NOISE_TITLE_PATTERNS:
            if p.search(t):
                return True
        return False

    def _build_graph_payloads(
        self,
        documents: list[Any],
        *,
        max_concepts_per_chapter: int,
        max_concepts_per_chunk: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        chapter_map: dict[str, dict[str, Any]] = {}
        chunk_payloads: list[dict[str, Any]] = []
        fallback_idx = 0

        for doc in documents:
            metadata = getattr(doc, "metadata", {}) or {}
            chapter_title = self._normalize_title(str(metadata.get("chapter_title") or ""))
            if not chapter_title:
                section_title = self._normalize_title(str(metadata.get("section_title") or ""))
                chapter_title = section_title
            if not chapter_title:
                fallback_idx += 1
                chapter_title = f"Untitled Chapter {fallback_idx}"

            section_indices_raw = str(metadata.get("section_indices") or "")
            section_candidates = [
                int(x) for x in section_indices_raw.split(",") if x.strip().isdigit()
            ]
            chapter_order = min(section_candidates) if section_candidates else 10**9

            row = chapter_map.setdefault(
                chapter_title,
                {"title": chapter_title, "order": chapter_order, "counter": Counter()},
            )
            row["order"] = min(row["order"], chapter_order)

            section_title = self._normalize_title(str(metadata.get("section_title") or ""))
            if section_title and section_title != chapter_title and not self._is_noise_text(section_title):
                row["counter"][section_title] += 3

            page_content = str(getattr(doc, "page_content", "") or "")
            chunk_counter: Counter[str] = Counter()
            if section_title and not self._is_noise_text(section_title):
                chunk_counter[section_title] += 2

            for concept in self._extract_concepts_from_text(page_content):
                row["counter"][concept] += 1
                chunk_counter[concept] += 1

            chunk_id = str(metadata.get("chunk_id") or "")
            if not chunk_id:
                continue
            chunk_index = int(metadata.get("chunk_index") or 0)
            top_chunk_concepts = [
                name for name, _ in chunk_counter.most_common(max_concepts_per_chunk) if name
            ]
            chunk_payloads.append(
                {
                    "chapter_title": chapter_title,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "text": page_content[:1200],
                    "concepts": top_chunk_concepts,
                }
            )

        payload: list[dict[str, Any]] = []
        for chapter in sorted(chapter_map.values(), key=lambda x: (x["order"], x["title"])):
            top_concepts = [
                name
                for name, _ in chapter["counter"].most_common(max_concepts_per_chapter)
                if name and not self._is_noise_text(name)
            ]
            payload.append(
                {
                    "title": chapter["title"],
                    "order": int(chapter["order"]) if chapter["order"] != 10**9 else 10**9,
                    "concepts": top_concepts,
                }
            )
        return payload, chunk_payloads

    @staticmethod
    def _build_related_pairs(chapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        related_counter: Counter[tuple[str, str]] = Counter()
        for chapter in chapters:
            concepts = chapter.get("concepts", [])[:6]
            cleaned = sorted({c for c in concepts if c})
            for left, right in combinations(cleaned, 2):
                related_counter[(left, right)] += 1
        return [
            {"left": left, "right": right, "weight": weight}
            for (left, right), weight in related_counter.items()
            if weight >= _MIN_RELATED_WEIGHT
        ]


_neo4j_store: Neo4jStore | None = None


def get_neo4j_store() -> Neo4jStore:
    global _neo4j_store
    if _neo4j_store is None:
        _neo4j_store = Neo4jStore()
    return _neo4j_store

