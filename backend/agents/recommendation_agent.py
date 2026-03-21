from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import sys

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage

from backend.config import get_settings
from backend.llm.openai_client import get_llm
from backend.rag.chroma.chroma_store import ChromaStore
from backend.rag.retriever import HybridConfig, HybridRetriever
from backend.xai.citation import Citation, build_citations

sep = "\n\n"

RECOMMEND_SYSTEM_PROMPT = """你是"书籍推荐助手（RecommendationAgent）"，专注于推荐小众好书。

核心职责：
1. 根据用户的阅读偏好或主题兴趣，从书库中检索相关书籍并给出推荐。
2. 每本推荐应包含：书名、作者、推荐理由、难度评级、阅读建议。
3. 推荐要有深度，不只是泛泛而谈，要说明这本书为什么值得读、适合什么样的读者。
4. 只推荐书库中实际存在的书籍，不要凭空编造书名或作者。

输出格式（Markdown）：
- 每本书用 ### 书名 开头
- 包含：作者、推荐理由、难度（⭐~⭐⭐⭐⭐⭐）、适合人群、阅读建议
"""

RECOMMEND_TYPE_HINTS: dict[str, str] = {
    "discover": "广泛扫描书库，推荐用户可能未曾听说的小众好书。",
    "similar": "以用户当前正在读/提到的书为锚点，推荐风格或主题相似的书。",
    "next": "基于用户已读书目的难度和主题，推荐下一步适合读的书（难度递进或主题延伸）。",
    "theme": "围绕用户指定的主题/关键词，跨全书库检索并推荐最相关的书。",
}


@dataclass(frozen=True)
class RecommendationResult:
    answer: str
    citations: list[Citation]
    retrieved_docs: list[Document]


class RecommendationAgent:
    def __init__(
        self,
        *,
        store: ChromaStore | None = None,
        llm=None,
        k: int = 8,
    ) -> None:
        if store is None:
            settings = get_settings()
            store = ChromaStore(collection_name=settings.chroma_database)

        self.store = store
        self.llm = llm or get_llm(temperature=0.5)
        self.k = k
        self._catalog_cache: dict | None = None

        # HybridRetriever 一次性构建
        self._retriever = HybridRetriever(
            store=self.store,
            collection_name=self.store.collection_name,
            config=HybridConfig(fetch_k=20, final_k=k),
            llm=self.llm,
        )

    # ------------------------------------------------------------------
    # 书目总览缓存
    # ------------------------------------------------------------------

    def _get_catalog_summary(self) -> str:
        sources = self.store.list_sources()
        cache_key = hash(tuple(sorted(sources)))
        if self._catalog_cache and self._catalog_cache.get("key") == cache_key:
            return self._catalog_cache["summary"]
        summary = self._build_catalog_summary(sources[:30])
        self._catalog_cache = {"key": cache_key, "summary": summary}
        return summary

    def _build_catalog_summary(self, sources: list[str]) -> str:
        lines: list[str] = []
        for source in sources:
            try:
                docs = self.store.similarity_search(
                    "introduction overview", k=2, filter={"source": source}
                )
            except Exception:
                continue
            if docs:
                meta = docs[0].metadata or {}
                title = meta.get("book_title") or source
                author = meta.get("author") or "未知作者"
                snippet = (docs[0].page_content or "")[:100].replace("\n", " ")
                lines.append(f"《{title}》/ {author} — {snippet}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        query: str,
        memory_context: str = "",
        recommend_messages: list[AnyMessage] | None = None,
        recommend_type: Literal["discover", "similar", "next", "theme"] = "discover",
    ) -> RecommendationResult:
        # 混合检索相关 chunks
        docs = self._retriever.search(query)
        citations = build_citations(docs)

        print(
            f"[RecommendationAgent] query={query!r}, type={recommend_type}, hits={len(docs)}",
            file=sys.stdout,
        )

        if not docs:
            return RecommendationResult(
                answer="本地书库暂时没有找到符合您偏好的书籍。请先将更多 EPUB 入库，或换一个描述方向。",
                citations=[],
                retrieved_docs=[],
            )

        answer = self._generate_recommendations(
            query, docs,
            memory_context=memory_context,
            recommend_messages=recommend_messages,
            recommend_type=recommend_type,
        )
        return RecommendationResult(answer=answer, citations=citations, retrieved_docs=docs)

    # ------------------------------------------------------------------

    def _extract_previous_titles(self, recommend_messages: list[AnyMessage] | None) -> list[str]:
        """从历史消息中提取已推荐过的书名，用于去重。"""
        if not recommend_messages:
            return []
        titles: list[str] = []
        import re
        for m in recommend_messages:
            if getattr(m, "type", "") == "ai":
                # 匹配 ### 书名 或 《书名》 格式
                content = getattr(m, "content", "") or ""
                titles += re.findall(r"###\s+(.+)", content)
                titles += re.findall(r"《(.+?)》", content)
        return list(dict.fromkeys(titles))  # 去重保序

    def _generate_recommendations(
        self,
        query: str,
        docs: list[Document],
        memory_context: str = "",
        recommend_messages: list[AnyMessage] | None = None,
        recommend_type: Literal["discover", "similar", "next", "theme"] = "discover",
    ) -> str:
        # 书名去重（展示给 LLM）
        seen_titles: set[str] = set()
        book_infos: list[str] = []
        evidence_blocks: list[str] = []

        for i, d in enumerate(docs, start=1):
            meta = d.metadata or {}
            title = meta.get("book_title") or "未知书名"
            author = meta.get("author") or "未知作者"
            pages = meta.get("section_indices") or ""
            snippet = (d.page_content or "").strip()
            if title not in seen_titles:
                seen_titles.add(title)
                book_infos.append(f"- 《{title}》 / {author}")
            evidence_blocks.append(
                f"[片段{i}] 书名：{title}  作者：{author}  页码：{pages}\n{snippet}"
            )

        books_list = "\n".join(book_infos) if book_infos else "（未提取到书名）"
        catalog_summary = self._get_catalog_summary()
        type_hint = RECOMMEND_TYPE_HINTS.get(recommend_type, "")
        prev_titles = self._extract_previous_titles(recommend_messages)
        exclusion_note = (
            f"\n\n【已推荐过的书（请不要重复推荐）】：\n"
            + "\n".join(f"- {t}" for t in prev_titles)
            if prev_titles else ""
        )

        user_prompt = (
            f"用户的阅读偏好/推荐需求：\n{query}\n\n"
            f"推荐策略：{type_hint}\n\n"
            f"【全书库概览（最多 30 本）】：\n{catalog_summary}\n\n"
            f"【检索到的相关书籍（去重）】：\n{books_list}\n\n"
            f"【详细片段（供参考内容特点）】：\n{sep.join(evidence_blocks)}"
            f"{exclusion_note}\n\n"
            "请根据以上内容，为用户推荐最符合其需求的书籍。"
            "只推荐上面列出的书库中实际存在的书籍，每本书给出推荐理由、难度和阅读建议。"
        )

        system = RECOMMEND_SYSTEM_PROMPT
        if memory_context:
            system += "\n\n【用户历史阅读记录（仅供参考）】\n" + memory_context

        history = []
        if recommend_messages:
            for m in recommend_messages[-4:]:
                role = "assistant" if getattr(m, "type", "") == "ai" else "user"
                history.append({"role": role, "content": getattr(m, "content", "")})

        msg = self.llm.invoke(
            history + [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ]
        )
        return getattr(msg, "content", str(msg))


def recommend_node(state: dict[str, Any], *, agent: RecommendationAgent) -> dict[str, Any]:
    """节点函数：返回 delta dict（包含 recommend_messages, compound_context）。"""
    query: str = state.get("recommend_query", "") or state.get("user_input", "")
    memory_context: str = state.get("memory_context", "") or ""
    recommend_messages: list[AnyMessage] = state.get("recommend_messages") or []
    recommend_type = state.get("recommend_type") or "discover"

    result = agent.run(
        query=query,
        memory_context=memory_context,
        recommend_messages=recommend_messages,
        recommend_type=recommend_type,
    )
    content = result.answer
    existing_ctx = state.get("compound_context") or ""
    new_ctx = (existing_ctx + f"\n\n[推荐结果]\n{content[:500]}").strip()

    return {
        "answer": content,
        "citations": result.citations,
        "retrieved_docs_count": len(result.retrieved_docs),
        "recommend_messages": [AIMessage(content=content)],
        "compound_context": new_ctx,
    }


__all__ = ["RecommendationAgent", "RecommendationResult", "recommend_node"]
