from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import sys

from langchain_core.documents import Document

from backend.config import get_settings
from backend.llm.openai_client import get_llm
from backend.rag.chroma.chroma_store import ChromaStore
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

    def run(self, *, query: str, memory_context: str = "") -> RecommendationResult:
        # 用偏好 query 检索相关 chunks，提取书名/作者等元数据
        docs = self.store.similarity_search(query, k=self.k)
        citations = build_citations(docs)

        print(
            f"[RecommendationAgent] query={query!r}, hits={len(docs)}",
            file=sys.stdout,
        )

        if not docs:
            return RecommendationResult(
                answer="本地书库暂时没有找到符合您偏好的书籍。请先将更多 PDF 入库，或换一个描述方向。",
                citations=[],
                retrieved_docs=[],
            )

        answer = self._generate_recommendations(query, docs, memory_context=memory_context)
        return RecommendationResult(answer=answer, citations=citations, retrieved_docs=docs)

    def _generate_recommendations(self, query: str, docs: list[Document], memory_context: str = "") -> str:
        # 从元数据中提取书名、作者，去重
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

        user_prompt = f"""用户的阅读偏好/推荐需求：
{query}

书库中检索到的相关书籍（去重后）：
{books_list}

详细片段（供参考内容特点）：
{sep.join(evidence_blocks)}

请根据以上内容，为用户推荐最符合其需求的书籍。
只推荐上面列出的书库中实际存在的书籍，每本书给出推荐理由、难度和阅读建议。
如果书库中没有足够相关的书籍，请如实说明并给出有限的推荐。"""

        system = RECOMMEND_SYSTEM_PROMPT
        if memory_context:
            system += "\n\n【用户历史阅读记录（仅供参考）】\n" + memory_context

        msg = self.llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ]
        )
        return getattr(msg, "content", str(msg))


def recommend_node(state: dict[str, Any], *, agent: RecommendationAgent) -> dict[str, Any]:
    """节点函数：读取 recommend_query，写回 answer/citations/retrieved_docs_count。"""
    query: str = state.get("recommend_query", "") or state.get("user_input", "")

    memory_context: str = state.get("memory_context", "") or ""
    result = agent.run(query=query, memory_context=memory_context)
    return {
        "answer": result.answer,
        "citations": result.citations,
        "retrieved_docs_count": len(result.retrieved_docs),
    }


__all__ = ["RecommendationAgent", "RecommendationResult", "recommend_node"]
