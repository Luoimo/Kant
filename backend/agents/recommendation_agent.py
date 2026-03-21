from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal
import sys

logger = logging.getLogger(__name__)

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage

from backend.config import get_settings
from backend.llm.openai_client import get_llm
from backend.rag.chroma.chroma_store import ChromaStore
from backend.xai.citation import Citation

RECOMMEND_SYSTEM_PROMPT = """你是"深度阅读顾问（RecommendationAgent）"。

你的职责是基于自身知识，为用户推荐值得精读的书籍——不限于用户当前书库，从整个出版物世界中推荐。

输出格式（Markdown）：
- 每本书用 ### 书名（作者）开头
- 包含：推荐理由、难度（⭐~⭐⭐⭐⭐⭐）、适合人群、阅读建议
- 若该书已在用户书库中，书名后标注「✅ 已在库」
- 若该书不在书库中，书名后标注「📥 可上传精读」

重要规则：
1. 只推荐确实存在的出版物，不要编造书名或作者。
2. 推荐要有深度，说明为什么值得读、与用户当前阅读或兴趣有何关联。
3. 推荐书目不应局限于用户已有的书，鼓励探索用户尚未接触的作品。
"""

RECOMMEND_TYPE_HINTS: dict[str, str] = {
    "discover": "广泛推荐用户可能未曾读过的小众或经典好书。",
    "similar": "以用户当前正在读/提到的书为锚点，推荐风格或主题相似的书。",
    "next": "基于用户已读书目的难度和主题，推荐下一步适合读的书（难度递进或主题延伸）。",
    "theme": "围绕用户指定的主题/关键词，推荐最相关的书。",
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
    ) -> None:
        if store is None:
            store = ChromaStore()

        self.store = store
        self.llm = llm or get_llm(temperature=0.7)

    # ------------------------------------------------------------------
    # 获取当前库中已有书目（仅用于标注）
    # ------------------------------------------------------------------

    def _get_library_books(self) -> list[dict[str, str]]:
        """返回库中已有书目列表，每项含 book_title / author。"""
        try:
            return self.store.list_book_titles()
        except Exception as exc:
            logger.warning("获取书库列表失败：%s", exc)
            return []

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        query: str,
        current_book: str = "",
        memory_context: str = "",
        recommend_messages: list[AnyMessage] | None = None,
        recommend_type: Literal["discover", "similar", "next", "theme"] = "discover",
    ) -> RecommendationResult:
        library_books = self._get_library_books()

        print(
            f"[RecommendationAgent] query={query!r}, type={recommend_type}, "
            f"library_size={len(library_books)}",
            file=sys.stdout,
        )

        answer = self._generate_recommendations(
            query,
            library_books=library_books,
            current_book=current_book,
            memory_context=memory_context,
            recommend_messages=recommend_messages,
            recommend_type=recommend_type,
        )
        return RecommendationResult(answer=answer, citations=[], retrieved_docs=[])

    # ------------------------------------------------------------------

    def _extract_previous_titles(self, recommend_messages: list[AnyMessage] | None) -> list[str]:
        """从历史消息中提取已推荐过的书名，用于去重。"""
        if not recommend_messages:
            return []
        import re
        titles: list[str] = []
        for m in recommend_messages:
            if getattr(m, "type", "") == "ai":
                content = getattr(m, "content", "") or ""
                titles += re.findall(r"###\s+(.+)", content)
                titles += re.findall(r"《(.+?)》", content)
        return list(dict.fromkeys(titles))

    def _generate_recommendations(
        self,
        query: str,
        library_books: list[dict[str, str]],
        current_book: str = "",
        memory_context: str = "",
        recommend_messages: list[AnyMessage] | None = None,
        recommend_type: Literal["discover", "similar", "next", "theme"] = "discover",
    ) -> str:
        type_hint = RECOMMEND_TYPE_HINTS.get(recommend_type, "")

        library_note = (
            "【用户书库中已有以下书籍】：\n"
            + "\n".join(
                f"- 《{b['book_title']}》 / {b['author']}" for b in library_books
            )
            if library_books
            else "【用户书库目前为空】"
        )

        current_note = f"【用户当前正在读】：{current_book}\n\n" if current_book else ""

        prev_titles = self._extract_previous_titles(recommend_messages)
        exclusion_note = (
            "\n\n【已推荐过的书（请不要重复推荐）】：\n"
            + "\n".join(f"- {t}" for t in prev_titles)
            if prev_titles else ""
        )

        user_prompt = (
            f"{current_note}"
            f"用户的阅读偏好/推荐需求：\n{query}\n\n"
            f"推荐策略：{type_hint}\n\n"
            f"{library_note}"
            f"{exclusion_note}\n\n"
            "请基于你的知识，为用户推荐 4~6 本值得精读的书。"
            "对已在书库中的书标注「✅ 已在库」，其余标注「📥 可上传精读」。"
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
    current_book: str = state.get("book_source", "") or ""

    result = agent.run(
        query=query,
        current_book=current_book,
        memory_context=memory_context,
        recommend_messages=recommend_messages,
        recommend_type=recommend_type,
    )
    content = result.answer
    existing_ctx = state.get("compound_context") or ""
    new_ctx = (existing_ctx + f"\n\n[推荐结果]\n{content[:1500]}").strip()

    return {
        "answer": content,
        "citations": result.citations,
        "retrieved_docs_count": 0,
        "recommend_messages": [AIMessage(content=content)],
        "compound_context": new_ctx,
    }


__all__ = ["RecommendationAgent", "RecommendationResult", "recommend_node"]
