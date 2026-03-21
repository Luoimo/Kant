# _*_ coding:utf-8 _*_
from __future__ import annotations

import logging
import re

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class LLMReranker:
    """
    使用 LLM 对候选文档进行相关性打分并重排。

    无需额外依赖，直接复用项目已有的 OpenAI 客户端。
    适合哲学/社科文本：LLM 能理解语义而非仅关键词匹配。
    """

    def __init__(self, llm=None) -> None:
        from backend.llm.openai_client import get_llm
        self._llm = llm or get_llm(temperature=0.0)

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        if len(docs) <= top_k:
            return docs

        numbered = "\n\n".join(
            f"[{i + 1}] {doc.page_content[:500]}" for i, doc in enumerate(docs)
        )
        prompt = (
            f"你是一位哲学文献相关性评估专家。\n"
            f"问题：{query}\n\n"
            f"请为以下每个候选段落与该问题的相关性打分（0-10整数），"
            f"只输出如下格式，每行一个：\n"
            f"1: 分数\n2: 分数\n...\n\n"
            f"候选段落：\n{numbered}"
        )
        try:
            msg = self._llm.invoke([{"role": "user", "content": prompt}])
            content = getattr(msg, "content", str(msg))
            scores = _parse_scores(content, len(docs))
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in ranked[:top_k]]
        except Exception as exc:
            logger.warning("LLMReranker 失败（%s），按原顺序截断", exc)
            return docs[:top_k]


class CrossEncoderReranker:
    """
    使用本地 sentence-transformers CrossEncoder 重排。

    默认模型 ``BAAI/bge-reranker-base`` 对中英文均有良好支持。
    首次使用时会自动下载模型（约 400 MB）。

    安装：pip install sentence-transformers
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_name)
            logger.info("CrossEncoderReranker 已加载：%s", model_name)
        except ImportError as exc:
            raise ImportError(
                "请先安装 sentence-transformers：pip install sentence-transformers"
            ) from exc

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        if len(docs) <= top_k:
            return docs
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]


def _parse_scores(content: str, n: int) -> list[float]:
    scores = [0.0] * n
    for m in re.finditer(r"(\d+)\s*[:：]\s*([0-9]+(?:\.[0-9]+)?)", content):
        idx = int(m.group(1)) - 1
        if 0 <= idx < n:
            scores[idx] = float(m.group(2))
    return scores
