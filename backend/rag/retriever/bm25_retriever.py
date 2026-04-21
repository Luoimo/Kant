# _*_ coding:utf-8 _*_
from __future__ import annotations

import logging

from langchain_core.documents import Document

from utils.text import tokenize

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    基于 BM25Okapi 的关键词检索。

    :param documents: 用于构建倒排索引的文档列表（通常来自 ChromaStore.get_all_documents）
    """

    def __init__(self, documents: list[Document]) -> None:
        from rank_bm25 import BM25Okapi

        self._docs = documents
        if not documents:
            self._bm25 = None
            logger.debug("BM25 索引为空（无文档）")
            return
        corpus = [tokenize(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(corpus)
        logger.debug("BM25 索引构建完成，共 %d 篇文档", len(documents))

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """返回 top-k (Document, BM25 score) 对，按分数降序。"""
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._docs[i], float(scores[i])) for i in top_idx]
