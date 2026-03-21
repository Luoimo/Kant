# _*_ coding:utf-8 _*_
from __future__ import annotations

import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """
    中文用 jieba 分词；未安装 jieba 时退回字符级 tokenize。
    jieba 会在首次调用时自动静默加载词典。
    """
    try:
        import jieba
        jieba.setLogLevel(logging.WARNING)
        return list(jieba.cut(text))
    except ImportError:
        logger.warning("jieba 未安装，退回字符级 tokenize（pip install jieba）")
        return list(text)


class BM25Retriever:
    """
    基于 BM25Okapi 的关键词检索。

    :param documents: 用于构建倒排索引的文档列表（通常来自 ChromaStore.get_all_documents）
    """

    def __init__(self, documents: list[Document]) -> None:
        from rank_bm25 import BM25Okapi

        self._docs = documents
        corpus = [_tokenize(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(corpus)
        logger.debug("BM25 索引构建完成，共 %d 篇文档", len(documents))

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """返回 top-k (Document, BM25 score) 对，按分数降序。"""
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._docs[i], float(scores[i])) for i in top_idx]
