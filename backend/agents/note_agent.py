# _*_ coding:utf-8 _*_
# @Time:2026/3/9
# @Author:Chloe
# @File:note_agent
# @Project:Kant

"""
NoteAgent 设计说明
==================

一、定位与意图
--------------
对应 orchestrator 意图 "notes"：用户请求「整理/总结/结构化读书笔记」。
与 DeepReadAgent 的区别：
- DeepRead：针对问题做「基于证据的一问一答」，产出 answer + citations。
- NoteAgent：针对书或已有文字做「归纳、结构化」，产出笔记（大纲/要点/思维导图式树），可带引用。

二、输入（由总控写入 state，本 Agent 只读）
------------------------------------------
- notes_query: str
  - 用户原话或结构化指令，例如：
    - "把《焦虑》第三章整理成大纲"
    - "帮我总结一下刚才读的内容"
    - "把这些零散笔记整理成结构化笔记"
- notes_book_source: str | None
  - 指定某本书（与 Chroma 的 source 一致）时做「按书检索再归纳」；为 None 时可做「纯文本整理」或结合上文。
- 可选（后续扩展）：notes_raw_text: str —— 用户粘贴的待整理原文（不经过 RAG）。

三、输出（写回 state / 供 API）
------------------------------
- answer: str
  - 主产出：结构化笔记正文（Markdown），可含标题、要点、引用页码、TODO、问题列表等。
- citations: list[Citation]
  - 若走了 RAG，与 DeepRead 相同的引用结构，便于前端展示来源。
- retrieved_docs_count: int
  - 检索到的 chunk 数量（0 表示未走 RAG 或 RAG 无结果）。
- 可选（与 SharedMemoryStore 对接时）：
  - 写入 memory：notes[]、note_index（书名/章节 -> 笔记 id），供 GET /notes、后续问答或书单参考。

四、核心功能（建议实现顺序）
---------------------------
1. 按书/章节归纳（RAG + 结构化）
   - 用 ChromaStore.similarity_search 按 notes_book_source（及可选章节）检索；
   - 将检索到的 chunks 交给 LLM，要求输出：大纲 / 要点列表 / 关键概念 + 对应页码；
   - 产出带 citations 的 Markdown 笔记。

2. 零散笔记结构化（无书 RAG）
   - 输入：用户粘贴的一段文字（或从 state 取的 notes_raw_text）；
   - LLM 输出：分级标题、要点、TODO、待解决问题列表等；
   - 不涉及检索时 citations 为空。

3. 文字版思维导图（可选）
   - 同一套 RAG 或原文输入，要求 LLM 输出「缩进树 / 大纲树」形式的 Markdown（如 - 主题 \\n   - 分支1 \\n   - 分支2），便于导出或展示。

4. 与记忆对接（可选，依赖 SharedMemoryStore）
   - 将本次产出的笔记写入 memory.notes，并更新 note_index；
   - GET /notes 时从 memory 读取，可由本 Agent 或 API 层负责。

五、与现有组件的关系
-------------------
- 与 DeepReadAgent：共用 ChromaStore、Citation 构建方式；不共用 prompt（DeepRead 强调「只答问」，Note 强调「归纳与结构」）。
- 与 Orchestrator：总控在 intent=notes 时写入 notes_query、notes_book_source，路由到 NoteAgent；NoteAgent 只读这两项，写回 answer/citations/retrieved_docs_count。
- 与 GraphDeps：与 DeepRead 一样，NoteAgent 实例通过依赖注入（GraphDeps.notes_agent）传入节点，不放入 state。

六、接口约定（与 DeepRead 对齐）
--------------------------------
- 类：NoteAgent(store: ChromaStore | None = None, llm=None, ...).
- 方法：run(*, query: str, book_source: str | None = None, raw_text: str | None = None) -> NoteResult.
- NoteResult：与 DeepReadResult 类似，至少含 answer, citations, retrieved_docs; 可选 structured_notes（dict/list）供 API 使用。
- 节点函数：notes_node(state, *, agent: NoteAgent) -> dict，只读 state["notes_query"]、state["notes_book_source"]，写回 answer/citations/retrieved_docs_count。
"""

from __future__ import annotations

# 待实现：NoteAgent、NoteResult、notes_node
# from backend.rag.chroma.chroma_store import ChromaStore
# from backend.xai.citation import Citation, build_citations

__all__ = []  # 实现后补充 "NoteAgent", "NoteResult", "notes_node"
