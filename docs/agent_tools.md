先用一句话概括你的目标：  
这 5 个 agent 最好都尽量“薄”，真正干活的是一组**可复用的工具函数 / LangChain tools**，以后你想接 MCP 也只是把这些工具外包出去。

下面按「1 agent = 一组推荐工具」来说。

---

### 1. 总控精读智能体（Supervisor Agent）

**核心职责**：意图识别 + 路由 + 状态编排 + 安全。

**推荐工具（内部工具 / LangChain Tools）**：

- **`classify_intent_tool`**  
  - 输入：`user_input` + 当前 `ReadingState`  
  - 输出：`intent ∈ {recommend, deepread, notes, plan, qa}`  
  - 实现：一个 structured LLM 调用。

- **`reading_state_store` / `reading_state_get`**  
  - 读/写 `ReadingState`（当前书、进度、偏好标签、最近问题等）。  
  - 所有子 agent 共用这一个工具。

- **`safety_guard_tool`**  
  - 对每次请求做 prompt 注入 / 敏感词 / 越权检查（是否超出“小众书精读”范围）。  
  - 输出：`allowed: bool, reason: str, sanitized_input: str`。

- **`log_event_tool`**  
  - 写 JSONL 事件日志（谁在什么时候调用了哪个 agent、query 内容、命中几条证据等）。  
  - 方便后面做简单的 MLSecOps/统计。

> 这些都可以先写成普通 Python 函数，后续要做 MCP，只需要把它们“包装”为 MCP 工具即可。

---

### 2. 小众书推荐智能体（Recommendation Agent）

**核心职责**：在现有本地库中，挑出“符合偏好的小众书 + 推荐理由”。

**推荐工具**：

- **`library_index_tool`**  
  - 基于 `ChromaStore.list_sources()` + PDF metadata（title/author），构建一本书级别的索引。  
  - 输出：可供排序/过滤的 book 列表（领域、作者、页数、入库时间等）。

- **`preference_profile_tool`**  
  - 从用户最近的问题、已读书/笔记中抽取“偏好标签”（如：存在主义、德语原著、难度 等级）。  
  - 可用一次 LLM 调用归纳成 Profile。

- **`book_similarity_search_tool`**（书级别）  
  - 把书的简介/前言向量化（可用同一 Embedding），做一个**书向量索引**（不是 chunk 级别）。  
  - 输入：偏好描述 / 目标；输出：相似书列表 + 分数。

- **`recommend_explain_tool`**  
  - 纯 LLM：根据选中的几本书及其 metadata，生成“推荐理由 + 适合谁 + 搭配阅读顺序”。

---

### 3. 内容精读智能体（DeepRead Agent）

**核心职责**：章节解析 / 知识点提炼 / 带引用的回答。

**推荐工具**（部分你已经有了）：

- **`retriever_tool`**  
  - 其实就是你现在的 `ChromaStore.as_retriever` / `similarity_search`。  
  - 支持 `filter={"source": book_source}`、`k`、`page_range` 之类。

- **`citation_builder_tool`**（已在 `backend/xai/citation.py` 开头实现）  
  - 从 `List[Document]` 构建带 `source/pdf_title/pdf_author/page_numbers/chunk_id/snippet` 的 `Citation` 列表。  
  - DeepRead/QA/Notes 都可以复用。

- **`chunk_context_tool`**  
  - 给定某个命中的 chunk（`chunk_id`），向前/后再拿若干 chunk 组成更大的上下文窗口（避免只看单块）。  
  - 适合作为 “二次精读” 的工具。

- **`answer_with_evidence_tool`**  
  - 一个严格模板化的 RAG Chain：输入 query + docs，输出 `{answer, reasoning, citations_used}`，  
  - 硬性要求每个关键结论引用至少 1 条 `Citation`，否则标记“不足以回答”。

- **`consistency_check_tool`（可选增强）**  
  - 简单版：再调一次 LLM，问“这些结论是否都能在给定证据中找到支持？如果不行指出哪条可疑”。  
  - 结果可以被 Supervisor 用来决定是否降级回答/提示“不确定”。

---

### 4. 笔记整理智能体（Note Agent）

**核心职责**：把原始笔记/想法整理成结构化笔记、带引用的“文字版思维导图”。

**推荐工具**：

- **`note_store_tool`**  
  - CRUD：`create_note / update_note / list_notes / get_note`。  
  - Note 结构建议：`{id, book_source, page_numbers, chunk_ids[], raw_text, summary, tags[]}`。

- **`note_summarize_tool`**  
  - 输入：原始笔记 + 相关 citations；  
  - 输出：要点列表（高亮关键句，避免长段落）。

- **`note_mindmap_tool`**  
  - 输出为 “缩进 Markdown 树”：  
    ```md
    - 主题 A
      - 子点 A1
      - 子点 A2
    - 主题 B
    ```  
  - 不用可视化，只管文本结构。

- **`note_backlink_tool`**  
  - 根据 note 的 `chunk_ids/page_numbers`，建立笔记 ↔ 书中位置 的反向索引；  
  - 方便以后从某个引用跳回所有相关笔记。

- **`note_search_tool`**  
  - 对笔记本身做一个小 RAG：  
  - 输入：问题；输出：最相关的笔记片段 + 关联 book/page。

---

### 5. 书单规划智能体（Reading Plan Agent）

**核心职责**：根据目标、时间与偏好，把若干小众书排成一条“可执行的路线图”。

**推荐工具**：

- **`goal_parse_tool`**  
  - 把自然语言目标（“社科入门”，“存在主义系统精读”）解析成：  
    - 领域/topic 列表  
    - 难度目标（入门/进阶/专业）  
    - 时间限制（每周几小时 / 计划时长）

- **`book_difficulty_estimate_tool`**  
  - 基于书的元数据 + chunk 统计（词长、句子长度、专有名词密度），给每本书打一个大致的难度分。  
  - 不用太准，只要能区分“轻阅读 vs 理论炸弹”。

- **`plan_scheduler_tool`**  
  - 输入：目标 + 候选书清单 + 用户时间预算；  
  - 输出：一个 JSON 计划，例如：  
    ```json
    [
      {"week": 1, "book": "A", "pages": "1-50"},
      {"week": 2, "book": "A", "pages": "51-120"},
      {"week": 3, "book": "B", "pages": "1-80"}
    ]
    ```

- **`plan_adjust_tool`**  
  - 根据实际进度（你可以从 `ReadingState` + 笔记活跃度推算）和用户反馈（太难/太水）来重排：  
  - 比如：插入过渡书、拉长某本书的时长。

---

### 6. 是否需要 MCP / 外部工具？

按你最初的约束「**纯本地开发，无 API 对接**」，现在这套更多是 **“LangChain Tool” / 内部 Python 工具函数** 就够了。

如果未来你想：

- 从外网抓“小众书元数据/评分”（豆瓣、Goodreads）；  
- 或把这个系统挂成一个 MCP Server 给别的 LLM 用；

那再考虑新建一个 MCP，比如：

- **`book-metadata-mcp`**：`search_book`, `get_book_detail`, `get_reviews` 等，  
- 但这一层可以完全建立在你现在这些“内部工具”之上。

---

如果你愿意，下一步我可以帮你把这 5 个 agent 的「工具清单」具体映射到你现有目录里应该有哪些模块/函数（比如放在哪个 `backend/...` 文件里、函数签名大致长什么样），方便你一步步实现。