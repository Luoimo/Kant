# DeepRead Agent 简化设计

**Date:** 2026-03-29
**Status:** Approved

---

## 背景与目标

当前项目存在两个问题：

1. **Agent 数量过多**：PlanEditor、RecommendAgent 与用户实际阅读行为（划词提问、有疑问时提问）脱节，使用率极低。
2. **编排层过重**：Queue/WorkerPool/Dispatcher/IntentGraph 为了解决并发和意图路由，引入了大量协调代码，但实际使用场景以单意图查询为主，收益远低于复杂度代价。

**目标：** 收敛为单 Agent + 单工具架构，通过系统 prompt 覆盖所有核心阅读场景。

---

## 核心用户痛点（设计依据）

| 痛点 | 解决方式 |
|---|---|
| A. 遇到生僻概念/术语 | 系统 prompt：概念解释模式 |
| B. 读完一章记不住脉络 | 系统 prompt + `summarize_chapter` 参数 |
| C. 想联系其他书的观点 | 系统 prompt + `search_books(book_id=None)` |
| D. 想验证自己的理解 | 系统 prompt：苏格拉底模式 |

---

## 最终架构

```
用户请求
    ↓
DeepRead Agent（单一 ReAct Agent）
    └── search_books(query, book_id, chapter)  ← 唯一工具
         ├── book_id + chapter → 章节范围检索
         ├── book_id only     → 单书语义检索
         └── book_id=None     → 跨全库检索
    ↓
回答完成后
    └── NoteAgent.process_qa()  ← auto-hook，自动沉淀 Q&A
```

**没有** Supervisor、Sub-agent、Queue、WorkerPool、Dispatcher、IntentGraph。

---

## 工具设计

### `search_books`

```python
@tool
def search_books(
    query: str,
    book_id: str | None = None,
    chapter: str | None = None,
) -> list[Document]:
    """
    搜索书库内容。

    - book_id + chapter：按章节元数据过滤检索，用于章节摘要
    - book_id only：单书语义检索，用于常规问答
    - 两者均 None：跨全库语义检索，用于跨书对比
    """
```

Agent 根据用户意图自行决定参数：

| 用户意图 | 传参 |
|---|---|
| "康德怎么看自由意志" | `book_id=当前书` |
| "总结这一章" | `book_id=当前书, chapter=当前章节` |
| "康德和黑格尔怎么看自由" | `book_id=None` |

---

## 系统 Prompt 行为模式

在原有问答原则基础上，新增四种行为模式：

**概念解释模式**（触发：`selected_text` 非空，或问"什么意思/解释"）
- 三层输出：词义 → 书内语境（引用原文）→ 哲学背景
- 调用 `search_books` 检索该词上下文（k=3）

**章节摘要模式**（触发：含"总结/这章/摘要/梳理"）
- 输出结构：核心论点 → 论证结构 → 关键术语
- 调用 `search_books(chapter=当前章节)`，限制 k=10

**跨书对比模式**（触发：含书名、"对比/比较"，或无 `book_id`）
- 自动模式：跨全库检索，返回相关段落 + 来源书名
- 对比模式：query 中含书名时，结构化呈现两书观点

**苏格拉底模式**（触发：含"我觉得/我认为/对吗/考我"）
- 验证模式：在书中找反例或支撑证据，提出追问，不直接给答案
- 测验模式：从书中抽取核心命题，出开放题

---

## 笔记处理

**删除：** `save_note` 工具、用户主动触发笔记保存

**保留：** `NoteAgent.process_qa()` 作为 auto-hook，每次 DeepRead 回答后自动追加 Q&A 到笔记。用户通过独立的 `/notes` API 查看和管理笔记。

**未来扩展点：** 若需要 Agent 检索历史笔记（"我之前记了什么"），可添加 `search_notes` 工具。

---

## 删除清单

| 文件/模块 | 操作 |
|---|---|
| `backend/agents/plan_editor.py` | 删除 |
| `backend/agents/recommendation_agent.py` | 删除 |
| `backend/team/` | 整个目录删除 |
| `backend/api/chat.py` 中 Dispatcher 逻辑 | 简化，直接调用 Agent |
| `frontend` 中 `active_tab` 相关逻辑 | 删除 |

## 修改清单

| 文件 | 改动 |
|---|---|
| `backend/agents/deepread_agent.py` | 合并工具为 `search_books`；扩展系统 prompt |
| `backend/api/chat.py` | 去掉 Dispatcher，直接调用 DeepRead Agent |
| `backend/main.py` | 去掉 AgentTeam lifespan，简化启动逻辑 |
| `backend/api/books.py` | `invalidate_bm25_caches` 改为直接调用 retriever |

## 不变清单

| 文件 | 说明 |
|---|---|
| `backend/agents/note_agent.py` | 保留 auto-hook，删除用户触发路径 |
| `backend/rag/` | 完全不变 |
| `backend/storage/` | 完全不变 |
| `backend/memory/mem0_store.py` | 完全不变 |
| `backend/security/input_filter.py` | 完全不变 |

---

## 演进路径

当前架构是个人使用的极简版本。以下是未来扩展点，**现在不实现**：

- **多用户**：为 `thread_id` 增加用户隔离（`user_id` 前缀）
- **笔记检索**：添加 `search_notes` 工具
- **流式中断**：添加取消机制
- **多 Agent**：若单 Agent context window 不足，或需要完全不同的人格（如严格学术模式），再拆分
