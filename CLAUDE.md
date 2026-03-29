# CLAUDE.md

## 项目概述

EPUB 阅读助手，基于 FastAPI + LangGraph + ChromaDB。RAG 管道：EPUB → ChromaDB；Agent 系统：supervisor 路由到 DeepRead / Notes / Plan / Recommend 四个子 Agent。

**技术栈：** FastAPI · LangGraph · ChromaDB · OpenAI · SQLite · Mem0 · rank_bm25 · jieba

## 文件结构

```
backend/
  agents/          # deepread_agent, note_agent, plan_editor, recommendation_agent, orchestrator_agent
  api/             # chat.py, reader.py, notes.py
  rag/             # chroma_store.py, retriever/ (BM25 + Hybrid)
  storage/         # book_catalog.py (SQLite), note_storage.py, plan_storage.py
  memory/          # mem0_store.py
  security/        # input_filter.py
  llm/             # openai_client.py
  config.py
tests/             # mirrors backend/ structure
```

## 环境变量

```env
OPENAI_API_KEY=sk_xxxxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
CHROMA_PERSIST_DIR=data/chroma
BOOKS_DATA_DIR=data/books
CHROMA_API_KEY=          # 留空用本地，填写用 Chroma Cloud
NOTE_STORAGE_DIR=data/notes
PLAN_STORAGE_DIR=data/plans
BOOK_CATALOG_DB=data/books.db
COVERS_DIR=data/covers
MEM0_USER_ID=default_user
```

## 可用命令

```bash
uvicorn backend.main:app --reload   # 启动 API
pytest tests/ -v                    # 运行所有测试
pytest tests/ --cov=backend         # 覆盖率报告
python -m scripts.rag_demo          # RAG 演示
```

## 关键规则

- **不可变性：** 永远返回新对象，不要原地修改
- **文件大小：** 单文件 ≤ 800 行，函数 ≤ 50 行
- **book_id：** 始终用 `uuid5(NAMESPACE_URL, source)` 生成，保持与 ChromaDB source 一致
- **BM25 缓存：** 上传书籍后必须调用 `invalidate_bm25_caches()`
- **安全过滤：** `InputSafetyFilter` 在每轮对话都重新运行（`safety_ok` 重置为 `None`）
- **测试：** 覆盖率 ≥ 80%；RAG 单元测试用 `tests/rag/conftest.py` 的内存 fixture，无需真实 EPUB

## Git 工作流

```
feat/fix/refactor/docs/test/chore: <描述>
```

- 提交前检查：无硬编码密钥、输入已校验、无 SQL 注入
- PR 使用 `git diff main...HEAD` 查看完整变更

## 可参考文档
- docs/superpowers 从前的计划与设计文档
- .claude/agents 执行特定任务时，从中选取角色进行扮演
- .claude/rules 执行任务时，需要参考的限制和规则