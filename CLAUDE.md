# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/rag/test_chroma_store.py -v

# Run a single test
pytest tests/rag/test_text_chunker.py::test_name -v

# Run tests with coverage
pytest tests/ --cov=backend --cov-report=html

# Run RAG demo (ingest EPUBs + query)
python -m scripts.rag_demo

# Run LangGraph orchestration demo
python -m scripts.deepread_graph_demo

# Start API server
uvicorn backend.main:app --reload
```

## Configuration

Copy `.env` and fill in values:

```env
OPENAI_API_KEY=sk_xxxxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

CHROMA_PERSIST_DIR=data/chroma
BOOKS_DATA_DIR=data/books

# Optional: set to enable Chroma Cloud (otherwise uses local PersistentClient)
CHROMA_API_KEY=
CHROMA_TENANT=default_tenant
CHROMA_DATABASE=default_database

# Mem0 long-term memory user ID
MEM0_USER_ID=default_user

# Persistent storage directories for notes and reading plans
NOTE_STORAGE_DIR=data/notes
PLAN_STORAGE_DIR=data/plans

# SQLite catalog for books/notes/plans metadata
BOOK_CATALOG_DB=data/books.db

# Directory for extracted EPUB covers
COVERS_DIR=data/covers
```

Settings are read by `backend/config.py` via Pydantic Settings.

## Architecture

### RAG Pipeline

The core pipeline runs EPUB → Vector DB in sequence:

```
EPUB → EpubExtractor → TextCleaner → TextChunker → OpenAI Embeddings → ChromaDB
```

`ChromaStore` (`backend/rag/chroma/chroma_store.py`) is the single entry point orchestrating all these stages. Use `ingest(path)` to run the full pipeline and `similarity_search(query, k, filter)` to retrieve.

All book chunks go into a **single collection** (`kant_library` by default, configurable via `books_collection_name`). Books are distinguished by the `source` metadata field (full file path). There is no separate per-book collection or catalog collection.

Key `ChromaStore` methods:
- `ingest(path)` — full pipeline: extract → clean → chunk → embed → upsert (batched dedup with `embed_batch_size` to respect Chroma Cloud 300-record Get limit); returns `IngestResult` with `book_title`, `author`, `source`, `total_chunks`, `added`, `skipped`
- `similarity_search(query, k, filter)` — vector search with optional `{"source": ...}` filter
- `get_all_documents(filter)` — bulk fetch used by BM25 index construction and chapter extraction; uses paginated loop (limit+offset) to work around Chroma Cloud's 300-record-per-request `Get` limit

`backend/rag/retriever/` provides additional retrieval components:
- `BM25Retriever` — keyword-based sparse retrieval using `rank_bm25` + jieba tokenization
- `HybridRetriever` — BM25 + vector score fusion (RRF) with `QueryRewriter` (LLM query rewriting) and optional reranking; used by `NoteAgent` and `ReadingPlanAgent` (shared instance per agent); `DeepReadAgent` creates a fresh instance per call
- `LLMReranker` / `CrossEncoderReranker` — result reranking (wired into HybridRetriever when configured)

**BM25 caching:** `HybridRetriever` lazily builds a BM25 index on first `search()` call and caches it in `_bm25_cache: dict[str, BM25Retriever]` keyed by filter string. Different `source` filters each get their own cached index, preventing cross-book pollution. Call `retriever.invalidate_bm25()` to clear all cached indices. `invalidate_bm25_caches()` in `orchestrator_agent.py` wraps this for all shared-retriever agents and is called automatically by `POST /books/upload` on successful ingest.

### Agent System

Built on **LangGraph** with a supervisor pattern. Graph structure:

```
START → memory_search → supervisor → [deepread|notes|plan|recommend] → supervisor → finalize → END
```

1. `memory_search` node — searches Mem0 for relevant past Q&A, injects `memory_context` into state
2. `supervisor` node — runs `InputSafetyFilter`, classifies intent (recommend/deepread/notes/plan), routes to the appropriate agent node; supports **compound pipeline** via `pending_agents` queue for multi-agent queries
3. Agent node runs, writes `answer` / `citations` / `retrieved_docs_count` / per-agent message history back to state
4. `finalize` node — synthesizes compound answer when multiple agents ran; saves Q&A to Mem0

Entry point: `run_minimal_graph(query, *, book_source, thread_id)` — `thread_id` enables multi-turn memory via `SqliteSaver` (persistent checkpoint at `data/checkpoints.db`; falls back to `MemorySaver` if sqlite unavailable).

**Why agents live in `GraphDeps` (not `GraphState`):** LangGraph's checkpointer serializes `GraphState`; agent objects aren't serializable, so they're injected via closure through `GraphDeps`.

**Compound pipeline:** `IntentSchema.compound_intents` allows the supervisor to queue multiple agents (e.g., deepread + notes) in one turn. Each agent appends its output to `compound_context`; `_finalize` synthesizes a unified response. Critical invariant: the supervisor only treats a turn as "done" (routes to finalize) when `pending_agents` is empty — never short-circuit a running pipeline by checking `answer` alone.

**Per-agent message history:** `GraphState` carries `deepread_messages`, `notes_messages`, `plan_messages`, `recommend_messages` (each a `list[BaseMessage]` with `add_messages` reducer). This gives each agent its own multi-turn context within a session.

**State reset per turn:** `run_minimal_graph` explicitly resets transient fields each invocation: `intent`, `action`, `answer`, `pending_agents`, `compound_context`, `safety_ok`. This prevents checkpoint pollution across turns. `safety_ok: None` triggers re-execution of the safety check on every turn (guard is `if state.get("safety_ok") is None`, not `if "safety_ok" not in state`).

**Compound recommend → plan/notes routing:** When `recommend` precedes `plan` or `notes` in a compound pipeline, `_extract_recommended_book(compound_context)` extracts the first `《书名》` from the recommend output and sets it as `plan_book_source` / `notes_book_source`.

### Sub-Agents

**DeepReadAgent** (`backend/agents/deepread_agent.py`)
- Evidence-based Q&A: retrieves top-k docs via HybridRetriever (creates a fresh retriever per `run()` call, so no BM25 cache accumulation), builds `Citation` objects from metadata, answers strictly from retrieved evidence

**NoteAgent** (`backend/agents/note_agent.py`)
- Structured Markdown notes in 4 formats: `structured` / `summary` / `qa` / `timeline`
- Supports `new` / `edit` / `extend` actions backed by `NoteStorage`
- Has a single shared `_retriever` instance whose BM25 cache persists across requests

**RecommendationAgent** (`backend/agents/recommendation_agent.py`)
- Recommends books from **LLM training knowledge**, not limited to the local library
- Never does RAG retrieval — `citations` and `retrieved_docs` are always empty lists
- Multi-turn de-duplication: extracts previously recommended titles from `recommend_messages` history and excludes them from the next prompt

**PlanEditor** (`backend/agents/plan_editor.py`)
- Two paths: `generate()` (non-ReAct, triggered by REST API when user opens a book) and `run()` (ReAct agent, triggered by chat for edit/extend)
- `generate()` extracts all chapter structure from ChromaDB via `get_all_documents()` (paginated), calls LLM only for the schedule section, writes plan to disk and registers in `PlanCatalog`
- `run()` builds a LangGraph ReAct agent with `load_existing_plan` and `get_chapter_structure` tools; reads/writes via `PlanCatalog` for file lookup
- Files are named by `book_id` (uuid5), stored under `PLAN_STORAGE_DIR`

### Persistent Storage

Two-layer approach: SQLite tracks metadata, file system holds content.

**SQLite catalog** (`backend/storage/book_catalog.py`, shared DB at `data/books.db`):
- `BookCatalog` — books table: `book_id`, `title`, `author`, `source`, `total_chunks`, `added_at`, `cover_path`, `status`, `progress`
- `NoteCatalog` — notes table: `note_id`, `book_id`, `file_path`, `created_at`, `updated_at`
- `PlanCatalog` — plans table: `plan_id`, `book_id`, `file_path`, `reading_goal`, `created_at`, `updated_at`
- All three share one DB file and a common `_DB` base class that runs full DDL on init
- `book_id_from_source(source)` — deterministic `uuid5(NAMESPACE_URL, source)`; same formula used by `ChromaStore.ingest()`, so IDs are consistent without a DB lookup
- Factory functions: `get_book_catalog()`, `get_note_catalog()`, `get_plan_catalog()`

**File-backed storage** (`backend/storage/`):
- `NoteStorage` / `LocalNoteStorage` — saves/loads/lists/deletes Markdown note files under `NOTE_STORAGE_DIR`; files named `{book_id}.md`
- `PlanStorage` / `LocalPlanStorage` — same pattern for reading plans under `PLAN_STORAGE_DIR`; files named `{book_id}.md`
- Both implement a `Protocol` (runtime-checkable) for easy substitution
- `NoteOutputMeta` / `PlanOutputMeta` TypedDicts stored in `GraphState` as serializable pointers to the last saved file

### Memory

`Mem0Store` (`backend/memory/mem0_store.py`) wraps the `mem0ai` library. Degrades gracefully if `mem0ai` is not installed. Uses ChromaDB as the vector backend for memory storage.

**Two-layer memory:**
| Layer | Storage | Scope | Contents |
|-------|---------|-------|----------|
| Short-term | `data/checkpoints.db` (SQLite) | Per `thread_id` | Exact `GraphState` snapshots; enables multi-turn within a session |
| Long-term | Mem0 / ChromaDB | Global per user | LLM-distilled semantic summaries; cross-session user profile |

The SQLite checkpoint has two tables: `checkpoints` (full state snapshots) and `writes` (per-node delta records). Both are msgpack-encoded.

### Security Layer

`InputSafetyFilter` (`backend/security/input_filter.py`) runs before any agent processing on **every turn** (`safety_ok` is reset to `None` in `run_minimal_graph` init). Hard blocks: secrets/API keys in input, filesystem access commands, prompt injection patterns, code execution keywords. Soft warnings: off-topic queries (non-reading-related).

### Chroma Dual Mode

`ChromaStore` automatically selects client based on config:
- `CHROMA_API_KEY` empty → `chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)` (local)
- `CHROMA_API_KEY` set → `chromadb.CloudClient(...)` (Chroma Cloud)

**Chroma Cloud quota:** Free tier has a 300-record-per-request limit on `Get` operations. `_ingest_chunks_to_db` batches dedup checks in slices of `embed_batch_size` (default 100); `get_all_documents()` uses a paginated while-loop (limit+offset) to fetch beyond 300 records.

### LLM Client

`backend/llm/openai_client.py` wraps LangChain's OpenAI clients. `get_llm()` returns `ChatOpenAI`, `get_embeddings()` returns `OpenAIEmbeddings`. Both read model names from config, supporting custom `base_url` for proxy/alternative endpoints.

### API Layer

`backend/api/chat.py` exposes two endpoints via FastAPI. Entrypoint for uvicorn: `backend.main:app`.

**POST /chat**
- Request: `{query, book_source?, thread_id?, active_tab?, selected_text?, current_chapter?}`
- Response: `{answer, citations, retrieved_docs_count, intent}`
- `active_tab` bypasses LLM intent classification and routes directly to the named agent
- `selected_text` / `current_chapter` inject reader context into the query

**GET /books**
- Returns all books from `BookCatalog` SQLite, including `book_id`, `status`, `progress`, `cover_path`

**POST /books/upload**
- Multipart file upload (`.epub` only)
- Saves file to `BOOKS_DATA_DIR`, runs full ingest pipeline, writes to `BookCatalog`, extracts cover to `COVERS_DIR`
- On success, calls `invalidate_bm25_caches()` to clear stale BM25 indices in all agents
- Response: `{book_id, source, collection_name, total_chunks, added, skipped}`

**POST /reader/{book_id}/init**
- Auto-generates a reading plan when user opens a book (idempotent)
- Looks up book in `BookCatalog`, calls `PlanEditor.generate()`, updates status to `"reading"`

**GET /reader/{book_id}/plan**
- Returns current plan markdown; looks up file path via `PlanCatalog`

**POST /reader/{book_id}/progress**
- Marks a chapter as complete (`- [ ]` → `- [x]`) in the plan file
- Recomputes `progress` fraction and syncs to `BookCatalog` + `PlanCatalog.touch()`

**GET /notes/books**, **POST /notes/append**, etc.
- Notes endpoints use `book_id` for all lookups; resolve to title via `BookCatalog` internally

## Test Fixtures

`tests/rag/conftest.py` provides pytest fixtures with in-memory EPUB data (no file I/O needed). All RAG component tests use these fixtures — no real EPUBs or API calls required for unit tests.
