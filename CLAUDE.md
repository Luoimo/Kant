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
```

Settings are read by `backend/config.py` via Pydantic Settings.

## Architecture

### RAG Pipeline

The core pipeline runs EPUB → Vector DB in sequence:

```
EPUB → EpubExtractor → TextCleaner → TextChunker → OpenAI Embeddings → ChromaDB
```

`ChromaStore` (`backend/rag/chroma/chroma_store.py`) is the single entry point orchestrating all these stages. Use `ingest(path)` to run the full pipeline and `similarity_search(query, k, filter)` to retrieve.

`backend/rag/retriever/` provides additional retrieval components:
- `BM25Retriever` — keyword-based sparse retrieval
- `HybridRetriever` — BM25 + vector score fusion (RRF) with `QueryRewriter` (LLM query rewriting) and optional reranking; **now used by all four sub-agents**
- `LLMReranker` / `CrossEncoderReranker` — result reranking (wired into HybridRetriever when configured)

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

**Compound pipeline:** `IntentSchema.compound_intents` allows the supervisor to queue multiple agents (e.g., deepread + notes) in one turn. Each agent appends its output to `compound_context`; `_finalize` synthesizes a unified response.

**Per-agent message history:** `GraphState` carries `deepread_messages`, `notes_messages`, `plan_messages`, `recommend_messages` (each a `list[BaseMessage]` with `add_messages` reducer). This gives each agent its own multi-turn context within a session.

### Sub-Agents

All four agents are fully implemented and wired into the graph, each using `HybridRetriever` (BM25 + vector + RRF + optional reranking) built once at construction time:

- **DeepReadAgent** (`backend/agents/deepread_agent.py`) — evidence-based Q&A: retrieves top-k docs via HybridRetriever, builds `Citation` objects from metadata (section_indices, book_title), answers strictly from retrieved evidence, optional consistency self-check
- **NoteAgent** (`backend/agents/note_agent.py`) — structured Markdown notes in 4 formats (`structured` / `summary` / `qa` / `timeline`); supports `new` / `edit` / `extend` actions backed by `NoteStorage`; raw-text mode skips RAG
- **RecommendationAgent** (`backend/agents/recommendation_agent.py`) — book recommendations from the local library with ratings and reading advice; hash-based catalog cache (max 30 sources); excludes previously recommended titles in multi-turn sessions
- **ReadingPlanAgent** (`backend/agents/reading_plan_agent.py`) — personalized reading plans with real chapter structure from ChromaDB (`_extract_chapter_structure`), time estimates (300 chars/min), progress tracking via regex patterns; supports `new` / `edit` / `extend` actions backed by `PlanStorage`

### Persistent Storage

`backend/storage/` provides pluggable file-backed storage:

- `NoteStorage` / `LocalNoteStorage` — saves/loads/lists/deletes Markdown note files under `NOTE_STORAGE_DIR`
- `PlanStorage` / `LocalPlanStorage` — same pattern for reading plans under `PLAN_STORAGE_DIR`
- Both implement a `Protocol` (runtime-checkable) for easy substitution
- `NoteOutputMeta` / `PlanOutputMeta` TypedDicts stored in `GraphState` as serializable pointers to the last saved file

### Memory

`Mem0Store` (`backend/memory/mem0_store.py`) wraps the `mem0ai` library. Degrades gracefully if `mem0ai` is not installed. Uses ChromaDB as the vector backend for memory storage.

### Security Layer

`InputSafetyFilter` (`backend/security/input_filter.py`) runs before any agent processing. Hard blocks: secrets/API keys in input, filesystem access commands, prompt injection patterns, code execution keywords. Soft warnings: off-topic queries (non-reading-related).

### Chroma Dual Mode

`ChromaStore` automatically selects client based on config:
- `CHROMA_API_KEY` empty → `chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)` (local)
- `CHROMA_API_KEY` set → `chromadb.CloudClient(...)` (Chroma Cloud)

### LLM Client

`backend/llm/openai_client.py` wraps LangChain's OpenAI clients. `get_llm()` returns `ChatOpenAI`, `get_embeddings()` returns `OpenAIEmbeddings`. Both read model names from config, supporting custom `base_url` for proxy/alternative endpoints.

### API Layer

`backend/api/chat.py` exposes `POST /chat` via FastAPI. Request: `{query, book_source, thread_id}`. Response: `{answer, citations, retrieved_docs_count, intent}`. Entrypoint for uvicorn: `backend.main:app`.

## Test Fixtures

`tests/rag/conftest.py` provides pytest fixtures with in-memory EPUB data (no file I/O needed). All RAG component tests use these fixtures — no real EPUBs or API calls required for unit tests.
