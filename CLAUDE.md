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

# Run RAG demo (ingest PDFs + query)
python -m scripts.rag_demo

# Run LangGraph orchestration demo
python -m scripts.deepread_graph_demo
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
```

Settings are read by `backend/config.py` via Pydantic Settings.

## Architecture

### RAG Pipeline

The core pipeline runs PDF → Vector DB in sequence:

```
PDF → PDFExtractor → TextCleaner → TextChunker → OpenAI Embeddings → ChromaDB
```

`ChromaStore` (`backend/rag/chroma/chroma_store.py`) is the single entry point orchestrating all these stages. Use `ingest_pdf(path)` to run the full pipeline and `similarity_search(query, k, filter)` to retrieve.

### Agent System

Built on **LangGraph** with a supervisor pattern:

1. `OrchestratorAgent` (`backend/agents/orchestrator_agent.py`) runs a graph: `START → supervisor_node → [route] → agent_node → supervisor_node → finalize_node → END`
2. The supervisor runs `InputSafetyFilter`, classifies intent (recommend/deepread/notes/plan), then routes to the appropriate agent.
3. Entry point: `run_minimal_graph(query, *, book_source, thread_id)` — `thread_id` enables multi-turn memory via LangGraph's `MemorySaver`.

**Why agents live in `GraphDeps` (not `GraphState`):** LangGraph's checkpointer serializes `GraphState`; agent objects aren't serializable, so they're injected via closure through `GraphDeps`.

### DeepRead Agent

`DeepReadAgent` (`backend/agents/deepread_agent.py`) does evidence-based Q&A:
1. Retrieves top-k docs from ChromaStore
2. Builds `Citation` objects from doc metadata (page numbers, chapter/section from TOC)
3. Answers strictly from retrieved evidence (hallucination-resistant system prompt)
4. Optionally runs a consistency self-check via a second LLM call

### Security Layer

`InputSafetyFilter` (`backend/security/input_filter.py`) runs before any agent processing. Hard blocks: secrets/API keys in input, filesystem access commands, prompt injection patterns, code execution keywords. Soft warnings: off-topic queries (non-reading-related).

### Chroma Dual Mode

`ChromaStore` automatically selects client based on config:
- `CHROMA_API_KEY` empty → `chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)` (local)
- `CHROMA_API_KEY` set → `chromadb.CloudClient(...)` (Chroma Cloud)

### LLM Client

`backend/llm/openai_client.py` wraps LangChain's OpenAI clients. `get_llm()` returns `ChatOpenAI`, `get_embeddings()` returns `OpenAIEmbeddings`. Both read model names from config, supporting custom `base_url` for proxy/alternative endpoints.

## Test Fixtures

`tests/rag/conftest.py` provides pytest fixtures with in-memory PDF data (no file I/O needed). All RAG component tests use these fixtures — no real PDFs or API calls required for unit tests.

## Agents Under Development

- `NoteAgent` (`backend/agents/note_agent.py`): Stub with design doc in comments
- `RecommendationAgent`, `ReadingPlanAgent`: Defined but not wired into the main graph
- `backend/api/chat.py`: API layer exists but not fully integrated
