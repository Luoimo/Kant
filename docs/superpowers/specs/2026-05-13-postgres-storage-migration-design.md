# PostgreSQL Storage Migration Design

## Goal

Migrate business metadata storage and LangGraph chat checkpoint storage from local SQLite files to PostgreSQL, while preserving current API behavior and keeping Chroma Cloud, OSS, and Neo4j unchanged.

This migration must also improve the storage boundary so business services no longer depend directly on SQLite-specific APIs, file paths, or table names.

## Scope

In scope:
- `books` metadata storage
- `notes` metadata storage
- LangGraph checkpoint persistence for multi-turn chat state
- Configuration and dependency changes required for PostgreSQL
- Data migration from existing SQLite files into PostgreSQL
- Test updates needed to support the new storage boundary

Out of scope:
- Chroma Cloud vector storage
- OSS object storage
- Neo4j graph storage
- Frontend API contract changes
- Large-scale ORM adoption beyond what is needed for this migration

## Current State

The codebase currently has two direct SQLite dependencies in business code:

1. `backend/storage/book_catalog.py`
   - Stores `books` and `notes` metadata in `data/books.db`
   - Uses direct `sqlite3` connections and inline DDL

2. `backend/agents/deepread_agent.py`
   - Stores LangGraph checkpoints in `data/chat_history.db`
   - Uses `SqliteSaver` / `AsyncSqliteSaver`
   - Reads and clears checkpoint tables by directly querying SQLite tables

This creates three problems:
- storage is tightly coupled to business code
- database-specific details leak into agent logic
- moving from local single-file storage to managed database infrastructure is awkward

## Recommended Approach

Use a repository-and-factory design:

- Replace the SQLite catalog implementation with a PostgreSQL-backed repository
- Replace the SQLite LangGraph saver with the PostgreSQL saver
- Introduce a thin storage abstraction so business code depends on interfaces and providers instead of raw database clients

This keeps the migration focused while establishing a stable storage boundary for future changes.

## Design

### 1. Storage Boundaries

Introduce two clear storage responsibilities:

- Catalog storage
  - owns `books` and `notes` metadata
  - exposes repository methods for CRUD and listing operations

- Checkpoint storage
  - owns LangGraph thread state persistence
  - exposes saver creation and thread maintenance operations

Business services should not open database connections directly. They should receive storage objects from a shared provider layer.

### 2. Catalog Repository

Keep the current behavioral surface close to the existing code to minimize churn.

Target interface:
- `add(book_id, title, author, source, total_chunks, cover_path="")`
- `update_progress(book_id, progress)`
- `update_status(book_id, status)`
- `delete(book_id)`
- `get_all()`
- `get_by_id(book_id)`
- `get_by_source(source)`

For notes:
- `upsert(book_id, file_path)`
- `touch(book_id)`
- `get_by_book_id(book_id)`
- `get_all()`
- `delete(book_id)`

Implementation plan:
- keep `get_book_catalog()` and `get_note_catalog()` as compatibility entry points
- replace the current SQLite implementation under those entry points with PostgreSQL-backed classes
- move schema creation out of inline SQLite DDL and into PostgreSQL setup SQL or migrations

### 3. Checkpoint Provider

Replace direct use of:
- `sqlite3.connect(...)`
- `SqliteSaver`
- `AsyncSqliteSaver`

with a provider/factory that returns:
- a sync PostgreSQL LangGraph saver for sync chat flows
- an async PostgreSQL LangGraph saver for SSE chat flows

The agent should no longer know:
- the database file path
- checkpoint table names
- how the underlying database connection is created

The agent should only know:
- how to request a saver
- how to request checkpoint history or clear a thread via a storage service

### 4. PostgreSQL Schema

Business metadata tables:

`books`
- `book_id TEXT PRIMARY KEY`
- `title TEXT NOT NULL`
- `author TEXT NOT NULL DEFAULT ''`
- `source TEXT NOT NULL UNIQUE`
- `total_chunks INTEGER NOT NULL DEFAULT 0`
- `added_at TIMESTAMPTZ NOT NULL`
- `cover_path TEXT NOT NULL DEFAULT ''`
- `status TEXT NOT NULL DEFAULT 'unread'`
- `progress DOUBLE PRECISION NOT NULL DEFAULT 0`

`notes`
- `note_id TEXT PRIMARY KEY`
- `book_id TEXT NOT NULL UNIQUE REFERENCES books(book_id) ON DELETE CASCADE`
- `file_path TEXT NOT NULL`
- `created_at TIMESTAMPTZ NOT NULL`
- `updated_at TIMESTAMPTZ NOT NULL`

LangGraph checkpoint tables:
- created and managed by the PostgreSQL LangGraph checkpointer setup routine

Notes:
- keep `TEXT` for IDs for compatibility with current UUID string usage
- keep timestamps timezone-aware
- keep `source` unique because current ingest logic depends on source identity

### 5. Configuration

Add PostgreSQL configuration to `backend/config.py`:
- `postgres_dsn`
- optional pool sizing settings if needed by the selected client

Remove storage-path assumptions from business code:
- `book_catalog_db`
- hardcoded `data/chat_history.db`

Backward compatibility during rollout:
- allow both old SQLite settings and new PostgreSQL settings to coexist briefly if needed
- prefer PostgreSQL when configured

### 6. Dependencies

Replace SQLite checkpoint dependency with PostgreSQL equivalents.

Expected additions:
- `langgraph-checkpoint-postgres`
- `psycopg[binary]`

Expected removals:
- `langgraph-checkpoint-sqlite`

The standard library `sqlite3` import should disappear from business code after migration.

### 7. Application Wiring

Initialize storage providers during app startup and attach them to app state or module-level providers.

The important rule is:
- connection construction belongs in startup/provider code
- query execution belongs in repository/checkpoint service code
- request handlers and agents use the abstraction only

### 8. File-Level Changes

Expected code changes:

- `backend/config.py`
  - add PostgreSQL settings
  - deprecate SQLite file-path settings used by business storage

- `backend/storage/book_catalog.py`
  - replace SQLite implementation with PostgreSQL repository implementation
  - or split into interface plus PostgreSQL implementation while preserving public factory helpers

- `backend/storage/`
  - add a PostgreSQL connection/provider module if needed
  - add checkpoint provider/service module

- `backend/agents/deepread_agent.py`
  - remove direct SQLite and saver construction
  - replace raw checkpoint table access with storage service methods

- `backend/api/books.py`
  - no API contract changes expected
  - should continue to use catalog factories without knowing storage backend details

- tests under `backend/tests/`
  - replace any SQLite-coupled assumptions
  - add coverage for PostgreSQL-backed catalog behavior and checkpoint behavior

## Migration Plan

### Phase 1. Introduce PostgreSQL storage layer

- add PostgreSQL settings
- add PostgreSQL repository implementation for catalog data
- add checkpoint provider based on PostgreSQL LangGraph saver
- keep existing public catalog accessors stable

### Phase 2. Refactor business code to use storage boundaries

- refactor `deepread_agent.py` to remove direct database operations
- move chat history loading and clearing into the checkpoint service layer
- update startup wiring to initialize providers

### Phase 3. Migrate data

Business metadata migration:
- read existing rows from `data/books.db`
- insert/upsert into PostgreSQL `books` and `notes`

Checkpoint migration:
- export existing thread state from SQLite checkpoint storage
- import into PostgreSQL checkpoint storage if historical continuity is required

If chat continuity is not important, checkpoint migration may be skipped and new threads can start in PostgreSQL.

Recommendation:
- migrate `books` and `notes` data
- treat old chat checkpoints as optional migration data unless there is a firm product requirement to preserve existing conversations

### Phase 4. Remove SQLite business dependencies

- drop SQLite-specific configuration from active use
- remove SQLite-specific business imports
- update documentation and environment examples

## Data Migration Strategy

### Books and Notes

Use a one-time migration script that:
- connects to the existing SQLite catalog database
- reads `books` and `notes`
- upserts them into PostgreSQL
- logs row counts and conflicts

The script must be idempotent so it can be rerun safely.

### Chat Checkpoints

Two options:

Option A: no history migration
- easiest and lowest risk
- existing users lose old multi-turn thread continuity

Option B: migrate checkpoints
- preserves chat continuity
- requires reading the current LangGraph checkpoint records and re-writing them through the PostgreSQL saver or a compatible import path

Recommendation:
- default to Option A unless the team explicitly needs historical conversations

## Testing Strategy

Add or update tests for:
- book catalog CRUD against PostgreSQL
- note catalog CRUD against PostgreSQL
- chat history retrieval and clear operations through the checkpoint service
- sync chat flow using PostgreSQL checkpoint saver
- async SSE chat flow using PostgreSQL checkpoint saver

Testing guidance:
- unit tests should mock repository interfaces where business logic is the focus
- integration tests should run against a disposable PostgreSQL instance

At minimum, verify:
- API behavior is unchanged
- thread isolation by `user_id + book_id` still works
- clearing history clears only the intended thread
- uploads, listing, note updates, and deletion still work end to end

## Risks

### 1. Agent code still knows table details

If `deepread_agent.py` continues to inspect checkpoint tables directly, the storage abstraction will be incomplete and future backend changes will remain expensive.

Mitigation:
- move all checkpoint inspection and deletion into a storage service module

### 2. PostgreSQL connection handling in async flows

The sync and async chat paths currently create storage objects inline. This pattern must be replaced carefully to avoid leaked connections or mismatched sync/async clients.

Mitigation:
- define explicit sync and async provider methods
- centralize connection lifecycle rules

### 3. Chat history compatibility

The current history reader relies on LangGraph checkpoint internals and message layout. Migration may break history retrieval if the new saver surfaces state differently.

Mitigation:
- keep history parsing logic isolated in one place
- verify retrieval against the actual PostgreSQL saver behavior before removing old code

### 4. Deployment configuration mistakes

Misconfigured DSNs or missing setup steps will fail application startup.

Mitigation:
- validate configuration on startup
- fail fast with a clear error
- document required environment variables

## Success Criteria

The migration is successful when:
- business metadata is stored in PostgreSQL instead of SQLite
- LangGraph checkpoints are stored in PostgreSQL instead of SQLite
- `deepread_agent.py` no longer imports `sqlite3` or uses SQLite savers
- `book_catalog.py` no longer imports `sqlite3`
- existing APIs behave the same from the frontend's perspective
- Chroma Cloud remains unchanged
- data migration for `books` and `notes` is repeatable

## Recommendation Summary

Implement PostgreSQL as the new system of record for business metadata and LangGraph checkpoints, but do not expand this work into a full ORM rewrite or vector-store redesign.

This is the smallest change set that both meets the infrastructure goal and establishes a clean storage boundary for future evolution.
