# User Isolation and Authentication Design

## Goal

Design a platform-level authentication and authorization mechanism that enforces strict per-user data isolation for normal users, while allowing administrators to transparently view user data in a read-only admin console.

The design must be directly compatible with the current Kant architecture and current delivery constraints.

## Scope

In scope:
- Account system for platform users (`member`, `admin`)
- JWT-based authentication with Redis-backed refresh token management
- Application-layer authorization and ownership checks
- Per-user isolation for books, conversations, chat messages, vector retrieval scope, and object storage paths
- Separate API domains and frontend entry points for user and admin roles
- Audit logging for authentication and admin data access

Out of scope:
- PostgreSQL RLS
- Chat thread mapping table
- Multi-tenant implementation in this phase

## Confirmed Constraints

This design is based on explicit team decisions:
- Single-tenant, multi-user platform
- Full private isolation for normal users
- `admin` can see normal user data, but admin capabilities are read-only
- Per-book multi-conversation model
- Refresh token persisted in Redis
- No RLS in this phase; enforce isolation in application layer
- No chat mapping table in this phase

## Current State

The current backend still accepts `user_id` from request payload in chat flows and uses default values in several places. Business data tables also do not consistently enforce `owner_user_id` boundaries.

This creates a primary risk:
- identity can be influenced by request input instead of being solely derived from a verified token

For a secure user-isolated platform, identity must become server-trusted only, and every data access path must be owner-constrained.

## Recommended Approach

Use a dual-layer application strategy:

1. Identity and role from JWT claims only  
2. Ownership enforcement at every repository/service query

This avoids the complexity of immediate RLS rollout while still giving clear, testable, and auditable boundaries.

## Design

### 1. Identity and Session Model

#### 1.1 User roles

- `member`: can read/write only owned resources
- `admin`: read-only visibility across users via dedicated admin APIs

#### 1.2 Token model

- Access token: JWT, short-lived (for example 15 minutes), stateless
- Refresh token: opaque or JWT-style token with `jti`, validated against Redis

Access token claims:
- `sub`: user id
- `role`: `member` or `admin`
- `jti`
- `iat`
- `exp`
- reserved `tenant_id: "default"` for future upgrade path

#### 1.3 Redis session keys (refresh token)

- `auth:refresh:{user_id}:{jti}` -> hash payload (`token_hash`, `expires_at`)
- `auth:user_sessions:{user_id}` -> set of active `jti` values

Rules:
- Store only refresh token hash, never plaintext
- Refresh endpoint must verify Redis key existence and hash match
- Token rotation on refresh: revoke old `jti`, issue new `jti`
- Logout deletes current `jti`
- Logout-all deletes all `jti` for that user

### 2. API Domains and Frontend Separation

#### 2.1 API split

- `/api/user/*`: normal user operations
- `/api/admin/*`: administrator read-only operations

#### 2.2 Frontend split

- Member frontend: only user-domain pages and APIs
- Admin frontend: dedicated admin pages and APIs

Backend remains the source of truth. UI routing is not treated as a security boundary.

#### 2.3 Authentication endpoints

- `POST /auth/register`
- `POST /auth/login`
- `POST /auth/refresh`
- `POST /auth/logout`
- `POST /auth/logout-all`

### 3. Data Model Changes

### 3.1 Core tables

- `users`
  - `id`, `email`, `password_hash`, `role`, `status`, `created_at`
- `books` (extend current)
  - add `owner_user_id`
- `conversations` (new)
  - `id`, `owner_user_id`, `book_id`, `title`, `created_at`, `updated_at`
- `chat_messages` (new)
  - `id`, `conversation_id`, `owner_user_id`, `role`, `content`, `created_at`
- `audit_logs` (new)
  - `id`, `actor_user_id`, `actor_role`, `action`, `resource_type`, `resource_id`, `ip`, `ua`, `result`, `created_at`

### 3.2 Ownership invariants

Must hold for all request paths:
- Book access: `book.owner_user_id == current_user.id` for member
- Conversation access: `conversation.owner_user_id == current_user.id`
- Chat write/read: `message.owner_user_id == current_user.id`
- Conversation-book integrity: `conversation.book_id == request.book_id`

### 4. Authorization Enforcement (Application Layer)

#### 4.1 Trusted identity source

Reject body/query-level identity inputs for authorization.  
All auth context is derived from verified access token.

#### 4.2 Dependency guards

- `get_current_user`: validates JWT and returns trusted auth context
- `require_role("admin")`: admin-only guard
- ownership check helper in service/repository layer for all member resources

#### 4.3 Admin read-only policy

- Admin endpoints expose query/view APIs only
- No admin write endpoints are provided
- If admin hits user write path, return `403`

### 5. Conversation Model (Per-Book Multi-Conversation)

#### 5.1 Conversation lifecycle

- Create conversation under a specific `book_id`
- List conversations per owned book
- Send chat with `{book_id, conversation_id, query}`

#### 5.2 Thread key strategy

No mapping table in this phase.  
Use server-owned `conversation_id` directly as the checkpoint thread key.

Validation before agent run:
- conversation exists
- conversation owner matches current user
- conversation book matches `book_id`

### 6. Storage Isolation Beyond SQL

### 6.1 Vector retrieval

All vector documents must include `owner_user_id` metadata.  
Retrieval and deletion queries must include owner constraints.

### 6.2 Object storage paths

Object keys must be user-scoped:
- `users/{user_id}/books/...`
- `users/{user_id}/covers/...`

### 6.3 Long-term memory scope

Memory store user scope must use authenticated user id (`sub`), not global static `MEM0_USER_ID`.

### 7. Audit and Governance

Record audit events for:
- login / refresh / logout / logout-all
- admin data views
- access denied decisions (`401`, `403` high-value paths)

Minimum audit fields:
- actor identity and role
- action and resource
- timestamp
- request IP and user-agent
- decision result

## Security Baseline

- Password hashing: Argon2id
- Refresh token storage: hash only
- Short-lived access tokens
- Strict server-side ownership checks on every resource query
- Role-domain API split

## Testing and Acceptance Criteria

### 1. Authentication tests

- unauthenticated user-domain request returns `401`
- member access to admin-domain returns `403`
- tampered or expired access token returns `401`
- rotated refresh token cannot be reused

### 2. Isolation tests

- user B cannot access user A books/conversations/messages
- forged `conversation_id` from another user is rejected
- delete operations only affect caller-owned data
- vector retrieval contains owner filter

### 3. Conversation consistency tests

- one book with multiple conversations keeps isolated histories
- mismatched `book_id` and `conversation_id` is rejected

### 4. Admin policy tests

- admin read endpoints succeed
- admin write attempts are rejected (`403`)
- admin read actions generate audit logs

### 5. Resilience checks

- Redis unavailable: login/refresh fail closed; user APIs do not bypass auth
- SSE chat path aborts immediately on auth failure

## Implementation Phasing

### Phase 1: Auth foundation

- add user model and auth endpoints
- add JWT validation dependency
- add Redis refresh token lifecycle

### Phase 2: Ownership retrofit

- add `owner_user_id` to core business entities
- remove request-level `user_id` trust
- enforce owner constraints in repositories/services

### Phase 3: Conversation and chat refactor

- introduce `conversations` and `chat_messages`
- switch chat APIs to `book_id + conversation_id`
- enforce conversation ownership and book consistency

### Phase 4: Admin read console and audit

- add `/api/admin/*` read-only APIs
- build admin frontend entry
- add audit log persistence and query support

### Phase 5: Hardening and evidence

- implement integration/security tests
- produce test evidence for project report and demo

## Risks and Mitigations

- Risk: missed ownership check in new endpoints  
  Mitigation: repository-level guard patterns + integration tests per resource type

- Risk: role leakage through frontend-only restrictions  
  Mitigation: strict backend role guards for all admin routes

- Risk: refresh token replay  
  Mitigation: Redis-backed hash validation + mandatory rotation

## Non-Goals for This Phase

- RLS-based data-plane hard enforcement
- Cross-tenant isolation implementation
- Admin delegated impersonation workflows

## Summary

This design provides a practical, course-ready isolation/auth foundation:
- strong identity trust boundary
- per-user private data isolation
- transparent but read-only admin visibility
- clear auditability
- phased implementation path that matches the existing Kant codebase and delivery constraints
