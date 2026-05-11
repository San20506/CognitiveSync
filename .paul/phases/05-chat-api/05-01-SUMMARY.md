---
phase: 05-chat-api
plan: 01
subsystem: api
tags: [fastapi, llm, openai, ollama, rbac, chat]

requires: []
provides:
  - POST /api/v1/chat/ endpoint with burnout context injection
  - LLM provider abstraction (Azure OpenAI + Ollama)
  - Chat request/response schemas
affects: [06-production-dashboard, 07-chat-widget, 08-teams-bot]

tech-stack:
  added: [openai==2.36.0]
  patterns:
    - Stateless LLM calls with injected burnout context
    - Provider abstraction via settings.llm_provider env var
    - RBAC on chat endpoint (hr_analyst, hr_admin, manager only)

key-files:
  created: [api/routes/chat.py, tests/unit/test_chat.py]
  modified: [config/settings.py, api/main.py]

key-decisions:
  - "Stateless chat — no conversation history stored (privacy constraint)"
  - "LLM provider switchable via env var: azure_openai | ollama"
  - "IT_ADMIN excluded from chat access (no burnout data need)"

patterns-established:
  - "Context injection: fetch latest BurnoutScore → inject into system prompt"
  - "Scope tagging: context_used.scope = employee | team | general"

duration: ~25min
started: 2026-05-11T00:00:00Z
completed: 2026-05-11T00:25:00Z
---

# Phase 5 Plan 01: Chat API Backend Summary

**Stateless POST /api/v1/chat/ endpoint that injects live burnout scores into LLM prompts, switchable between Azure OpenAI and Ollama via env var, with RBAC and 5 passing unit tests.**

## Performance

| Metric | Value |
|--------|-------|
| Duration | ~25 min |
| Tasks | 4/4 completed |
| Files modified | 4 |
| Tests | 5/5 passing |

## Acceptance Criteria Results

| Criterion | Status | Notes |
|-----------|--------|-------|
| AC-1: Chat endpoint returns advisory response | Pass | 200 + reply + context_used |
| AC-2: Context injection — burnout data enriches prompt | Pass | Latest BurnoutScore fetched and injected into system prompt |
| AC-3: Team-level query | Pass | Aggregates team scores, returns scope=team |
| AC-4: LLM provider configurable | Pass | azure_openai / ollama via settings.llm_provider |
| AC-5: RBAC — employee/IT_ADMIN role blocked | Pass | IT_ADMIN → 403, hr_analyst → 200 |

## Files Created/Modified

| File | Change | Purpose |
|------|--------|---------|
| `api/routes/chat.py` | Created | POST /chat/ endpoint, LLM abstraction, context injection |
| `config/settings.py` | Modified | LLM provider fields + demo_enabled field |
| `api/main.py` | Modified | Chat router mounted at /api/v1/chat |
| `tests/unit/test_chat.py` | Created | 5 unit tests covering auth, RBAC, context, Ollama |

## Decisions Made

| Decision | Rationale | Impact |
|----------|-----------|--------|
| Stateless — no history stored | Privacy constraint; no PII in conversation logs | Each request is independent; dashboard/Teams must manage UI history client-side |
| IT_ADMIN excluded from /chat | No operational need for burnout data access | Consistent with principle of minimum necessary access |
| `_call_llm` extracted as top-level async fn | Enables clean patching in unit tests | All 5 tests mock at this boundary |

## Deviations from Plan

### Summary

| Type | Count | Impact |
|------|-------|--------|
| Auto-fixed | 1 | Essential — pre-existing gap |
| Scope additions | 0 | — |
| Deferred | 0 | — |

### Auto-fixed Issues

**1. Missing `demo_enabled` field in Settings**
- **Found during:** Task 3 (mount router — verify step)
- **Issue:** `api/main.py:83` referenced `settings.demo_enabled` but field was never defined in `Settings`; `main.py` failed to import
- **Fix:** Added `demo_enabled: bool = Field(default=False)` to `config/settings.py`
- **Files:** `config/settings.py`
- **Verification:** `uv run python -c "from api.main import app"` — success

## Issues Encountered

| Issue | Resolution |
|-------|------------|
| `TokenPayload` requires `exp` field not in mock | Added `exp=9999999999` to `_mock_token()` helper |
| Mypy `Incompatible default` on `require_role` | Pre-existing pattern in codebase — added `# type: ignore[assignment]` matching existing routes |

## Next Phase Readiness

**Ready:**
- `/api/v1/chat/` live and documented in OpenAPI (`/docs`)
- Phases 06, 07, 08 can all call this endpoint unchanged
- LLM backend swappable without code changes (env var only)
- Test pattern established for future chat-adjacent routes

**Concerns:**
- Ollama must be running locally for dev; Azure OpenAI creds needed for prod — document in `.env.example`
- No streaming support yet — responses may feel slow for long LLM outputs (deferred)

**Blockers:** None

---
*Phase: 05-chat-api, Plan: 01*
*Completed: 2026-05-11*
