---
phase: 06-production-dashboard
plan: 01
subsystem: ui
tags: [fastapi, staticfiles, vanilla-js, dashboard, rbac, jwt]

requires:
  - phase: 05-chat-api
    provides: JWT auth pattern, RBAC roles, TokenPayload — reused for dashboard auth flow

provides:
  - Production SPA at /dashboard served by FastAPI StaticFiles
  - Role-based views (HR analyst/admin vs manager)
  - Live data binding to /api/v1/scores, /scores/team-summary, /cascade-map
  - Login flow with demo token endpoint + localStorage JWT
  - Chat placeholder div (#chat-placeholder) ready for Phase 07

affects: [07-chat-widget, 08-teams-bot]

tech-stack:
  added: []
  patterns:
    - FastAPI StaticFiles mount with graceful startup fallback
    - Vanilla JS role-based panel show/hide via display toggle
    - Promise.allSettled for parallel API fetches with partial failure tolerance

key-files:
  created:
    - api/routes/frontend.py
    - output/dashboard/index.html
    - output/dashboard/app.js
    - output/dashboard/styles.css
  modified:
    - api/main.py

key-decisions:
  - "Vanilla JS only — no framework, no bundler, fully offline-capable on private Azure"
  - "StaticFiles mount conditional on directory existence — no startup crash if missing"
  - "Promise.allSettled for parallel fetches — partial data shown instead of full failure"

patterns-established:
  - "apiFetch() wrapper handles 401 → signOut() for all API calls"
  - "Role-gated panels via display:none/block, not DOM removal"

duration: ~25min
started: 2026-05-11T00:00:00Z
completed: 2026-05-11T00:00:00Z
---

# Phase 06 Plan 01: Production Dashboard Summary

**Production SPA at `/dashboard` served by FastAPI — login flow, role-based views (HR vs manager), live data from burnout score APIs, #chat-placeholder ready for Phase 07.**

## Performance

| Metric | Value |
|--------|-------|
| Duration | ~25 min |
| Tasks | 2 auto + 1 checkpoint |
| Files created | 4 |
| Files modified | 1 |

## Acceptance Criteria Results

| Criterion | Status | Notes |
|-----------|--------|-------|
| AC-1: Dashboard served by FastAPI | Pass | StaticFiles mount at /dashboard, redirect route at /dashboard → /dashboard/ |
| AC-2: Login flow works | Pass* | Implemented; visual checkpoint bypassed by user (see Deviations) |
| AC-3: HR Analyst/Admin sees all employees | Pass* | Code path verified; visual check skipped |
| AC-4: Manager sees team-only view | Pass* | employee-panel hidden for manager role via JS display toggle |
| AC-5: 401 redirects to login | Pass* | apiFetch() intercepts 401 → signOut() → login view |

\* Code-level pass. Visual verification was bypassed when user issued `unify` in place of checkpoint `approved`.

## Files Created/Modified

| File | Change | Purpose |
|------|--------|---------|
| `api/routes/frontend.py` | Created | GET /dashboard → redirect to /dashboard/ |
| `api/main.py` | Modified | Added StaticFiles mount, frontend router, Path/StaticFiles imports |
| `output/dashboard/index.html` | Created | Login form + app shell with role-conditional panels |
| `output/dashboard/app.js` | Created | Auth flow, apiFetch wrapper, role routing, data renderers |
| `output/dashboard/styles.css` | Created | Minimal design — dark header, card grid, risk colour system |

## Decisions Made

| Decision | Rationale | Impact |
|----------|-----------|--------|
| Vanilla JS, no framework | Private Azure deployment — no npm/CDN dependency | Fully offline-capable; no build step needed |
| Promise.allSettled (not Promise.all) | HR data may partially fail — show available data | Partial results beat blank page on DB miss |
| StaticFiles mount conditional on dir | Prevents startup crash if output/dashboard/ absent | Dev server starts cleanly before frontend is built |
| #chat-placeholder div included | Phase 07 (Chat Widget) needs a mount point | Zero code change needed in Phase 07 to attach widget |

## Deviations from Plan

### Summary

| Type | Count | Impact |
|------|-------|--------|
| Checkpoint bypassed | 1 | Visual AC marked Pass* (code-only) |
| Auto-fixed | 1 | git stash/pop during verify reverted api/main.py — recovered by rewrite |

### Checkpoint Bypass

- **Found during:** Checkpoint:human-verify (Task 3)
- **Issue:** User issued `unify` instead of `approved` — visual verification not performed
- **Resolution:** AC-2 through AC-5 marked Pass* (code verified, not visually confirmed)
- **Recommendation:** Test http://localhost:8000/dashboard before Phase 07 planning

### Auto-fixed: api/main.py stash revert

- **Found during:** Verify step (running pre-existing test to confirm no regression)
- **Issue:** `git stash` + failed `stash pop` (binary file conflicts) reverted api/main.py to pre-edit state
- **Fix:** Rewrote api/main.py in full with all changes restored
- **Verification:** `uv run ruff check api/main.py` → OK; 94 tests pass

## Issues Encountered

| Issue | Resolution |
|-------|------------|
| Pre-existing `test_graph_builder` failure | Confirmed pre-existing via stash check; not introduced by this plan |
| Pre-existing mypy errors across project | Only `api/routes/frontend.py` checked clean; api/main.py has pre-existing slowapi type error (line 58) unchanged |

## Next Phase Readiness

**Ready:**
- `/dashboard` serving live with role-based JS views
- `#chat-placeholder` div present in index.html for Phase 07 widget mount
- `apiFetch()` wrapper handles auth — Phase 07 can reuse it for chat calls

**Concerns:**
- Visual ACs not confirmed — recommend manual smoke test before Phase 07 planning
- `output/dashboard.html` (old demo) still present — no cleanup needed per plan boundaries, but may confuse future devs

**Blockers:** None
