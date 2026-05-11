# STATE.md — CognitiveSync

## Current Position

Milestone: v0.2 — AI Advisor + Production Dashboard
Phase: 7 of 8 (Chat Widget — Dashboard) — Not started
Plan: Not started
Status: Ready to plan Phase 07
Last activity: 2026-05-11 — Phase 06 complete, transitioned to Phase 07

Progress:
- Milestone: [████░░░░░░] 25%
- Phase 6: [██████████] 100%

## Loop Position

Current loop state:
```
PLAN ──▶ APPLY ──▶ UNIFY
  ✓        ✓        ✓     [Loop complete — ready for next PLAN]
```

## Session Continuity

Last session: 2026-05-11
Stopped at: Phase 06 complete — Production dashboard shipped
Next action: /paul:plan for Phase 07 (Chat Widget in dashboard)
Resume file: .paul/ROADMAP.md

## Decisions

| ID | Decision | Rationale |
|----|----------|-----------|
| D-01 | Path 1: Full dashboard (not Teams-only) | User confirmed: production web UI is the canonical output surface |
| D-02 | All surfaces share one `/api/chat` backend | DRY — dashboard widget + Teams bot both call same endpoint |
| D-03 | LLM: Azure OpenAI (prod) / Ollama (dev) | No external egress constraint; configurable via env var |
| D-04 | Start with Chat API backend first | Unblocks both dashboard widget and Teams bot in parallel |
| D-05 | Vanilla JS, no framework for dashboard | Offline-capable on private Azure; no npm/build step needed |
| D-06 | StaticFiles mount conditional on dir existence | Prevents startup crash before frontend present |

## Accumulated Context

### Concerns
- Visual AC verification was bypassed for Phase 06 — recommend manual smoke test of /dashboard before Phase 07 planning
- Pre-existing mypy errors in ingestion/adapters/ and intelligence/ (not introduced by v0.2 work)
- Pre-existing test failure: test_graph_builder.py::TestBuildFromCSV::test_node_count_equals_employee_count
