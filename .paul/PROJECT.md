# PROJECT.md — CognitiveSync

## What it is
Privacy-first enterprise burnout prediction platform. Ingests passive behavioral metadata (MS Graph, Slack, GitHub) and models burnout risk using Graph Attention Networks (GAT). No surveys. No PII stored.

## Value proposition
Turns lagging burnout signals into a live, explainable risk graph — scored daily, propagated across reporting lines. Confidence intervals via MC Dropout. GDPR-compliant.

## Stack
- Python 3.11+, uv, ruff, mypy
- FastAPI + Uvicorn (API)
- PyTorch + PyTorch Geometric (GNN)
- SQLAlchemy 2.x async + PostgreSQL 16
- NetworkX (graph MVP)
- Vanilla JS SPA (no framework) for dashboard

## Architecture (4-layer)
- L1: Connector adapters (MS Graph, Slack, GitHub)
- L2: Feature extractor + anonymiser (UUID v5 pseudonymisation)
- L3: Graph builder + GAT model + cascade scorer
- L4: API + output (dashboard, Teams bot, Power BI)

## Constraints
- No pip install — use `uv add` / `uv sync`
- No PII storage — UUID v5 at ingestion boundary
- No sync ORM — async SQLAlchemy only
- No external data egress — on-prem / private Azure deployment
- No raw payloads on disk — process in-memory, discard after anonymisation
- Ruff zero errors + mypy strict must pass before any task is complete
- 80%+ pytest coverage target
- No npm/bundler for frontend — vanilla JS only for offline-capable deployment

## Current output surfaces
- `output/dashboard/` — production SPA at /dashboard (Phase 06) ✓
- `output/dashboard.html` — legacy demo HTML (kept for reference)
- `output/teams_bot/` — Teams bot scaffolded (bot.py, cards.py)
- `output/powerbi_connector.py` — Power BI connector built
- `api/routes/chat.py` — POST /api/v1/chat/ with RBAC (Phase 05) ✓

## Key Decisions

| Decision | Phase | Rationale |
|----------|-------|-----------|
| LLM: Azure OpenAI (prod) / Ollama (dev) | 05 | No external egress; configurable via env var |
| Stateless LLM calls (no conversation history) | 05 | Privacy constraint — no chat logs stored |
| Vanilla JS dashboard, no framework | 06 | Offline-capable on private Azure; no build step |
| StaticFiles mount conditional on dir existence | 06 | Prevents startup crash before frontend is present |
| Promise.allSettled for parallel API fetches | 06 | Partial data shown rather than full page failure |

## Deployment target
On-prem or private Azure. No SaaS. LLM must be Azure OpenAI (private endpoint) or local Ollama.

---
*Last updated: 2026-05-11 after Phase 06*
