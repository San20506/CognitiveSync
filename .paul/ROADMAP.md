# ROADMAP.md — CognitiveSync

## Milestone: v0.2 — AI Advisor + Production Dashboard

**Goal:** Replace demo HTML with a real production dashboard and add an AI chatbot advisor on all surfaces (dashboard + Teams bot). Same `/api/chat` backend powers both.

**Status:** In progress

---

## Phases

### Phase 05 — Chat API Backend
**Status:** Complete ✓
**Goal:** Build the `/api/chat` endpoint that accepts an employee/team context + question and returns an LLM-generated burnout advisory response. Powered by Azure OpenAI (configurable, falls back to Ollama for dev).

**Deliverables:**
- `api/routes/chat.py` — POST `/chat` with RBAC
- `config/settings.py` — LLM provider settings added
- Unit tests for chat route

---

### Phase 06 — Production Dashboard
**Status:** Complete ✓
**Completed:** 2026-05-11
**Goal:** Replace `output/dashboard.html` demo with a production-quality SPA served by FastAPI. Live data from API, role-based views (HR analyst / manager), real auth flow.

**Deliverables:**
- `output/dashboard/` — structured frontend (HTML/CSS/JS, no framework required)
- `output/dashboard/index.html` — entry point served by FastAPI
- `api/routes/frontend.py` — static file serving route
- Live data binding to `/api/scores`, `/api/employees`, `/api/cascade`

---

### Phase 07 — Chat Widget (Dashboard)
**Status:** Not started  
**Goal:** Embed a chat panel in the production dashboard. Floating button → side drawer → conversation UI. Calls `/api/chat`.

**Deliverables:**
- `output/dashboard/chat.js` — chat widget component
- Chat drawer UI integrated into dashboard
- Context-aware: passes current employee/team ID with each message

---

### Phase 08 — Teams Bot Chat Extension
**Status:** Not started  
**Goal:** Extend `output/teams_bot/bot.py` with conversational handling. HR managers can ask burnout questions directly in Teams. Calls same `/api/chat` endpoint.

**Deliverables:**
- `output/teams_bot/bot.py` — conversational message handler added
- `output/teams_bot/cards.py` — chat response Adaptive Card
- End-to-end: Teams message → bot → `/api/chat` → response card

---

## Completion criteria for v0.2
- [ ] `/api/chat` live with Azure OpenAI / Ollama backend
- [ ] Production dashboard serving live data
- [ ] Chat widget in dashboard
- [ ] Teams bot responding to free-text burnout questions
- [ ] All surfaces share one chat backend
