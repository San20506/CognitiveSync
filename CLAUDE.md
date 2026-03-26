# CognitiveSync — Claude Code Context

## Identity
- **What it is:** Enterprise burnout prediction platform that ingests passive behavioral metadata from MS Graph, Slack, and GitHub to model burnout risk using Graph Attention Networks (GAT).
- **Stack:** Python 3.11+, uv (package manager), ruff (linter/formatter), mypy (type checker), FastAPI + Uvicorn (API), Pydantic v2 (validation), PyTorch + PyTorch Geometric (ML/GNN), SQLAlchemy 2.x async (ORM), PostgreSQL 16 (DB), NetworkX (Graph MVP), MSAL/slack-sdk/PyGithub (Connectors).
- **Runtime target:** On-prem or private Azure deployment (Privacy-first).
- **Repo root:** /home/sandy/Projects/CognitiveSync

---

## Current Status
- **SDLC Phase:** Phase 3 — Scaffold complete, moving into Phase 4A construction.
- **Last completed milestone:** Scaffold complete (inferred from CLAUDE.md stage).
- **Active task:** Phase 4A construction initialization.
- **Blocked on:** nothing detected

---

## Architecture Snapshot
- **4-Layer Architecture**: Enforced separation: Connector adapters (L1), Feature extractor (L2), Graph builder (L3), Output layer (L4).
- **Privacy-First Design**: Raw API payloads never written to disk; PII never stored; UUID v5 pseudonymization; Manager role score masking.
- **GNN Model (GAT)**: 2-layer Graph Attention Network with MC Dropout for confidence and attention weights for attribution.
- **Decoupled Scoring/Cascade**: 2-hop cascade propagation with configurable decay and thresholds.

---

## File Map
- **api/**: FastAPI application, routes, middleware (RBAC, JWT), and schemas.
- **ingestion/**: Adapters (MS Graph, Slack, GitHub), anonymizer, and feature extractor.
- **intelligence/**: Graph construction (NetworkX), GAT model (PyTorch Geometric), trainer, and inference scoring.
- **output/**: Power BI connector and MS Teams bot (Adaptive Cards).
- **config/**: Settings management via Pydantic Settings.
- **docs/**: Project documentation (PRD, PID, ARCHITECTURE, TECH_STACK, WORKFLOWS, TASK_PLAN).
- **tests/**: Unit and integration test suites.

---

## Sub-agents / Modules
| Name | Role | Status |
|------|------|--------|
| Ingestion | Connectors & Feature Extraction | [~] In Progress |
| Intelligence | GNN Modeling & Cascade Prediction | [~] In Progress |
| API/Output | HR Dashboards & Teams Bot | [~] In Progress |

---

## DO NOTs
- **Never use pip install**: Use `uv sync` or `uv add` as the primary package manager.
- **Never store PII**: Email, usernames, and user IDs must be pseudonymized to UUID v5 immediately.
- **No sync ORM calls**: All database interactions must use SQLAlchemy 2.x async sessions.
- **No external data egress**: Do not add external API calls or telemetry that exits the organization boundary.
- **No raw payloads on disk**: Behavioral metadata must be processed in-memory and discarded after anonymization.
- **Never suppress type errors**: Avoid `as any`, `@ts-ignore`, or `@ts-expect-error`.

---

## Coding Conventions
- **Linter/Formatter**: Ruff (zero errors required before completion).
- **Type Checking**: Mypy strict mode (must pass 100%).
- **Async Style**: Native async/await for all I/O, database, and API operations.
- **Secrets**: Load exclusively via environment variables in `config/settings.py`.
- **Testing**: Pytest with pytest-asyncio; 80%+ coverage target.
- **Logging**: Use standard Python `logging` module; no `print()` statements.

---

## Phase Gate Rule
- **T-XXX Protocol**: Read `docs/TASK_PLAN.md` -> Check dependencies -> Implement full (no stubs) -> Run Ruff/Mypy/Pytest -> Mark complete.

---

## Open Questions / Deferred Decisions
- **Neo4j Migration**: Deferred to V2 roadmap (currently using NetworkX for MVP).
- **Wearable Integration (HRV/Sleep)**: Placeholder for V2 (currently using neutral baseline 0.5 imputation).

---

## Session Startup Checklist
1. Read this file fully
2. Read docs/ARCHITECTURE.md if it exists (Note: Project docs are currently .docx files in root)
3. Confirm active task with user before touching any code
4. State current phase and last milestone in first response
