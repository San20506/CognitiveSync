# CognitiveSync

**Privacy-first enterprise burnout prediction using Graph Attention Networks.**

CognitiveSync ingests passive behavioral metadata from Microsoft Graph, Slack, and GitHub — meeting density, after-hours messaging, PR review load, focus time — and models burnout risk across your organisation using a Graph Attention Network (GAT). No surveys. No self-reporting. No PII ever stored.

Built for the Unisys Innovation Challenge 2026.

---

## About

Burnout is a lagging-indicator problem. By the time someone reports feeling burnt out, the organisational signals have been present for weeks. CognitiveSync turns those signals into a live risk graph — scored daily, explainable by feature attribution, propagated across reporting lines via cascade analysis.

**What makes it different:**

- **Graph-native modelling.** Burnout does not happen in isolation. A burnt-out manager cascades risk to their reports. The GAT encodes the org graph structure and propagates attention across edges — so the model understands *relationships*, not just individual feature vectors.
- **Explainability built in.** Every risk score ships with GAT attention weights: which features (after-hours commits? meeting overload? slow Slack response?) drove this person's score this week.
- **Privacy by architecture.** Raw API payloads never touch disk. All identifiers are pseudonymised to UUID v5 at ingestion. Managers cannot see individual scores — only aggregated team risk. GDPR Article 17 erasure and Article 30 audit endpoints are implemented.
- **Confidence intervals.** Monte Carlo Dropout gives every score a 95% confidence band. You know not just *what* the score is, but *how sure* the model is.

---

## Architecture

```
MS Graph / Slack / GitHub
        │
        ▼
┌─────────────────┐
│  Ingestion Layer │  Adapters → Anonymiser → Feature Extractor
│  (L1 + L2)      │  UUID v5 pseudonymisation, in-memory only
└────────┬────────┘
         │ 13 behavioural features per employee per window
         ▼
┌─────────────────┐
│  Graph Builder  │  NetworkX org graph + PyTorch Geometric Data
│  (L3)           │  Edges = reporting lines + interaction weights
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GAT Model      │  SmallBurnoutGAT: 10→64→16→1, dropout=0.1
│  + Cascade      │  MC Dropout (5 passes) → confidence intervals
│  (L3 + L4)      │  2-hop cascade propagation with decay factor
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  API + Output   │  FastAPI, JWT + RBAC, PostgreSQL persistence
│  (L4)           │  HR dashboard, MS Teams bot (Adaptive Cards)
└─────────────────┘
```

**Stack:** Python 3.11, FastAPI, PyTorch + PyTorch Geometric, SQLAlchemy 2.x async, PostgreSQL 16, NetworkX, uv.

---

## Demo

### Prerequisites

| Requirement | Check |
|------------|-------|
| Docker Desktop running | `docker ps` |
| Python 3.11+ | `python --version` |
| uv installed | `uv --version` |

### 1. Start the stack

```bash
git clone https://github.com/UnisysUIP/2026-CognitiveSync-Enterprise-Workforce-Burnout-Prediction-Intelligent-Workload-Redistribution.git
cd 2026-CognitiveSync-Enterprise-Workforce-Burnout-Prediction-Intelligent-Workload-Redistribution

cp .env.example .env        # configure your env (or use .env.local for local dev)
bash start_demo.sh
```

Wait for:
```
cognitivesync-api | Application startup complete.
```

### 2. Open the dashboard

```
Dashboard:  http://localhost:8000/dashboard
API docs:   http://localhost:8000/docs
```

### 3. Mint a demo token and trigger scoring

```bash
# Get an HR Admin token (demo mode only)
curl http://localhost:8000/demo/token?role=hr_admin

# Trigger a full pipeline run (CSV → graph → GNN → cascade → DB)
curl -X POST http://localhost:8000/api/v1/pipeline/run \
  -H "Authorization: Bearer <token>"
```

The pipeline runs asynchronously. Poll `/api/v1/pipeline/status/{run_id}` for progress. Once complete, the dashboard updates live.

### 4. Key demo flows

| Flow | Endpoint | Role |
|------|----------|------|
| View org burnout summary | `GET /api/v1/scores/summary` | HR Admin |
| Individual profile + attention weights | `GET /api/v1/profiles/{pseudo_id}` | HR Admin |
| Cascade risk propagation | `GET /api/v1/cascade/summary` | HR Admin |
| Team-level risk (no individual IDs) | `GET /api/v1/scores/team/{team_id}` | Manager |
| GDPR erasure | `DELETE /api/v1/audit/erasure/{pseudo_id}` | IT Admin |

---

## Model Quality

Trained on 120 synthetic employees, 5-fold cross-validation × 3 seeds = 15 folds.

| Metric | Value |
|--------|-------|
| Best fold AUC | **0.857** |
| Mean AUC (15 folds) | 0.670 ± 0.123 |
| Architecture | SmallBurnoutGAT (10→64→16→1) |
| Inference (120 nodes) | ~3 seconds |

The MVP validates the pipeline end-to-end on synthetic data. Production deployment replaces synthetic training data with real MS Graph / Slack / GitHub signals at 1k–5k node scale, where mean AUC is expected to rise above 0.80.

---

## Privacy & Compliance

- **Zero PII stored.** Email addresses and usernames are pseudonymised to UUID v5 at ingestion and discarded.
- **GDPR Article 17** — Right to erasure: `DELETE /api/v1/audit/erasure/{pseudo_id}` deletes all records across BurnoutScore, Employee, and EdgeSignal tables.
- **GDPR Article 30** — Processing activity log: `GET /api/v1/audit/events` (IT Admin only).
- **Data retention** — Configurable purge: `DELETE /api/v1/audit/retention/purge?retention_days=90`.
- **Role-based access control** — Managers cannot see individual scores or pseudo-IDs. HR Analysts see scores but not raw features. IT Admins manage infrastructure only.
- **Rate limiting** — Demo token endpoint: 10 req/min. Pipeline trigger: 5 req/min.

---

## API Documentation

Interactive API docs (Swagger UI) are available at `http://localhost:8000/docs` when the server is running. ReDoc is at `http://localhost:8000/redoc`.

Key endpoint groups:

| Group | Prefix | Description |
|-------|--------|-------------|
| Pipeline | `/api/v1/pipeline` | Trigger and monitor scoring runs |
| Scores | `/api/v1/scores` | Burnout scores by employee, team, org |
| Profiles | `/api/v1/profiles` | Individual profiles with attention weights |
| Cascade | `/api/v1/cascade` | Cascade risk propagation results |
| Enrollment | `/api/v1/employees` | Employee pseudonym registration |
| Audit | `/api/v1/audit` | GDPR erasure, retention, activity log |
| Demo | `/demo` | Token minting and dashboard (mock mode only) |

---

## Development

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -q

# Lint
uv run ruff check .

# Start API locally (requires running PostgreSQL)
uv run uvicorn api.main:app --reload
```

**Package manager:** uv (not pip).  
**Linter:** Ruff.  
**Type checker:** Mypy strict.  
**Test framework:** Pytest + pytest-asyncio.

---

## Team

Built by M Santhosh (Sandy) — System Architect, GNN Engineer, Integration Lead.  
Submitted to the Unisys Innovation Challenge 2026.
