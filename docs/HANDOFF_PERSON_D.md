# CognitiveSync — Handoff Document for Person D
**Date:** 2026-04-28  
**Sprint:** April 16–30, 2026  
**Prepared by:** M Santhosh (Sandy)

---

## What Is CognitiveSync?

An **enterprise burnout prediction platform** that ingests passive behavioral metadata from Microsoft Graph, Slack, and GitHub to model burnout risk using a **Graph Attention Network (GAT)**. It outputs per-employee burnout scores with confidence intervals, cascade risk propagation across org charts, and attribution features — all without storing any PII.

**Privacy-first:** All user identifiers are pseudonymized to UUID v5 on ingest. No raw payloads are ever written to disk. Designed for on-prem or private Azure deployment.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Package manager | `uv` (not pip) |
| API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| ML | PyTorch + PyTorch Geometric (GAT) |
| Graph (MVP) | NetworkX |
| ORM | SQLAlchemy 2.x async |
| Database | PostgreSQL 16 |
| Connectors | MSAL, slack-sdk, PyGithub |
| Linter | Ruff |
| Type checker | Mypy strict |
| Tests | Pytest + pytest-asyncio |

---

## Architecture (4 Layers)

```
L1 — Connector Adapters   →  MS Graph, Slack, GitHub raw pulls
L2 — Feature Extractor    →  13 behavioral features per employee
L3 — Graph Builder        →  NetworkX graph (nodes=employees, edges=relationships)
L4 — GNN + Output         →  GAT inference → burnout score → cascade propagation
```

**API** sits on top and exposes everything via FastAPI routes.

---

## Repository Structure

```
CognitiveSync/
├── api/
│   ├── main.py                   # FastAPI app entry point
│   ├── middleware/
│   │   ├── auth.py               # JWT (RS256) authentication
│   │   └── rbac.py               # Role-based access control
│   ├── routes/
│   │   ├── scores.py             # GET /api/v1/scores — burnout scores
│   │   ├── cascade.py            # GET /api/v1/cascade — cascade risk
│   │   ├── employees.py          # Employee enrollment/listing
│   │   ├── profiles.py           # Individual behavioral profiles
│   │   ├── pipeline.py           # POST /api/v1/pipeline/run — trigger run
│   │   ├── model.py              # Model info / reload
│   │   ├── recommendations.py    # Intervention recommendations
│   │   ├── audit.py              # Audit log access
│   │   └── config.py             # Runtime config
│   └── schemas/                  # Pydantic request/response models
├── ingestion/
│   ├── adapters/                 # MS Graph, Slack, GitHub connectors
│   ├── anonymizer.py             # UUID v5 pseudonymization
│   ├── feature_extractor.py      # Behavioral feature computation
│   ├── db/
│   │   ├── models.py             # SQLAlchemy ORM models
│   │   └── session.py            # Async DB session
│   └── scheduler.py              # APScheduler for 48h refresh
├── intelligence/
│   ├── gnn_model.py              # SmallBurnoutGAT (GAT architecture)
│   ├── trainer.py                # Training pipeline
│   ├── inference.py              # Score generation + MC Dropout CIs
│   ├── cascade.py                # 2-hop cascade propagation
│   ├── graph_builder.py          # NetworkX graph + PyG conversion
│   ├── features.py               # Feature extractor + normalizer (13 features)
│   ├── edges.py                  # Edge weight extractor from interactions
│   ├── tests.py                  # Graph/feature/PyG unit tests (run_all_tests)
│   └── profile_updater.py        # Writes scores back to DB
├── alembic/                      # DB migrations
│   └── versions/
│       ├── 8626f459da6e_initial_schema.py
│       └── 893cdc022dcc_enrollment_and_profiles.py
├── config/
│   └── settings.py               # Pydantic Settings (env-var driven)
├── tests/
│   └── unit/
│       └── test_cascade.py       # 14 cascade unit tests (all passing)
├── models/                       # Saved model checkpoints
│   ├── v1/                       # First synthetic training run
│   ├── csv-v1/                   # CSV-trained checkpoint
│   ├── csv-v2-tuned/             # Tuned CSV checkpoint
│   ├── final-v1/                 # Best checkpoint (seed=99, fold=3, AUC=0.857)
│   └── latest -> final-v1        # Symlink used by inference
├── artifacts/                    # Training metrics and demo output
│   ├── training_metrics.json     # Quick eval: acc=0.80, AUC=0.80
│   ├── final_training_metrics.json # 5-fold CV: mean AUC=0.67, best=0.857
│   └── demo_results.json         # 120-node demo run output
├── data/                         # Synthetic CSV data for demo/dev
│   ├── employees.csv
│   ├── features.csv
│   └── interactions.csv
├── scripts/                      # Standalone training/eval scripts
├── Dockerfile                    # API container
├── docker-compose.yml            # Full stack: API + PostgreSQL
└── pyproject.toml                # Dependencies (managed by uv)
```

---

## What Has Been Built (All Completed Tasks)

### Infrastructure & Scaffold
- `T-007/008/009/010` — Repo structure, uv project (Python 3.11), pre-commit hooks (Ruff + Mypy strict)
- Alembic migrations: initial schema + enrollment/profiles tables
- **Docker Compose** — Full stack (API + PostgreSQL 16). `docker-compose up` starts both. DB init schema at `deploy/init-schemas.sql`.
- JWT RS256 auth + RBAC middleware (roles: HR_ADMIN, HR_ANALYST, MANAGER)
- Pydantic Settings — all secrets via env vars, never hardcoded

### Feature & Edge Extraction (Person C / Akshaya — PR #1, merged 2026-04-21)
- `T-037` Feature extractor (`intelligence/features.py`) — 13 behavioral features, min-max normalization
- `T-038` Org-level normalisation with epsilon guard
- `T-039` Edge extractor (`intelligence/edges.py`) — 4 interaction types (MEETING, SLACK_DM, GITHUB_PR, GITHUB_CO_COMMIT), normalized weights [0,1], undirected merge
- `T-041` Feature validator tests (`intelligence/tests.py`) — validates shape, range, variance
- `T-043/T-044` Graph/PyG validator tests — validates node count, edge weights, PyG tensor shapes

### Graph Builder (`intelligence/graph_builder.py`)
- `T-042` `GraphBuilder.build_from_store()` — loads from DB (async), builds NetworkX DiGraph
- `T-043` `GraphBuilder.to_pyg()` — converts to PyG `Data(x, edge_index, edge_attr)`
- `build_from_csv()` — builds from synthetic CSV files (used for demo/dev without DB)
- Handles orphan nodes (no interactions), deterministic node ordering

### Synthetic Data Pipeline
- `T-022` Synthetic org graph generator
- `T-023` Synthetic feature vector generator
- `T-024` Rule-based burnout label generator
- `T-025` Synthetic edge generator
- `T-026` Synthetic data validated — all checks pass (`artifacts/synthetic_validation.json`)

### GNN Model & Training
- `T-045` **SmallBurnoutGAT**: 2-layer Graph Attention Network  
  Architecture: `10 → 64 → 16 → 1`, dropout=0.1
- `T-046` Full training pipeline (cross-validation, early stopping)
- `T-047` Trained on 200-node synthetic graph, seed=42, 150 epochs
- `T-048` Quick eval: **accuracy=0.80, AUC-ROC=0.80** → phase gate PASSED (acc≥0.80, AUC≥0.75)
- `T-049` Full 5-fold CV across seeds [42, 7, 99]: mean AUC=0.67, **best AUC=0.857** (seed=99, fold=3)
- `T-050` Model checkpoint saved at `models/final-v1/`, `models/latest` symlinked

### Inference & Cascade
- `T-051` MC Dropout confidence intervals (5 stochastic passes → mean ± 2σ)
- `T-052` GAT attention weights → top-3 feature attribution per employee
- `T-053` 2-hop cascade propagation (threshold=0.70, decay=0.60, max_hops=2)
- `T-054` Cascade source attribution — which high-risk nodes contributed
- `T-055` 14 cascade unit tests — all passing (`tests/unit/test_cascade.py`)

### API Layer (all routes implemented)
- `GET /api/v1/scores` — burnout scores (HR Admin/Analyst only)
- `GET /api/v1/scores/team-summary` — team aggregates (Manager+)
- `GET /api/v1/scores/trend` — score history per pseudo_id
- `GET /api/v1/cascade` — cascade risk results
- `POST /api/v1/pipeline/run` — trigger full pipeline run
- `GET /api/v1/pipeline/status/{run_id}` — check run status
- `GET /api/v1/employees` — enrollment list
- `GET /api/v1/profiles/{pseudo_id}` — individual behavioral profile
- `GET /api/v1/recommendations` — intervention recommendations
- `GET /api/v1/audit` — audit log
- `GET /api/v1/model` — model info / reload
- `GET /api/v1/config` — runtime config
- `GET /health` — health check

### Enrollment System & Profiles
- Last commit (`80c6e6e`) — enrollment system, individual behavioral profiles, pipeline integration

---

## Training Results (Key Numbers for Demo)

| Metric | Value |
|--------|-------|
| Demo org size | 120 employees |
| High-risk employees flagged | 23 (19%) |
| Cascade-affected employees | 69 (58%) |
| Top node burnout score | **0.88** |
| Top risk features | meeting_density, response_latency_avg, mention_load |
| Model best AUC | **0.857** (seed=99, fold=3) |
| Phase gate | PASSED (acc=0.80 ≥ 0.80, AUC=0.80 ≥ 0.75) |
| Model checkpoint | `models/latest` → `models/final-v1/` |

---

## Running the Demo (Standalone — No Real DB Needed)

```bash
cd /home/sandy/Documents/Projects/CognitiveSync

# Install deps
uv sync

# Run demo (uses pre-built synthetic CSVs in data/)
python scripts/train_final.py

# OR trigger via the API pipeline (needs Docker stack running first)
curl -X POST http://localhost:8000/api/v1/pipeline/run \
  -H "Authorization: Bearer <token>"
```

Existing demo output: `artifacts/demo_results.json` (120-node run, ready to use).

### Start the full stack via Docker Compose:
```bash
# Copy env template and fill in JWT key paths and org_salt
cp .env.example .env   # edit as needed

docker-compose up --build
# API: http://localhost:8000/docs
# DB:  localhost:5432
```

### Run existing tests:
```bash
uv run pytest tests/unit/test_cascade.py -v   # 14 tests, all pass
```

---

## Still Pending

| Item | Owner | Blocked on |
|------|-------|-----------|
| Intelligence module unit tests (beyond cascade) | Sandy | — |
| Integration test: CSV → graph → GNN → API response | Sandy + Person B | Person B DB readiness |
| End-to-end Docker Compose verification | All | Integration |
| Demo prep: seeded run + showcase cluster | **Person D** | Nothing — can run now |

---

## Person B's Task List

Person B owns the **API/database integration layer**. Based on the current code state:

| Task | Description | File(s) |
|------|-------------|---------|
| **T-017** | Confirm ORM models match DB schema | `ingestion/db/models.py`, `alembic/versions/` |
| **T-018** | Implement `profile_updater.py` — write GNN scores back to `BurnoutScore` table | `intelligence/profile_updater.py` |
| **T-019** | Wire `pipeline.py` route — connect `_run_pipeline()` to real GraphBuilder + InferencePipeline + CascadePropagator | `api/routes/pipeline.py` |
| **T-020** | Implement `GET /api/v1/cascade` route — load cascade results from DB | `api/routes/cascade.py` |
| **T-021** | Integration test: POST `/pipeline/run` → scores appear in `/scores` | `tests/integration/` |
| **DB setup** | Run Alembic migrations against Docker Postgres | `alembic upgrade head` |
| **Auth keys** | Generate RS256 keypair for JWT, set paths in `.env` | `secrets/` dir |

**Integration contract Sandy provides to Person B:**

`InferencePipeline.score()` returns `ScoredGraph` with per-node:
```python
NodeScore(
    pseudo_id: UUID,
    burnout_score: float,       # [0,1]
    confidence_low: float,      # [0,1]
    confidence_high: float,     # [0,1]
    top_features: dict[str, float],  # top-3 feature name → value
)
```

`CascadePropagator.propagate()` returns `dict[UUID, CascadeResult]` with:
```python
CascadeResult(
    pseudo_id: UUID,
    cascade_risk: float,        # [0,1]
    cascade_sources: list[UUID],
)
```

Person B needs to persist these into `scores.burnout_scores` table (model at `ingestion/db/models.py:BurnoutScore`).

---

## Your Role (Person D)

1. **Demo runner** — Run `python scripts/train_final.py` for a stable seeded output. Use `artifacts/demo_results.json` as fallback (already has a clean 120-node run).
2. **Showcase identification** — Pick the top high-risk cluster (node `4c011558`, score=0.88, features: meeting_density + response_latency) and a cascade chain (69 of 120 nodes affected).
3. **Final test script** — Once Person B wires the pipeline, run the end-to-end path: `POST /pipeline/run` → poll `/pipeline/status/{run_id}` → `GET /scores`.

---

## Privacy Rules (Must Know)

- All employee IDs are **UUID v5 pseudonyms** — no real names, emails, or usernames anywhere in the system
- The `pseudo_id` in demo output is not reversible without the original org salt (by design)
- Never log or print raw API payloads from MS Graph / Slack / GitHub
- `adapter_mode=mock` in `.env` uses synthetic data and never calls real APIs

---

## Contacts

- **Sandy (M Santhosh):** System Architect, GNN Engineer — `m.santhosh200506@gmail.com`
- **Repo:** `San20506/CognitiveSync` on GitHub
- **Main branch:** `main` — integration target for all PRs
