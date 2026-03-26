# Implementation Plan: CognitiveSync Full Build
**Source of Truth**: CognitiveSync_PRD.docx + ARCHITECTURE.docx + TASK_PLAN.docx + WORKFLOWS.docx
**Developer**: M Santhosh (Solo)
**Target**: Unisys Innovation Program Y17
**Current Status**: Phase 3 → Phase 4A (scaffold reportedly complete per CLAUDE.md, T-007:T-016 still show [ ] in TASK_PLAN)
**Generated**: 2026-03-24

---

## Task Type
- [x] Backend (Python, FastAPI, PyTorch, PostgreSQL)
- [ ] Frontend (Power BI — external tool)
- [ ] Fullstack

---

## Technical Solution

4-layer privacy-first pipeline:
1. **Connector Adapters** (MS Graph / Slack / GitHub / Mock) → raw signals in-memory only
2. **Privacy & Ingestion** (Anonymizer + Feature Extractor + APScheduler)
3. **Intelligence Engine** (NetworkX Graph Builder + GAT GNN + Cascade Propagator)
4. **Output Layer** (FastAPI REST + Power BI connector + Teams Bot + RBAC)

Stack: Python 3.11, uv, FastAPI, Pydantic v2, SQLAlchemy 2.x async, PostgreSQL 16, PyTorch + PyG, NetworkX, APScheduler, Docker Compose.

---

## Implementation Steps

### PHASE 3 — Design & Scaffold (T-007 → T-016) [~17h]

#### Step 1: Repo Scaffold (T-007, T-008)
Create the full directory tree:
```
cognitiveSync/
├── api/
│   ├── routes/         # __init__.py, scores.py, cascade.py, pipeline.py, config.py, audit.py
│   ├── middleware/     # auth.py (JWT), rbac.py
│   ├── schemas/        # request.py, response.py
│   └── main.py
├── ingestion/
│   ├── adapters/       # base.py, msgraph.py, slack.py, github.py, mock.py
│   ├── anonymizer.py
│   ├── feature_extractor.py
│   └── scheduler.py
├── intelligence/
│   ├── graph_builder.py
│   ├── gnn_model.py
│   ├── trainer.py
│   ├── inference.py
│   └── cascade.py
├── output/
│   ├── powerbi_connector.py
│   └── teams_bot/
│       ├── bot.py
│       └── cards.py
├── models/             # versioned .pt checkpoints
├── data/               # synthetic generators (dev only)
├── config/
│   └── settings.py
├── tests/
│   ├── unit/
│   └── integration/
├── deploy/
│   ├── docker-compose.yml
│   └── helm/
├── docs/               # SDLC .docx files
├── .python-version     # 3.11
├── pyproject.toml
└── README.md
```
All leaf `.py` files start as empty module stubs (no logic yet).

**Deliverable**: All dirs and `__init__.py` stubs exist; `git init` + initial commit.

---

#### Step 2: uv Project Setup (T-009)
`pyproject.toml` content:
```toml
[project]
name = "cognitivesync"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.111",
  "uvicorn[standard]>=0.29",
  "pydantic>=2.7",
  "pydantic-settings>=2.2",
  "sqlalchemy[asyncio]>=2.0",
  "asyncpg>=0.29",
  "alembic>=1.13",
  "python-jose[cryptography]>=3.3",
  "passlib[bcrypt]>=1.7",
  "httpx>=0.27",
  "msal>=1.28",
  "slack-sdk>=3.27",
  "PyGithub>=2.3",
  "networkx>=3.3",
  "torch>=2.2",
  "torch-geometric>=2.5",
  "apscheduler>=3.10",
  "cryptography>=42",
  "botbuilder-core>=4.16",
  "botbuilder-schema>=4.16",
]

[project.optional-dependencies]
dev = [
  "pytest>=8",
  "pytest-asyncio>=0.23",
  "pytest-cov>=5",
  "respx>=0.21",
  "ruff>=0.4",
  "mypy>=1.10",
  "pre-commit>=3.7",
]

[tool.ruff]
line-length = 100
select = ["E","F","I","N","UP","ANN"]

[tool.mypy]
strict = true
python_version = "3.11"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```
Run `uv sync` to generate `uv.lock`.

**Deliverable**: `pyproject.toml`, `uv.lock`, `.python-version` (content: `3.11`).

---

#### Step 3: Pre-commit Hooks (T-010)
`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic>=2.7]
```
Run `uv run pre-commit install`.

**Deliverable**: Hooks installed; `uv run pre-commit run --all-files` passes on stubs.

---

#### Step 4: Docker Compose Skeleton (T-011)
`deploy/docker-compose.yml` services:
```yaml
services:
  db:          # postgres:16-alpine, port 5432, named volume
  api:         # python:3.11-slim, port 8000, depends: db
  scheduler:   # same image, runs scheduler entrypoint, depends: db
  bot:         # same image, port 3978, depends: api
  proxy:       # nginx:1.25-alpine, port 443→8000 TLS termination
  audit-db:    # postgres:16-alpine (separate audit schema), port 5433
```
Environment variables injected from `.env` (never secrets in compose file).
All services on a private `cognitivesync_net` bridge network.

**Deliverable**: `docker-compose.yml`; `docker compose config` validates without errors.

---

#### Step 5: PostgreSQL Schema Design (T-012)
Four schemas in one PostgreSQL instance:
```sql
-- Schema: features
CREATE TABLE features.feature_vectors (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pseudo_id   UUID NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    window_end   TIMESTAMPTZ NOT NULL,
    feature_json JSONB NOT NULL,   -- 13-dim vector as {key: float}
    is_imputed   BOOLEAN DEFAULT FALSE,
    data_completeness FLOAT,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    INDEX (pseudo_id, window_start)
);

-- Schema: scores
CREATE TABLE scores.burnout_scores (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id          UUID NOT NULL,
    pseudo_id       UUID NOT NULL,
    burnout_score   FLOAT NOT NULL CHECK (burnout_score BETWEEN 0 AND 1),
    confidence_low  FLOAT,
    confidence_high FLOAT,
    cascade_risk    FLOAT CHECK (cascade_risk BETWEEN 0 AND 1),
    cascade_sources JSONB,   -- list of source pseudo_ids
    top_features    JSONB,   -- {feature_name: attention_weight}
    team_id         UUID,
    window_end      TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Schema: audit
CREATE TABLE audit.events (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type  TEXT NOT NULL,
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Schema: config
CREATE TABLE config.orgs (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_name     TEXT NOT NULL,
    timezone     TEXT DEFAULT 'UTC',
    work_hours_start TIME DEFAULT '09:00',
    work_hours_end   TIME DEFAULT '18:00',
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Encrypted ID mapping: stored in separate encrypted volume (NOT PostgreSQL)
-- File: /vault/id_mapping.enc  (AES-256-GCM)
```

**Deliverable**: Schema DDL SQL file at `deploy/schema.sql`.

---

#### Step 6: Pydantic v2 Schema Design (T-013)
`api/schemas/response.py` key models:
```python
class BurnoutScoreResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    pseudo_id: UUID
    burnout_score: float = Field(ge=0.0, le=1.0)
    confidence_interval: tuple[float, float]
    cascade_risk: float = Field(ge=0.0, le=1.0)
    top_features: dict[str, float]
    window_end: datetime

class CascadeMapResponse(BaseModel):
    nodes: list[CascadeNodeResponse]
    edges: list[CascadeEdgeResponse]
    high_risk_sources: list[UUID]

class PipelineRunResponse(BaseModel):
    run_id: UUID
    status: Literal["started","running","completed","failed"]
    node_count: int | None
    duration_seconds: float | None
```

**Deliverable**: All Pydantic schemas in `api/schemas/`; mypy passes.

---

#### Step 7: Synthetic Data Spec (T-014) + GAT Architecture (T-015)
Synthetic generator spec (in `data/`):
- `SyntheticOrgGraph(n_employees: int, burnout_fraction: float = 0.15)`
- Feature distribution per burnout label (high-risk = elevated after-hours, high meetings, low focus)
- 13-dim feature vector with realistic correlations (NumPy RNG with seed)
- Edge density ~15% of all possible pairs (org collaboration typical)
- Burnout labels: rule-based from feature thresholds (3+ features in 90th percentile → burnout=1)

GAT Architecture spec:
```
Input: (N×13 feature tensor, edge_index COO tensor, edge_attr weights)
Layer 1: GATConv(in=13, out=64, heads=4, dropout=0.3, concat=True) → (N×256)
         + ELU activation + BatchNorm
Layer 2: GATConv(in=256, out=32, heads=2, dropout=0.3, concat=False) → (N×32)
         + ELU activation
Output:  Linear(32 → 1) + Sigmoid → burnout_score per node [0,1]
MC Dropout: 5 stochastic forward passes, std as confidence proxy
```

**Deliverable**: `data/synthetic.py` (spec only, impl in Phase 4A); GAT dims documented.

---

#### Step 8: Design Review (T-016)
Verify every design decision traces to a PRD requirement:
- [ ] All 6 MVP in-scope items from PRD §5.1 covered
- [ ] All 8 non-functional requirements from PRD §7 addressed
- [ ] All 5 technical constraints from PRD §8 enforced in design
- [ ] Success criteria from PRD §10 achievable with this plan

**Deliverable**: `docs/DESIGN_REVIEW.md` with traceability matrix.

---

### PHASE 4A — Core Pipeline (T-017 → T-041) [~66h, Weeks 3-6]

#### Step 9: SQLAlchemy Async ORM Models (T-017)
`config/settings.py`:
```python
class Settings(BaseSettings):
    database_url: PostgresDsn
    audit_database_url: PostgresDsn
    jwt_secret_key: str
    jwt_algorithm: str = "RS256"
    org_salt: str           # pseudonymization salt
    vault_key: str          # AES-256 master key
    refresh_interval_hours: int = 48
    alert_threshold: float = 0.75
    cascade_threshold: float = 0.70
    decay_factor: float = 0.60
    max_hops: int = 2
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
```

`ingestion/db/models.py` (SQLAlchemy 2.x async):
```python
class FeatureVector(Base):
    __tablename__ = "feature_vectors"
    __table_args__ = {"schema": "features"}
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    pseudo_id: Mapped[UUID] = mapped_column(index=True)
    window_start: Mapped[datetime]
    window_end: Mapped[datetime]
    feature_json: Mapped[dict] = mapped_column(JSONB)
    is_imputed: Mapped[bool] = mapped_column(default=False)
    data_completeness: Mapped[float | None]
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

class BurnoutScore(Base):
    __tablename__ = "burnout_scores"
    __table_args__ = {"schema": "scores"}
    # ... all fields per schema design above
```

**Deliverable**: `ingestion/db/models.py`, `api/db/models.py`; mypy strict passes.

---

#### Step 10: Alembic Migrations (T-018)
```bash
uv run alembic init alembic
# Edit alembic/env.py for async engine + multi-schema support
uv run alembic revision --autogenerate -m "initial_schema"
uv run alembic upgrade head
```

**Deliverable**: `alembic/` directory; migrations apply clean on fresh DB.

---

#### Step 11: Secrets Loader (T-019)
`config/vault.py`:
```python
def load_secret(key: str) -> str:
    """Read from environment variable; raise ConfigError if missing."""
    val = os.environ.get(key)
    if val is None:
        raise ConfigError(f"Required secret '{key}' not found in environment")
    return val
```
No secrets ever in source code. `.env.example` documents all required vars with placeholder values.

**Deliverable**: `config/vault.py`; `config/settings.py` uses it; `.env.example`.

---

#### Step 12: JWT Auth Middleware (T-020)
`api/middleware/auth.py`:
```python
async def verify_jwt(token: str = Depends(oauth2_scheme)) -> TokenPayload:
    try:
        payload = jose.jwt.decode(token, PUBLIC_KEY, algorithms=["RS256"])
        return TokenPayload(**payload)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```
RS256 — private key signs (service), public key verifies (API).
Claims: `sub` (pseudo_id), `role` (hr_admin|hr_analyst|manager|it_admin), `org_id`, `exp`.

**Deliverable**: `api/middleware/auth.py` with full JWT decode/validate logic.

---

#### Step 13: RBAC Middleware (T-021)
`api/middleware/rbac.py`:
```python
ENDPOINT_ROLES = {
    "/api/v1/scores": {"hr_admin", "hr_analyst"},
    "/api/v1/cascade-map": {"hr_admin", "hr_analyst"},
    "/api/v1/recommendations": {"hr_admin", "manager"},
    "/api/v1/pipeline/run": {"it_admin"},
    "/api/v1/model/retrain": {"it_admin"},
}

def require_role(*roles: str):
    """FastAPI dependency — raises 403 if token role not in allowed set."""
    async def _check(token: TokenPayload = Depends(verify_jwt)):
        if token.role not in roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return token
    return _check
```
CRITICAL: Manager role MUST NOT receive individual pseudo_id scores — only team aggregates.

**Deliverable**: `api/middleware/rbac.py`; RBAC tested at API layer (not just UI).

---

#### Step 14: Synthetic Data Generator (T-022 → T-026)
`data/synthetic.py`:
```python
class SyntheticOrgGenerator:
    def __init__(self, n: int = 100, burnout_frac: float = 0.15, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_employees(self) -> list[EmployeeNode]:
        # Assign pseudo_ids (UUID4 for synthetic, no real identifiers)
        # Assign team_id (5-10 teams), manager_id
        ...

    def generate_feature_vectors(self, employees) -> dict[UUID, np.ndarray]:
        # 13 features, shape (N, 13)
        # Burnout-positive nodes: elevated after_hours_*, high meeting_density
        # Distributions: normal with different mu per label
        ...

    def generate_edges(self, employees) -> list[tuple[UUID, UUID, float]]:
        # Interaction edges: within-team density 0.4, cross-team 0.05
        # Edge weight = interaction_score in [0.1, 1.0]
        ...

    def generate_labels(self, features: np.ndarray) -> np.ndarray:
        # Rule: burnout=1 if 3+ features in 90th percentile
        ...
```

**Deliverable**: `data/synthetic.py`; generates valid 100-node graph; spot-check distributions pass (T-026).

---

#### Step 15: Connector Adapters (T-027 → T-031)

**Base Adapter** (`ingestion/adapters/base.py`):
```python
class BaseAdapter(ABC):
    @abstractmethod
    async def fetch_signals(
        self, window_start: datetime, window_end: datetime
    ) -> dict[str, RawSignals]:
        """Returns {user_identifier: RawSignals} — PII still present at this stage."""
        ...
```

**MS Graph Adapter** (`ingestion/adapters/msgraph.py`):
```python
class MSGraphAdapter(BaseAdapter):
    def __init__(self, client_id: str, client_secret: str, tenant_id: str):
        self.msal_app = ConfidentialClientApplication(...)

    async def fetch_signals(self, window_start, window_end):
        token = self._get_token()  # MSAL client credentials
        users = await self._get_users(token)
        for user in users:
            calendar = await self._get_calendar_events(token, user.id, window_start, window_end)
            signals[user.mail] = RawSignals(
                meeting_density=len(calendar) / days,
                after_hours_meetings=...,
                focus_blocks=...,
                response_latency=...,
            )
        return signals
```

**Slack Adapter** (`ingestion/adapters/slack.py`):
- `AsyncWebClient` with bot token
- Fetch channel history for all members in window
- Compute: message_volume, after_hours_messages, avg_response_time, mention_count

**GitHub Adapter** (`ingestion/adapters/github.py`):
- `PyGithub` org-level client
- Fetch commits, PRs, issues in window
- Compute: commit_frequency, after_hours_commits, pr_review_load, context_switch_rate

**Mock Adapter** (`ingestion/adapters/mock.py`):
```python
class MockAdapter(BaseAdapter):
    def __init__(self, synthetic_gen: SyntheticOrgGenerator):
        self.gen = synthetic_gen

    async def fetch_signals(self, window_start, window_end):
        # Returns synthetic data — used for ALL testing and demo
        return self.gen.generate_raw_signals()
```

**Unit Tests** (T-031): Use `respx` to mock HTTP; assert signal shapes correct; assert no network calls in tests.

**Deliverable**: 4 adapters + unit tests; all pass `ruff` + `mypy`.

---

#### Step 16: Anonymization Engine (T-032 → T-036)
`ingestion/anonymizer.py`:
```python
class Anonymizer:
    def __init__(self, org_salt: str, vault_key: str):
        self._salt = org_salt.encode()
        self._cipher_key = derive_key(vault_key)  # PBKDF2HMAC

    def pseudonymize(self, identifier: str) -> UUID:
        """SHA-256(identifier + salt) → UUID v5. Deterministic."""
        digest = hashlib.sha256(identifier.encode() + self._salt).digest()
        return uuid.UUID(bytes=digest[:16], version=5)

    def anonymize_signals(
        self, raw: dict[str, RawSignals]
    ) -> dict[UUID, AnonymizedSignals]:
        result: dict[UUID, AnonymizedSignals] = {}
        for identifier, signals in raw.items():
            pseudo_id = self.pseudonymize(identifier)
            self._upsert_mapping(pseudo_id, identifier)  # encrypted volume write
            # Drop identifier from memory immediately
            del identifier
            result[pseudo_id] = AnonymizedSignals.from_raw(signals)
        return result
        # raw dict goes out of scope → GC'd

    def _upsert_mapping(self, pseudo_id: UUID, real_id: str) -> None:
        """AES-256-GCM write to encrypted volume. Never to PostgreSQL."""
        ...
```

**Unit Tests** (T-036):
- Assert zero real identifiers in anonymize_signals output
- Assert same input → same UUID (deterministic)
- Assert different orgs (different salts) → different UUIDs

**Deliverable**: `ingestion/anonymizer.py`; privacy tests green.

---

#### Step 17: Feature Extraction & Scheduler (T-037 → T-041)
`ingestion/feature_extractor.py` — 13 features per user per 48hr window:
```
Feature Index | Name                    | Source
0             | meeting_density         | MS Graph
1             | after_hours_meetings    | MS Graph
2             | focus_blocks            | MS Graph
3             | email_response_latency  | MS Graph
4             | accept_rate             | MS Graph
5             | message_volume          | Slack
6             | after_hours_messages    | Slack
7             | response_time_slack     | Slack
8             | mention_frequency       | Slack
9             | commit_frequency        | GitHub
10            | after_hours_commits     | GitHub
11            | pr_review_load          | GitHub
12            | context_switch_rate     | GitHub
```
Missing source → neutral baseline 0.5, `is_imputed=True`.
Min-max normalization: `(x - rolling_min) / (rolling_max - rolling_min + ε)`.

`ingestion/scheduler.py`:
```python
scheduler = AsyncIOScheduler(jobstores={"default": SQLAlchemyJobStore(url=DB_URL)})

@scheduler.scheduled_job("interval", hours=settings.refresh_interval_hours)
async def pipeline_run():
    async with advisory_lock(db, "pipeline_run"):
        signals = await run_ingestion()    # WF-02
        features = extract_features(signals)  # WF-03
        await upsert_feature_store(features)
        await trigger_intelligence_pipeline()  # WF-04→WF-08
```

**Deliverable**: Feature extractor + scheduler; unit tests for all 13 features; APScheduler starts without error.

---

### PHASE 4B — Intelligence Engine (T-042 → T-059) [~51h, Weeks 7-10]

#### Step 18: Graph Builder (T-042 → T-044)
`intelligence/graph_builder.py`:
```python
class GraphBuilder:
    async def build(self, db: AsyncSession, window_end: datetime) -> nx.DiGraph:
        vectors = await db.execute(
            select(FeatureVector).where(FeatureVector.window_end == window_end)
        )
        G = nx.DiGraph()
        for fv in vectors:
            G.add_node(fv.pseudo_id, feature_vector=np.array(fv.feature_json["features"]))
        # Add edges from edge_signal store
        for src, dst, weight in edge_signals:
            G.add_edge(src, dst, weight=weight)
        return G

    def to_pyg(self, G: nx.DiGraph) -> Data:
        # torch_geometric.utils.from_networkx with node/edge attrs
        data = from_networkx(G, group_node_attrs=["feature_vector"], group_edge_attrs=["weight"])
        return data  # Data(x: N×13, edge_index: 2×E, edge_attr: E×1)
```

**Unit Tests** (T-044): node count matches feature store; edge weights in [0,1]; feature matrix shape (N,13).

---

#### Step 19: GAT GNN Model (T-045 → T-052)
`intelligence/gnn_model.py`:
```python
class BurnoutGAT(torch.nn.Module):
    def __init__(self, in_channels=13, hidden=64, heads1=4, heads2=2, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden, heads=heads1, dropout=dropout, concat=True)
        self.bn1 = BatchNorm(hidden * heads1)
        self.conv2 = GATConv(hidden * heads1, hidden // 2, heads=heads2, dropout=dropout, concat=False)
        self.out = Linear(hidden // 2, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None, return_attention=False):
        x, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(self.bn1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, attn2 = self.conv2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        scores = torch.sigmoid(self.out(x)).squeeze(-1)
        if return_attention:
            return scores, attn2
        return scores
```

`intelligence/trainer.py`:
```python
def train(model, data, epochs=100, lr=0.001):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    # Class-weighted BCE (burnout class minority ~15%)
    pos_weight = torch.tensor([(1 - burnout_frac) / burnout_frac])
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
    # Train/val/test split: 80/10/10 stratified
    # Early stopping on val_loss (patience=10)
    # Mini-batch via DataLoader if N > 1000
    ...
```

`intelligence/inference.py` (MC Dropout):
```python
def infer_with_confidence(model, data, n_passes=5) -> ScoredGraph:
    model.train()  # Keep dropout active for MC passes
    scores_mc = [model(data.x, data.edge_index) for _ in range(n_passes)]
    scores_stack = torch.stack(scores_mc)
    mean_score = scores_stack.mean(dim=0)
    std_score = scores_stack.std(dim=0)
    return ScoredGraph(
        pseudo_ids=data.pseudo_ids,
        burnout_scores=mean_score.numpy(),
        confidence_low=(mean_score - 2*std_score).clamp(0,1).numpy(),
        confidence_high=(mean_score + 2*std_score).clamp(0,1).numpy(),
    )
```

Model Registry: `models/<version>/model.pt` + `models/<version>/metadata.json`
```json
{"version": "1.0.0", "training_date": "2026-04-01", "val_accuracy": 0.85,
 "graph_size": 200, "feature_schema_version": "1.0"}
```

**Deliverable**: Model trains to val_acc ≥ 0.80 on synthetic 200-node graph; checkpoint saved.

---

#### Step 20: Cascade Propagation (T-053 → T-055)
`intelligence/cascade.py`:
```python
class CascadePropagator:
    def __init__(self, threshold=0.70, decay=0.60, max_hops=2):
        ...

    def propagate(self, G: nx.DiGraph, scores: dict[UUID, float]) -> dict[UUID, CascadeResult]:
        cascade_risk = defaultdict(float)
        cascade_sources = defaultdict(list)

        high_risk = {nid: s for nid, s in scores.items() if s > self.threshold}

        for hop in range(1, self.max_hops + 1):
            decay_applied = self.decay ** hop
            for source_id, source_score in high_risk.items():
                for neighbor in nx.ego_graph(G, source_id, radius=hop, undirected=False):
                    if neighbor == source_id:
                        continue
                    edge_weight = G[source_id][neighbor].get("weight", 1.0)
                    cascade_risk[neighbor] += source_score * edge_weight * decay_applied
                    cascade_sources[neighbor].append(source_id)

        # Normalize to [0, 1]
        max_risk = max(cascade_risk.values(), default=1.0)
        return {
            nid: CascadeResult(
                cascade_risk=min(risk / max_risk, 1.0),
                cascade_sources=list(set(cascade_sources[nid]))
            )
            for nid, risk in cascade_risk.items()
        }
```

**Unit Tests** (T-055): verify propagation math; verify DECAY_FACTOR^2 applied at hop 2; verify hop limit enforced.

---

#### Step 21: FastAPI Intelligence Endpoints (T-056 → T-059)
`api/routes/scores.py`:
```python
@router.get("/api/v1/scores", response_model=list[BurnoutScoreResponse])
async def get_scores(
    token: TokenPayload = Depends(require_role("hr_admin", "hr_analyst")),
    db: AsyncSession = Depends(get_db),
):
    # HR sees all pseudo_id scores with full details
    scores = await score_repo.get_latest_run(db, token.org_id)
    return [BurnoutScoreResponse.from_orm(s) for s in scores]

@router.get("/api/v1/scores/team-summary", response_model=list[TeamSummaryResponse])
async def get_team_summary(
    token: TokenPayload = Depends(require_role("manager")),
    db: AsyncSession = Depends(get_db),
):
    # Manager sees ONLY team aggregate — NO individual pseudo_id scores
    return await score_repo.get_team_aggregates(db, token.org_id)
```

`api/routes/cascade.py`, `api/routes/pipeline.py` follow same pattern.

**Integration Tests** (T-059):
- HR token → 200 on /scores with individual data
- Manager token → 403 on /scores; 200 on /scores/team-summary
- Unauthenticated → 401 on all endpoints
- IT Admin → 200 on /pipeline/run

---

### PHASE 4C — Output Layer (T-060 → T-065) [~18h, Weeks 11-12]

#### Step 22: Power BI Connector (T-060, T-061)
`output/powerbi_connector.py`:
```python
class PowerBIConnector:
    async def push_scores(self, scored_data: list[ScoredRow]) -> None:
        token = await self._get_service_principal_token()
        rows = [
            {
                "pseudo_id": str(r.pseudo_id),
                "team_id": str(r.team_id),
                "burnout_score": r.burnout_score,
                "cascade_risk": r.cascade_risk,
                "top_features": json.dumps(r.top_features),
                "window_end": r.window_end.isoformat(),
            }
            for r in scored_data
        ]
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{POWERBI_PUSH_URL}/rows",
                json={"rows": rows},
                headers={"Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()
```

Power BI Dashboard (manual build in Power BI Desktop):
1. Connect to streaming dataset (REST API push)
2. Heatmap: treemap visual → Rows=team_id, Values=avg(burnout_score), Color saturation=risk level
3. Cascade Network: custom visual (Network Navigator) → nodes=pseudo_id, edges from cascade_sources
4. Trend lines: line chart → X=window_end, Y=burnout_score, Legend=pseudo_id (HR view only)
5. Rolling Index: card visual → measure: AVERAGE(burnout_scores[burnout_score])

**Deliverable**: PowerBI connector pushes test rows; dashboard renders from pushed data.

---

#### Step 23: MS Teams Bot (T-062 → T-064)
`output/teams_bot/bot.py`:
```python
class CognitiveSyncBot(ActivityHandler):
    async def send_hr_alert(self, risk_cluster: RiskCluster) -> None:
        card = build_hr_adaptive_card(risk_cluster)  # No individual names/IDs
        await self._connector.conversations.send_to_conversation(
            HR_CHANNEL_ID,
            Activity(type=ActivityTypes.message, attachments=[card])
        )

    async def send_manager_alert(self, team_id: UUID, recommendations: list[str]) -> None:
        card = build_manager_adaptive_card(recommendations)  # Redistribution only
        channel_id = MANAGER_CHANNELS[team_id]
        await self._connector.conversations.send_to_conversation(
            channel_id,
            Activity(type=ActivityTypes.message, attachments=[card])
        )
```

`output/teams_bot/cards.py` — Adaptive Card schemas:
```python
def build_hr_adaptive_card(cluster: RiskCluster) -> Attachment:
    return CardFactory.adaptive_card({
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": [
            {"type": "TextBlock", "text": f"⚠️ Burnout Risk Alert", "size": "large", "weight": "bolder"},
            {"type": "TextBlock", "text": f"Teams affected: {cluster.team_count}"},
            {"type": "TextBlock", "text": f"Risk level: {cluster.risk_level}"},
            {"type": "TextBlock", "text": f"Top signals: {', '.join(cluster.top_signals)}"},
            {"type": "TextBlock", "text": "Recommendations:"},
            *[{"type": "TextBlock", "text": f"• {r}"} for r in cluster.recommendations],
        ],
        "actions": [{"type": "Action.OpenUrl", "title": "View Dashboard", "url": PBI_DASHBOARD_URL}]
    })

def build_manager_adaptive_card(recommendations: list[str]) -> Attachment:
    # NO risk scores, NO pseudo_ids, redistribution guidance only
    ...
```

**Deliverable**: Bot sends test alert to a designated channel; manager card contains zero score data.

---

#### Step 24: End-to-End Integration Test (T-065)
Full pipeline flow on Mock Adapter:
1. Start Docker Compose
2. Trigger `POST /api/v1/pipeline/run` (IT Admin JWT)
3. Assert: feature vectors in PostgreSQL
4. Assert: graph built with correct node count
5. Assert: burnout scores [0,1] for all nodes
6. Assert: cascade_risk populated for at-risk neighbors
7. Assert: Power BI push succeeded (mock endpoint)
8. Assert: Teams bot alert fired (mock webhook)
9. Assert: manager endpoint returns team aggregates only (no individual scores)

**Deliverable**: `tests/integration/test_e2e_pipeline.py` passes on Docker Compose.

---

### PHASE 5 — Test (T-066 → T-071) [~14h, Week 13]

#### Step 25: Full Test Suite (T-066 → T-071)

**Coverage targets** (80% minimum per module):
```
Module                      | Target Coverage
ingestion/anonymizer.py     | 95%  (privacy-critical)
ingestion/feature_extractor | 90%
intelligence/cascade.py     | 95%  (math-critical)
intelligence/gnn_model.py   | 80%
api/middleware/rbac.py      | 100% (security-critical)
api/routes/                 | 85%
output/                     | 80%
```

**RBAC Penetration Tests** (T-067):
```python
def test_manager_cannot_access_individual_scores():
    manager_token = generate_jwt(role="manager")
    resp = client.get("/api/v1/scores", headers={"Authorization": f"Bearer {manager_token}"})
    assert resp.status_code == 403

def test_manager_cannot_see_pseudo_ids_in_team_summary():
    manager_token = generate_jwt(role="manager")
    resp = client.get("/api/v1/scores/team-summary", headers={"Authorization": f"Bearer {manager_token}"})
    assert resp.status_code == 200
    for item in resp.json():
        assert "pseudo_id" not in item  # Only team_id visible
```

**Privacy Verification Tests** (T-068):
```python
def test_no_pii_in_feature_store(db):
    vectors = db.execute(select(FeatureVector)).all()
    for v in vectors:
        assert "@" not in str(v.pseudo_id)          # No email
        assert not is_valid_email(str(v.pseudo_id)) # No email format
        assert len(str(v.pseudo_id)) == 36           # UUID format only

def test_anonymizer_deterministic(anonymizer):
    id1 = anonymizer.pseudonymize("user@example.com")
    id2 = anonymizer.pseudonymize("user@example.com")
    assert id1 == id2

def test_anonymizer_different_salts_different_uuids():
    a1 = Anonymizer(org_salt="salt1", vault_key="key")
    a2 = Anonymizer(org_salt="salt2", vault_key="key")
    assert a1.pseudonymize("user@example.com") != a2.pseudonymize("user@example.com")
```

**Performance Test** (T-070): Pipeline run on 500-node synthetic graph < 5 minutes end-to-end on RTX 4060.

**Deliverable**: `pytest --cov` reports ≥80% overall; all tests green; `tests/REPORT.md` documents results.

---

### PHASE 6 — Release & Submission (T-072 → T-076) [~10.5h, Week 14]

#### Step 26: Demo & Submission (T-072 → T-076)

**Demo Script**:
1. Show Docker Compose startup (clean environment)
2. Trigger pipeline run with Mock Adapter (100-node graph)
3. Show PostgreSQL feature vectors populated
4. Show Power BI dashboard: heatmap + cascade graph
5. Show Teams bot HR alert firing
6. Show RBAC: manager API call returns 403 on /scores

**Pitch Deck Structure** (T-073):
1. Problem: burnout costs + detection gap
2. Solution: passive behavioral graph intelligence
3. Architecture: 4-layer diagram from ARCHITECTURE.md
4. Privacy-first: anonymization flow
5. Demo: GNN risk map + cascade visualization
6. Impact: proactive vs reactive HR intervention
7. Success Criteria: all 7 from PRD §10 met
8. Roadmap: V2 additions (Neo4j, wearables, real-time)

**Submission Package** (T-075):
- `/repo` — GitHub link
- `/docs` — All 6 SDLC documents
- `/demo` — Demo video (3-5 min Loom)
- `/deck` — UIP Y17 pitch deck

---

## Key Files Summary

| File | Operation | Description |
|------|-----------|-------------|
| `pyproject.toml` | Create | uv project, ruff/mypy config |
| `config/settings.py` | Create | Pydantic Settings, all env vars |
| `ingestion/adapters/base.py` | Create | Abstract adapter interface |
| `ingestion/adapters/mock.py` | Create | Synthetic data adapter (primary test tool) |
| `ingestion/adapters/msgraph.py` | Create | MSAL + calendar/email signals |
| `ingestion/adapters/slack.py` | Create | slack-sdk async + message signals |
| `ingestion/adapters/github.py` | Create | PyGithub + commit/PR signals |
| `ingestion/anonymizer.py` | Create | SHA-256 UUID v5 + AES-256 mapping store |
| `ingestion/feature_extractor.py` | Create | 13-dim feature vector computation |
| `ingestion/scheduler.py` | Create | APScheduler 48hr pipeline job |
| `ingestion/db/models.py` | Create | SQLAlchemy 2.x async ORM |
| `intelligence/graph_builder.py` | Create | NetworkX graph + PyG converter |
| `intelligence/gnn_model.py` | Create | 2-layer GAT with MC Dropout |
| `intelligence/trainer.py` | Create | Training pipeline + model registry |
| `intelligence/inference.py` | Create | Inference with confidence intervals |
| `intelligence/cascade.py` | Create | 2-hop cascade propagation |
| `api/main.py` | Create | FastAPI app with all routers |
| `api/middleware/auth.py` | Create | RS256 JWT validation |
| `api/middleware/rbac.py` | Create | Role-based access control |
| `api/routes/scores.py` | Create | /scores endpoint (HR only) |
| `api/routes/cascade.py` | Create | /cascade-map endpoint |
| `api/routes/pipeline.py` | Create | /pipeline/run endpoint |
| `output/powerbi_connector.py` | Create | httpx async Power BI push |
| `output/teams_bot/bot.py` | Create | Bot Framework activity handler |
| `output/teams_bot/cards.py` | Create | Adaptive Card builders |
| `data/synthetic.py` | Create | Org graph + feature + edge generators |
| `deploy/docker-compose.yml` | Create | Full stack service definitions |
| `deploy/schema.sql` | Create | PostgreSQL DDL |
| `tests/unit/` | Create | Per-module unit tests |
| `tests/integration/` | Create | End-to-end pipeline test |

---

## Risks and Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Enterprise API credentials unavailable | High | High | Mock Adapter (T-030) used for all dev/demo — real APIs tested only if credentials available |
| GNN val_accuracy < 0.80 | Medium | High | Fallback to GCN; tune with Optuna; increase synthetic dataset to 1000 nodes |
| OOM on RTX 4060 during GAT training | Medium | Medium | Mini-batch via PyG DataLoader; reduce hidden_dim from 64 to 32 |
| Power BI REST API licensing issue | Low | Medium | Power BI Desktop (free) sufficient for demo; REST push tested on dev tenant |
| Teams bot proactive messaging complexity | Medium | Low | Fallback: webhook-based alert via incoming webhook if Bot Framework proves complex |
| Docker Compose issues in clean env | Low | Medium | Test on fresh VM/container in Phase 5; document all required env vars in .env.example |
| Solo 14-week timeline pressure | Medium | Medium | Phase 4A Mock Adapter allows parallel frontend work; phases have buffer time built in |

---

## PRD Success Criteria Traceability

| Success Criterion (PRD §10) | Implementation |
|-----------------------------|----------------|
| All 3 integrations operational with synthetic/mock data | T-027:T-030 + Mock Adapter |
| GNN produces scores on 100+ node graph | T-045:T-049 + synthetic 200-node graph |
| Cascade propagation identifies at-risk neighbors | T-053:T-055 |
| Power BI dashboard renders heatmap + cascade viz | T-060:T-061 |
| MS Teams bot delivers test alert to HR channel | T-062:T-064 |
| RBAC: manager cannot access individual scores via API | T-021 + T-067 (pen test) |
| Full stack deployable via Docker Compose | T-011 + T-069 |

---

## SESSION_ID (for /ccg:execute use)
- CODEX_SESSION: N/A (codeagent-wrapper not available — plan synthesized by Claude directly)
- GEMINI_SESSION: N/A

---

## Definition of Done (per TASK_PLAN.docx)

Each task complete when:
- [ ] Implementation matches ARCHITECTURE.md / WORKFLOWS.md spec
- [ ] Unit tests written and passing
- [ ] `uv run ruff check .` — zero errors
- [ ] `uv run mypy .` — zero errors (strict mode)
- [ ] Self-reviewed against logic/edge cases/error handling/logging checklist
- [ ] `print()` replaced with `logging.*`
