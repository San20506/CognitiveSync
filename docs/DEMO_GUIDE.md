# CognitiveSync — Demo Guide
**Sprint:** April 16–30, 2026 | **Prepared by:** M Santhosh (Sandy)

---

## What You're Demoing

A live burnout prediction platform running on your local machine:
- Real GNN model (GAT) scoring 120 pseudonymised employees
- Live scores persisted in PostgreSQL
- Role-gated dashboard (HR Admin sees everything; Manager sees team-only)
- Full pipeline triggerable from the UI

---

## Prerequisites

| Requirement | Check |
|------------|-------|
| Docker Desktop running | `docker ps` → no error |
| Python 3.11+ + uv | `uv --version` |
| Repo cloned, deps installed | `uv sync` |

---

## Step 1 — Start the stack (one command)

```bash
cd /path/to/CognitiveSync

# Use the pre-configured local dev env
cp .env.local .env

bash start_demo.sh
```

Docker pulls python:3.11-slim + CPU-only PyTorch (~500MB total, one-time).  
Migrations run automatically via the `migrate` service before the API starts.

Wait for:
```
cognitivesync-db      | database system is ready to accept connections
cognitivesync-migrate | INFO [alembic] Running upgrade -> 893cdc022dcc
cognitivesync-api     | Application startup complete.
```

**API:** http://localhost:8000  
**Dashboard:** http://localhost:8000/dashboard  
**Swagger:** http://localhost:8000/docs

---

## Step 2 — Seed the database (first run only)

Migrations run automatically via docker-compose. Just trigger the first pipeline run:

```bash
# Mint an IT Admin token
uv run python scripts/mint_demo_token.py --role it_admin

# Trigger pipeline (replace <token> with output above)
curl -s -X POST http://localhost:8000/api/v1/pipeline/run \
  -H "Authorization: Bearer <token>"
```

Or just use the dashboard's IT Admin view (Step 3 covers this).

---

## Step 3 — Open the Dashboard

Go to: **http://localhost:8000/dashboard**

The dashboard auto-fetches demo tokens — no login needed.

### Role Switcher (top-right dropdown)

| Role | What they see |
|------|--------------|
| **HR Admin** | Full risk table (120 employees), score distribution, cascade network, CI chart, feature signals |
| **HR Analyst** | Same as HR Admin (read-only, no config access) |
| **Team Manager** | Team-level cards only — individual scores are redacted |
| **IT Admin** | Pipeline trigger panel + system info |

---

## Demo Script (suggested flow, ~10 minutes)

### 1. Open as HR Admin (default)

**Say:** *"This is what an HR Admin would see — 120 employees monitored, no names, no emails, just pseudonymised IDs. Privacy by design."*

Point out:
- **KPI row:** Critical risk count, cascade exposure count
- **Distribution chart:** Majority are safe; the critical bucket is the concern
- **Top signals chart:** Which behaviours are driving risk org-wide (meeting density, response latency)

### 2. Scroll to the Risk Table

**Say:** *"Every employee gets a burnout score, a confidence interval from the model's uncertainty quantification, and their top contributing signal. Clicking into any individual profile shows the full behavioural breakdown."*

Point out:
- The confidence interval column (MC Dropout — the model tells you how sure it is)
- The cascade risk column — some high-risk employees have near-zero cascade (isolated), others are spreading risk

### 3. Show the Cascade Network

**Say:** *"This is the org collaboration graph. Red nodes are critical-risk employees. White-outlined nodes are the cascade sources. The edges are interaction signals — meetings, Slack, GitHub PRs. You can see how burnout risk propagates through a team."*

Point out:
- Clusters of red/orange nodes = team under pressure
- Isolated green nodes = not exposed to cascade

### 4. Switch to Manager view

**Say:** *"A team manager logs in and sees a completely different view — no individual scores. Just team-level aggregates and recommended actions. RBAC enforced at the API level."*

Point out:
- Team cards with risk levels
- Automatic recommendations (meeting reduction, focus time blocks)
- The lock notice: "Individual employee data is redacted"

### 5. Switch to IT Admin — Trigger a pipeline run

**Say:** *"IT Admin can trigger a fresh scoring run. This kicks off: synthetic data → graph build → GNN inference → cascade propagation → persisted to Postgres. Takes about 8 seconds."*

Click **▶ Trigger Pipeline Run** and watch the status update live.

---

## What's Synthetic vs Real

| Component | Status for Demo |
|-----------|----------------|
| Employee data | Synthetic (120 employees, realistic distributions) |
| GNN model | Real — trained, 5-fold CV, AUC=0.857 |
| Burnout scores | Real model output on synthetic inputs |
| Cascade propagation | Real algorithm |
| Database | Real PostgreSQL, real schema |
| API auth | Real JWT RS256, real RBAC |
| MS Graph / Slack / GitHub connectors | Ready, not called (ADAPTER_MODE=mock) |

**Honest framing for the audience:**  
*"The model is real. The data is synthetic stand-ins for what the live connectors would pull. Phase 2 wires in MS Graph, Slack, and GitHub."*

---

## If Something Breaks

**Dashboard shows "Error loading view"**
```bash
# Check API is running
curl http://localhost:8000/health

# If not, restart
docker-compose restart api
```

**Scores endpoint returns empty array**
```bash
# Re-run the pipeline
uv run python scripts/mint_demo_token.py --role it_admin
# Then POST /api/v1/pipeline/run with that token
```

**Docker won't start**
```bash
docker-compose down -v   # wipe volumes
bash start_demo.sh
# Then re-run Alembic and pipeline
```

**Port 8000 already in use**
```bash
kill $(lsof -ti:8000)
docker-compose up
```

---

## Power BI & Teams (Roadmap Slide)

For the demo, show the live web dashboard as the **POC layer**.  
Frame Power BI and Teams as Phase 2 delivery surfaces:

> *"The dashboard you're seeing calls the same API that Power BI Embedded would call.  
> The report layout is already specced — it's a skin on top of what's running here.  
> MS Teams Adaptive Cards are the alert surface for managers — we have the bot code,  
> it needs a Teams App registration to go live."*

**Estimated Phase 2 timeline (after POC sign-off):**
- Power BI Embedded report: 2–3 days (workspace + report layout + embed token)
- Teams bot: 1–2 days (App registration + channel wiring)

---

## Key Numbers to Quote

| Metric | Value |
|--------|-------|
| Employees scored | 120 |
| Critical risk | 23 (19%) |
| Cascade-affected | 69 (58%) |
| Top risk score | 88% (meeting density + response latency) |
| Model AUC | **0.857** (best fold) |
| Phase gate | PASSED |
| Inference time | ~3 seconds for 120 nodes |
| Privacy | Zero PII — UUID v5 pseudonyms only |
