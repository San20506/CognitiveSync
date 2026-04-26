#!/usr/bin/env bash
# CognitiveSync — local dev setup (fully offline)
# Usage: bash scripts/dev-setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
die()   { echo -e "${RED}[error]${NC} $*"; exit 1; }

# ── 1. Copy .env ─────────────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
    cp .env.local .env
    info "Created .env from .env.local"
else
    warn ".env already exists — skipping copy"
fi

# ── 2. Generate JWT keys ──────────────────────────────────────────────────────
mkdir -p secrets
if [[ ! -f secrets/jwt_private.pem ]]; then
    openssl genrsa -out secrets/jwt_private.pem 2048 2>/dev/null
    openssl rsa -in secrets/jwt_private.pem -pubout -out secrets/jwt_public.pem 2>/dev/null
    info "Generated RSA-2048 JWT key pair → secrets/"
else
    warn "JWT keys already exist — skipping generation"
fi
# Ensure secrets are not world-readable
chmod 600 secrets/jwt_private.pem secrets/jwt_public.pem 2>/dev/null || true

# ── 3. Start PostgreSQL ───────────────────────────────────────────────────────
info "Starting PostgreSQL via Docker Compose …"
docker compose up -d db

info "Waiting for PostgreSQL to be ready …"
until docker compose exec db pg_isready -U cognitivesync -d cognitivesync -q 2>/dev/null; do
    sleep 1
done
info "PostgreSQL is ready"

# ── 4. Run Alembic migrations ─────────────────────────────────────────────────
info "Running database migrations …"
uv run alembic upgrade head

info "Migrations applied"

# ── 5. Summary ────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  CognitiveSync local stack is ready${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  PostgreSQL  → localhost:5432  (user: cognitivesync)"
echo "  JWT keys    → secrets/jwt_private.pem + jwt_public.pem"
echo ""
echo "  Start API:  uv run uvicorn api.main:app --reload --port 8000"
echo "  Run demo:   uv run python scripts/run_demo.py"
echo "  API docs:   http://localhost:8000/docs"
echo "  Stop DB:    docker compose down"
echo ""
