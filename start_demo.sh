#!/usr/bin/env bash
# CognitiveSync — one-command demo startup
set -e

cd "$(dirname "$0")"

echo "=== CognitiveSync Demo ==="

# 1. Start DB
echo "[1/4] Starting PostgreSQL..."
docker-compose up -d db
echo "      Waiting for DB to be ready..."
until docker exec cognitivesync-db pg_isready -U cognitivesync -q 2>/dev/null; do
  sleep 1
done
echo "      DB ready."

# 2. Copy env if not present
if [ ! -f .env ]; then
  cp .env.local .env
  echo "[2/4] Copied .env.local -> .env"
else
  echo "[2/4] .env already exists, skipping."
fi

# 3. Run migrations
echo "[3/4] Running database migrations..."
DATABASE_URL="postgresql+asyncpg://cognitivesync:cognitivesync_local@localhost:5432/cognitivesync" \
  uv run alembic upgrade head 2>&1 | grep -E "Running|already|head|INFO" || true
echo "      Migrations applied."

# 4. Start API
echo "[4/4] Starting API on http://localhost:8000"
echo ""
echo "  Dashboard  -> http://localhost:8000/dashboard"
echo "  Swagger    -> http://localhost:8000/docs"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

JWT_PRIVATE_KEY_PATH=secrets/jwt_private.pem \
JWT_PUBLIC_KEY_PATH=secrets/jwt_public.pem \
DATABASE_URL="postgresql+asyncpg://cognitivesync:cognitivesync_local@localhost:5432/cognitivesync" \
AUDIT_DATABASE_URL="postgresql+asyncpg://cognitivesync:cognitivesync_local@localhost:5432/cognitivesync" \
ORG_SALT="devonly00000000000000000000000000" \
VAULT_KEY="devonly00000000000000000000000000" \
ADAPTER_MODE=mock \
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
