"""Pipeline management endpoints — implementation in Phase 4A (T-040)."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()

# POST /api/v1/pipeline/run → IT Admin
# GET  /api/v1/pipeline/status/{run_id} → IT Admin
