"""Model registry and retraining endpoints — Phase 4B (T-050)."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()

# POST /api/v1/model/retrain      → IT Admin
# GET  /api/v1/model/versions     → IT Admin
# PUT  /api/v1/model/activate/{v} → IT Admin
