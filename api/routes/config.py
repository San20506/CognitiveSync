"""Org config and alert threshold endpoints — Phase 4C."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()

# POST /api/v1/config/org      → IT Admin
# PUT  /api/v1/alerts/config   → HR Admin
# GET  /api/v1/alerts/config   → HR Admin + IT Admin
