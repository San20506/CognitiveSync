"""SHAP explainability endpoint — GET /api/v1/explain/{employee_id}."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import UUID

import torch
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import TokenPayload
from api.middleware.rbac import require_role
from api.schemas.common import RiskLevel, UserRole, score_to_risk_level
from ingestion.db.models import BurnoutScore
from ingestion.db.session import get_db
from intelligence.inference import FEATURE_NAMES

logger = logging.getLogger(__name__)
router = APIRouter()

_N_FEATURES: int = len(FEATURE_NAMES)

# Module-level holder avoids global assignment (PLW0603).
_state: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class ShapFeatureAttribution(BaseModel):
    model_config = ConfigDict(strict=True)

    feature: str
    shap_value: float
    confidence_low: float
    confidence_high: float
    direction: str = Field(description="'risk' if shap_value > 0 else 'protective'")


class ShapExplanationResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    pseudo_id: UUID
    model_output: float = Field(ge=0.0, le=1.0)
    base_value: float
    risk_level: RiskLevel
    features: list[ShapFeatureAttribution]
    generated_at: datetime


# ---------------------------------------------------------------------------
# Lazy explainer singleton
# ---------------------------------------------------------------------------


def _get_explainer() -> object:
    if "explainer" in _state:
        return _state["explainer"]

    try:
        from config.settings import settings  # noqa: PLC0415
        from intelligence.gnn_model import SmallBurnoutGAT  # noqa: PLC0415
        from intelligence.shap_explainer import ShapExplainer  # noqa: PLC0415

        model = SmallBurnoutGAT()
        registry = settings.model_registry_path
        candidates = [
            registry / settings.active_model_version,
            (registry / "latest").resolve() if (registry / "latest").exists() else None,
        ]
        model_pt: object = None
        for candidate in candidates:
            if candidate is not None and (candidate / "model.pt").exists():
                model_pt = candidate / "model.pt"
                break

        if model_pt is not None:
            from pathlib import Path  # noqa: PLC0415

            state_dict = torch.load(
                Path(str(model_pt)), map_location="cpu", weights_only=True
            )
            model.load_state_dict(state_dict)
            logger.info("ShapExplainer: model loaded from %s", model_pt)
        else:
            logger.warning("ShapExplainer: no model checkpoint found; using random weights")

        model.eval()
        # Neutral background: uniform [0.25, 0.75] avoids boundary effects
        background = torch.rand(_N_FEATURES * 10, _N_FEATURES) * 0.5 + 0.25
        _state["explainer"] = ShapExplainer(model=model, background=background)

    except Exception:
        logger.warning("ShapExplainer could not be initialised", exc_info=True)
        _state["explainer"] = None

    return _state["explainer"]


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.get("/explain/{employee_id}", response_model=ShapExplanationResponse)
async def explain_employee(
    employee_id: UUID,
    token: TokenPayload = require_role(UserRole.HR_ADMIN, UserRole.HR_ANALYST),
    db: AsyncSession = Depends(get_db),
) -> ShapExplanationResponse:
    """Return SHAP feature attributions for a single pseudonymized employee.

    Reconstructs the node's feature vector from the stored burnout score,
    then runs ShapExplainer to attribute each feature's contribution.

    Access: HR Admin, HR Analyst only.
    """
    explainer = _get_explainer()
    if explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHAP explainer not available — model registry not configured.",
        )

    result = await db.execute(
        select(BurnoutScore)
        .where(BurnoutScore.pseudo_id == employee_id)
        .order_by(BurnoutScore.window_end.desc())
        .limit(1)
    )
    score_row: BurnoutScore | None = result.scalar_one_or_none()
    if score_row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No burnout score found for this employee.",
        )

    # Reconstruct feature vector: stored top_features supply their values;
    # remaining features default to neutral 0.5.
    feat_vec = [0.5] * _N_FEATURES
    for name, val in (score_row.top_features or {}).items():
        if name in FEATURE_NAMES:
            feat_vec[FEATURE_NAMES.index(name)] = float(val)

    node_features = torch.tensor([feat_vec], dtype=torch.float32)  # [1, F]
    # Single-node self-loop graph keeps graph structure trivial for attribution.
    edge_index = torch.zeros(2, 1, dtype=torch.long)

    from intelligence.shap_explainer import ShapExplainer  # noqa: PLC0415

    typed_explainer: ShapExplainer = explainer  # type: ignore[assignment]
    explanation = await typed_explainer.explain(
        pseudo_id=employee_id,
        node_features=node_features,
        edge_index=edge_index,
    )

    features = [
        ShapFeatureAttribution(
            feature=name,
            shap_value=sv,
            confidence_low=cl,
            confidence_high=ch,
            direction="risk" if sv >= 0 else "protective",
        )
        for name, sv, cl, ch in zip(
            explanation.feature_names,
            explanation.shap_values,
            explanation.confidence_low,
            explanation.confidence_high,
            strict=True,
        )
    ]
    features.sort(key=lambda f: abs(f.shap_value), reverse=True)

    return ShapExplanationResponse(
        pseudo_id=employee_id,
        model_output=max(0.0, min(1.0, explanation.model_output)),
        base_value=explanation.base_value,
        risk_level=score_to_risk_level(explanation.model_output),
        features=features,
        generated_at=datetime.now(timezone.utc),
    )
