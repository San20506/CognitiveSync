"""Pipeline management endpoints — T-040."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import TokenPayload
from api.middleware.rbac import require_role
from api.schemas.common import PipelineStatus, UserRole
from ingestion.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory run status registry (single-process demo; replace with DB for prod)
_run_registry: dict[UUID, dict] = {}

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODEL_REGISTRY = Path(__file__).parent.parent.parent / "models"


class PipelineRunResponse(BaseModel):
    run_id: UUID
    status: PipelineStatus
    started_at: datetime
    node_count: int | None = None
    high_risk_count: int | None = None
    message: str = ""


class PipelineStatusResponse(BaseModel):
    run_id: UUID
    status: PipelineStatus
    started_at: datetime
    completed_at: datetime | None = None
    node_count: int | None = None
    high_risk_count: int | None = None
    message: str = ""


async def _run_pipeline(run_id: UUID) -> None:
    """Execute full pipeline: CSV → graph → GNN → cascade → persist scores."""
    from ingestion.db.session import AsyncSessionLocal
    from intelligence.cascade import CascadePropagator
    from intelligence.graph_builder import GraphBuilder
    from intelligence.inference import InferencePipeline

    _run_registry[run_id]["status"] = PipelineStatus.RUNNING

    try:
        features_path = str(DATA_DIR / "features.csv")
        interactions_path = str(DATA_DIR / "interactions.csv")

        if not Path(features_path).exists():
            raise FileNotFoundError(f"features.csv not found at {features_path}")
        if not Path(interactions_path).exists():
            raise FileNotFoundError(f"interactions.csv not found at {interactions_path}")

        # Step 1: Build graph from CSVs
        builder = GraphBuilder()
        built = builder.build_from_csv(features_path, interactions_path)
        logger.info("Graph built: %d nodes", len(built.node_ids))

        # Step 2: Load model and run inference
        pipeline = InferencePipeline(
            model_registry_path=MODEL_REGISTRY,
            device="cuda",
        )
        pipeline.load_model(version="latest")
        scored = pipeline.score(
            pyg_data=built.pyg_data,
            node_ids=built.node_ids,
            run_id=run_id,
        )

        # Step 3: Cascade propagation
        propagator = CascadePropagator()
        burnout_scores = {pid: ns.burnout_score for pid, ns in scored.node_scores.items()}
        cascade_results = propagator.propagate(built.nx_graph, burnout_scores)

        window_end = datetime.now(UTC)
        high_risk_count = sum(1 for ns in scored.node_scores.values() if ns.burnout_score >= 0.70)

        # Step 4: Persist scores + update profiles (always use a fresh session)
        from intelligence.profile_updater import update_profiles
        async with AsyncSessionLocal() as fresh_db:
            await _persist_scores(run_id, window_end, scored, cascade_results, fresh_db)
            await update_profiles(run_id, scored.node_scores, cascade_results, fresh_db)
            await fresh_db.commit()

        _run_registry[run_id].update({
            "status": PipelineStatus.COMPLETED,
            "completed_at": datetime.now(UTC),
            "node_count": len(built.node_ids),
            "high_risk_count": high_risk_count,
        })
        logger.info(
            "Pipeline run %s completed — %d nodes, %d high-risk",
            run_id, len(built.node_ids), high_risk_count,
        )

    except Exception as exc:
        logger.exception("Pipeline run %s failed: %s", run_id, exc)
        _run_registry[run_id].update({
            "status": PipelineStatus.FAILED,
            "completed_at": datetime.now(UTC),
            "message": str(exc),
        })


async def _persist_scores(
    run_id: UUID,
    window_end: datetime,
    scored: object,
    cascade_results: dict,
    db: AsyncSession,
) -> None:
    from uuid import uuid4

    from ingestion.db.models import BurnoutScore
    from intelligence.inference import ScoredGraph
    sg: ScoredGraph = scored  # type: ignore[assignment]

    for pid, ns in sg.node_scores.items():
        cr = cascade_results.get(pid)
        record = BurnoutScore(
            id=uuid4(),
            run_id=run_id,
            pseudo_id=pid,
            burnout_score=ns.burnout_score,
            confidence_low=ns.confidence_low,
            confidence_high=ns.confidence_high,
            cascade_risk=cr.cascade_risk if cr else 0.0,
            cascade_sources={"sources": [str(s) for s in cr.cascade_sources]}
            if cr
            else {"sources": []},
            top_features=ns.top_features,
            window_end=window_end,
            team_id=None,
        )
        db.add(record)

    await db.commit()


@router.post(
    "/pipeline/run",
    response_model=PipelineRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_pipeline_run(
    token: TokenPayload = require_role(UserRole.IT_ADMIN),
) -> PipelineRunResponse:
    """Trigger a full scoring pipeline run.

    Starts async pipeline execution (CSV → graph → GNN → cascade → DB).
    Returns immediately with run_id; poll /pipeline/status/{run_id} for progress.

    Access: IT Admin only.
    """
    run_id = uuid4()
    started_at = datetime.now(UTC)
    _run_registry[run_id] = {
        "status": PipelineStatus.STARTED,
        "started_at": started_at,
        "completed_at": None,
        "node_count": None,
        "high_risk_count": None,
        "message": "",
    }

    # Fire and forget — task opens its own DB session to avoid request-scope teardown
    asyncio.create_task(_run_pipeline(run_id))

    return PipelineRunResponse(
        run_id=run_id,
        status=PipelineStatus.STARTED,
        started_at=started_at,
    )


@router.get("/pipeline/status/{run_id}", response_model=PipelineStatusResponse)
async def get_pipeline_status(
    run_id: UUID,
    token: TokenPayload = require_role(UserRole.IT_ADMIN),
) -> PipelineStatusResponse:
    """Poll pipeline run status.

    Access: IT Admin only.
    """
    entry = _run_registry.get(run_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return PipelineStatusResponse(
        run_id=run_id,
        status=entry["status"],
        started_at=entry["started_at"],
        completed_at=entry.get("completed_at"),
        node_count=entry.get("node_count"),
        high_risk_count=entry.get("high_risk_count"),
        message=entry.get("message", ""),
    )
