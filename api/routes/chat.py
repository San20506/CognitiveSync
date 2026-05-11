"""AI advisor chat endpoint — single backend for dashboard widget and Teams bot."""

from __future__ import annotations

import logging
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import TokenPayload
from api.middleware.rbac import require_role
from api.schemas.common import UserRole, score_to_risk_level
from config.settings import settings
from ingestion.db.models import BurnoutScore
from ingestion.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

_SYSTEM_PROMPT = """\
You are CognitiveSync, an AI advisor for enterprise workforce wellbeing.
You help HR analysts and managers understand burnout risk data and take action.
Be concise, empathetic, and actionable. Never identify individuals by name.
Limit responses to 3-5 sentences unless more detail is specifically requested.\
"""


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    employee_id: str | None = None
    team_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    context_used: dict  # type: ignore[type-arg]


def _build_context_block(
    score: float | None,
    top_features: dict | None,  # type: ignore[type-arg]
    scope: str,
) -> str:
    if score is None:
        return ""
    risk = score_to_risk_level(score)
    signals = ", ".join(list((top_features or {}).keys())[:3]) or "none available"
    return (
        f"\nCurrent risk context ({scope}):\n"
        f"- Burnout score: {score:.0%} ({risk.value} risk)\n"
        f"- Top contributing signals: {signals}\n"
    )


async def _call_ollama(prompt: str, system: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": f"{system}\n\nUser: {prompt}\nAssistant:",
                "stream": False,
                "options": {"num_predict": settings.chat_max_tokens},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("response", "")).strip()


async def _call_azure_openai(prompt: str, system: str) -> str:
    from openai import AsyncAzureOpenAI

    client = AsyncAzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_key,
        api_version=settings.azure_openai_api_version,
    )
    completion = await client.chat.completions.create(
        model=settings.azure_openai_deployment,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=settings.chat_max_tokens,
    )
    return completion.choices[0].message.content or ""


async def _call_llm(prompt: str, system: str) -> str:
    if settings.llm_provider == "azure_openai":
        return await _call_azure_openai(prompt, system)
    return await _call_ollama(prompt, system)


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    token: TokenPayload = require_role(UserRole.HR_ANALYST, UserRole.HR_ADMIN, UserRole.MANAGER),  # type: ignore[assignment]
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """AI advisor — accepts a free-text question with optional burnout context.

    Access: HR Analyst, HR Admin, Manager.
    Stateless: no conversation history stored (privacy constraint).
    """
    score: float | None = None
    top_features: dict | None = None  # type: ignore[type-arg]
    scope = "general"

    if request.employee_id is not None:
        try:
            pseudo_uuid = UUID(request.employee_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="employee_id must be a valid UUID.",
            )
        result = await db.execute(
            select(BurnoutScore)
            .where(BurnoutScore.pseudo_id == pseudo_uuid)
            .order_by(BurnoutScore.window_end.desc())
            .limit(1)
        )
        row = result.scalar_one_or_none()
        if row is not None:
            score = row.burnout_score
            top_features = row.top_features or {}
            scope = "employee"

    elif request.team_id is not None:
        try:
            team_uuid = UUID(request.team_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="team_id must be a valid UUID.",
            )
        result = await db.execute(
            select(BurnoutScore.run_id)
            .order_by(BurnoutScore.window_end.desc())
            .limit(1)
        )
        latest_run = result.scalar_one_or_none()
        if latest_run is not None:
            result = await db.execute(
                select(BurnoutScore).where(
                    BurnoutScore.run_id == latest_run,
                    BurnoutScore.team_id == team_uuid,
                )
            )
            team_scores = result.scalars().all()
            if team_scores:
                score = sum(s.burnout_score for s in team_scores) / len(team_scores)
                all_features: dict[str, float] = {}
                for s in team_scores:
                    for feat, weight in (s.top_features or {}).items():
                        all_features[feat] = all_features.get(feat, 0.0) + float(weight)
                top_features = dict(
                    sorted(all_features.items(), key=lambda x: x[1], reverse=True)[:5]
                )
                scope = "team"

    context_block = _build_context_block(score, top_features, scope)
    system = _SYSTEM_PROMPT + context_block

    try:
        reply = await _call_llm(request.message, system)
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI advisor temporarily unavailable.",
        ) from exc

    return ChatResponse(
        reply=reply,
        context_used={
            "score": score,
            "top_signals": list((top_features or {}).keys()),
            "scope": scope,
        },
    )
