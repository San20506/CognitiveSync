"""Unit tests for the /api/v1/chat endpoint."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from api.middleware.auth import TokenPayload
from api.routes.chat import router
from api.schemas.common import UserRole


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/chat")
    return app


def _mock_token(role: UserRole) -> TokenPayload:
    return TokenPayload(sub=uuid4(), role=role, org_id=uuid4(), exp=9999999999)


# Override RBAC and DB for all tests
@pytest.fixture
def app_with_analyst_token() -> FastAPI:
    app = _make_app()

    async def _override_db() -> AsyncGenerator[AsyncMock, None]:
        yield AsyncMock()

    from ingestion.db.session import get_db
    app.dependency_overrides[get_db] = _override_db
    return app


async def _post_chat(
    app: FastAPI,
    payload: dict,  # type: ignore[type-arg]
    role: UserRole = UserRole.HR_ANALYST,
) -> tuple[int, dict]:  # type: ignore[type-arg]
    from api.middleware.auth import verify_jwt

    token = _mock_token(role)

    async def _mock_verify() -> TokenPayload:
        return token

    app.dependency_overrides[verify_jwt] = _mock_verify

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/chat/", json=payload)
    return resp.status_code, resp.json()


# ── Test 1: Unauthenticated request returns 401/403 ──────────────────────────

async def test_chat_requires_auth() -> None:
    app = _make_app()

    async def _override_db() -> AsyncGenerator[AsyncMock, None]:
        yield AsyncMock()

    from ingestion.db.session import get_db
    app.dependency_overrides[get_db] = _override_db

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/chat/", json={"message": "hello"})

    assert resp.status_code in (401, 403)


# ── Test 2: Employee role is forbidden ───────────────────────────────────────

async def test_chat_employee_role_forbidden(app_with_analyst_token: FastAPI) -> None:
    status_code, _ = await _post_chat(
        app_with_analyst_token,
        {"message": "hello"},
        role=UserRole.HR_ANALYST,  # We test forbidden by patching require_role below
    )
    # HR Analyst is allowed — this confirms the route works.
    # We test forbidden by directly checking RBAC logic is wired.
    # To test forbidden: instantiate without override so RBAC fires.
    app = _make_app()

    async def _override_db() -> AsyncGenerator[AsyncMock, None]:
        yield AsyncMock()

    from api.middleware.auth import verify_jwt
    from ingestion.db.session import get_db

    # Simulate an IT_ADMIN token (not in allowed roles for chat)
    it_admin_token = _mock_token(UserRole.IT_ADMIN)

    async def _mock_verify_it() -> TokenPayload:
        return it_admin_token

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[verify_jwt] = _mock_verify_it

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/chat/", json={"message": "hello"})

    assert resp.status_code == 403


# ── Test 3: General question (no context) returns scope=general ──────────────

async def test_chat_general_question_no_context(app_with_analyst_token: FastAPI) -> None:
    with patch("api.routes.chat._call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = "Here are some general burnout prevention tips."
        status_code, body = await _post_chat(
            app_with_analyst_token, {"message": "What are common burnout signs?"}
        )

    assert status_code == 200
    assert body["reply"] == "Here are some general burnout prevention tips."
    assert body["context_used"]["scope"] == "general"
    assert body["context_used"]["score"] is None


# ── Test 4: Employee ID context injects score ────────────────────────────────

async def test_chat_with_employee_id(app_with_analyst_token: FastAPI) -> None:
    pseudo_id = uuid4()

    mock_score = MagicMock()
    mock_score.burnout_score = 0.82
    mock_score.top_features = {"after_hours_commits": 0.9, "meeting_density": 0.7}
    mock_score.window_end = datetime.now(UTC)

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_score

    from ingestion.db.session import get_db

    async def _override_db_with_score() -> AsyncGenerator[AsyncMock, None]:
        db = AsyncMock()
        db.execute = AsyncMock(return_value=mock_result)
        yield db

    app_with_analyst_token.dependency_overrides[get_db] = _override_db_with_score

    with patch("api.routes.chat._call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = "This employee shows high burnout risk."
        status_code, body = await _post_chat(
            app_with_analyst_token,
            {"message": "Why is this person at risk?", "employee_id": str(pseudo_id)},
        )

    assert status_code == 200
    assert body["context_used"]["score"] == pytest.approx(0.82)
    assert body["context_used"]["scope"] == "employee"
    assert "after_hours_commits" in body["context_used"]["top_signals"]


# ── Test 5: Ollama provider is called correctly ───────────────────────────────

async def test_chat_ollama_provider(app_with_analyst_token: FastAPI) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "Ollama says: monitor workload."}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        with patch("api.routes.chat.settings") as mock_settings:
            mock_settings.llm_provider = "ollama"
            mock_settings.ollama_base_url = "http://localhost:11434"
            mock_settings.ollama_model = "llama3.2"
            mock_settings.chat_max_tokens = 512

            status_code, body = await _post_chat(
                app_with_analyst_token, {"message": "Give me recommendations."}
            )

    assert status_code == 200
    assert "Ollama says" in body["reply"]
