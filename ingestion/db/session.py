"""SQLAlchemy 2.x async session factories.

Main engine: feature store + scores.
Audit engine: separate database for GDPR Art. 30 audit trail (append-only).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config.settings import settings

# --- Main engine (features, scores, config schemas) ---
engine = create_async_engine(
    str(settings.database_url),
    echo=False,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields an async DB session per request."""
    async with AsyncSessionLocal() as session:
        yield session


# --- Audit engine (separate database for tamper-evident audit log) ---
audit_engine = create_async_engine(
    str(settings.audit_database_url),
    echo=False,
    pool_pre_ping=True,
)

AuditSessionLocal = async_sessionmaker(
    bind=audit_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_audit_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields an audit DB session per request."""
    async with AuditSessionLocal() as session:
        yield session
