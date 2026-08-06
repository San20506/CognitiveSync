"""Tests for GDPR erasure, audit logging, and hash chain integrity."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from api.services.audit_service import _ZERO_HASH, _compute_event_hash, log_event


# ---------------------------------------------------------------------------
# Hash chain unit tests
# ---------------------------------------------------------------------------


class TestComputeEventHash:
    def test_deterministic(self) -> None:
        args = (
            "abc123",
            uuid4(),
            "erased",
            "Employee",
            uuid4(),
            datetime(2026, 1, 1, tzinfo=UTC),
        )
        h1 = _compute_event_hash(*args)
        h2 = _compute_event_hash(*args)
        assert h1 == h2

    def test_different_inputs_produce_different_hashes(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        pid = uuid4()
        h1 = _compute_event_hash(None, pid, "erased", "Employee", pid, now)
        h2 = _compute_event_hash(None, pid, "purged", "Employee", pid, now)
        assert h1 != h2

    def test_chain_links_to_previous(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        pid = uuid4()
        h1 = _compute_event_hash(None, pid, "erased", "Employee", pid, now)
        h2 = _compute_event_hash(h1, pid, "purged", "Employee", pid, now)
        assert h2 != h1
        assert h1 in hashlib.sha256(
            f"{_ZERO_HASH}|{now.isoformat()}|{pid}|erased|Employee|{pid}".encode()
        ).hexdigest()

    def test_zero_hash_used_when_prev_is_none(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        pid = uuid4()
        result = _compute_event_hash(None, pid, "erased", "Employee", pid, now)
        expected = hashlib.sha256(
            f"{_ZERO_HASH}|{now.isoformat()}|{pid}|erased|Employee|{pid}".encode()
        ).hexdigest()
        assert result == expected

    def test_produces_64_char_hex(self) -> None:
        result = _compute_event_hash(
            None, uuid4(), "test", "Resource", None, datetime.now(UTC)
        )
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


# ---------------------------------------------------------------------------
# log_event integration (mocked DB)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLogEvent:
    async def test_creates_audit_event_with_hash(self) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = None
        mock_db.execute.return_value = mock_result

        event = await log_event(
            mock_db,
            event_type="gdpr.erasure",
            actor_id=uuid4(),
            action="erased",
            resource_type="Employee",
            resource_id=uuid4(),
            payload={"deleted_scores": 5},
        )

        mock_db.add.assert_called_once()
        added_event = mock_db.add.call_args[0][0]
        assert added_event.event_type == "gdpr.erasure"
        assert added_event.action == "erased"
        assert added_event.resource_type == "Employee"
        assert len(added_event.event_hash) == 64
        assert added_event.prev_hash is None

    async def test_chains_to_previous_event(self) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = "abc123def456"
        mock_db.execute.return_value = mock_result

        event = await log_event(
            mock_db,
            event_type="gdpr.retention_purge",
            actor_id=uuid4(),
            action="purged",
            resource_type="BurnoutScore",
            payload={"count": 10},
        )

        added_event = mock_db.add.call_args[0][0]
        assert added_event.prev_hash == "abc123def456"

    async def test_flush_called_but_not_commit(self) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = None
        mock_db.execute.return_value = mock_result

        await log_event(
            mock_db,
            event_type="gdpr.test",
            actor_id=uuid4(),
            action="tested",
            resource_type="TestResource",
            payload={},
        )

        mock_db.flush.assert_called_once()
        mock_db.commit.assert_not_called()


# ---------------------------------------------------------------------------
# Vault purge unit test
# ---------------------------------------------------------------------------


class TestVaultPurge:
    def test_purge_removes_mapping(self, tmp_path: object) -> None:
        from config.vault import EncryptedMappingStore

        vault_path = tmp_path / "test_vault.enc"  # type: ignore[operator]
        store = EncryptedMappingStore(vault_path, "test-master-key-123")  # type: ignore[arg-type]

        pid = uuid4()
        store.upsert(pid, "real-user@example.com")
        assert store.lookup(pid) == "real-user@example.com"

        store.purge(pid)
        assert store.lookup(pid) is None

    def test_purge_nonexistent_is_noop(self, tmp_path: object) -> None:
        from config.vault import EncryptedMappingStore

        vault_path = tmp_path / "test_vault2.enc"  # type: ignore[operator]
        store = EncryptedMappingStore(vault_path, "test-master-key-456")  # type: ignore[arg-type]

        store.purge(uuid4())
        assert store.lookup(uuid4()) is None
