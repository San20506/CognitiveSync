"""Unit tests for ingestion/anonymizer.py — T-032, T-033, T-034, T-035.

Tests UUID v5 determinism and PII-negative invariants.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.adapters.base import RawSignals
from ingestion.anonymizer import AnonymizedSignals, Anonymizer


class TestAnonymizerPseudonymize:
    """Tests for Anonymizer.pseudonymize() — UUID v5 determinism."""

    @pytest.fixture
    def anonymizer(self, tmp_path: Path) -> Anonymizer:
        """Create Anonymizer with fixed org salt for deterministic tests."""
        vault_path = tmp_path / "vault.bin"
        return Anonymizer(org_salt="test-org-salt-12345", vault_path=vault_path, vault_key="test-key")

    def test_same_input_produces_same_uuid(self, anonymizer: Anonymizer) -> None:
        """UUID v5 determinism: identical input + salt → identical UUID."""
        identifier = "alice@company.com"
        result1 = anonymizer.pseudonymize(identifier)
        result2 = anonymizer.pseudonymize(identifier)
        assert result1 == result2
        assert isinstance(result1, uuid.UUID)

    def test_different_inputs_produce_different_uuids(self, anonymizer: Anonymizer) -> None:
        """Different inputs produce different UUIDs."""
        id1 = "alice@company.com"
        id2 = "bob@company.com"
        result1 = anonymizer.pseudonymize(id1)
        result2 = anonymizer.pseudonymize(id2)
        assert result1 != result2

    def test_different_salt_produces_different_uuid(self) -> None:
        """Different org salts produce different UUIDs for same identifier."""
        vault_path = Path("/tmp/vault.bin")
        anon1 = Anonymizer(org_salt="salt-one", vault_path=vault_path, vault_key="key")
        anon2 = Anonymizer(org_salt="salt-two", vault_path=vault_path, vault_key="key")
        identifier = "alice@company.com"
        result1 = anon1.pseudonymize(identifier)
        result2 = anon2.pseudonymize(identifier)
        assert result1 != result2

    def test_pseudonymize_returns_valid_uuid(self, anonymizer: Anonymizer) -> None:
        """Output is a valid UUID object."""
        result = anonymizer.pseudonymize("test-user")
        assert isinstance(result, uuid.UUID)
        # Verify it's a valid UUID (no validation errors)
        str(result)  # Should not raise

    def test_pseudonymize_email_produces_uuid(self, anonymizer: Anonymizer) -> None:
        """Real email addresses are converted to valid UUIDs."""
        emails = [
            "alice@company.com",
            "bob.smith@subdomain.company.org",
            "user+tag@example.com",
        ]
        for email in emails:
            result = anonymizer.pseudonymize(email)
            assert isinstance(result, uuid.UUID)
            assert result != uuid.NAMESPACE_DNS  # Not a nil UUID

    def test_pseudonymize_github_username(self, anonymizer: Anonymizer) -> None:
        """GitHub usernames are converted to valid UUIDs."""
        usernames = ["alice-dev", "bob_smith", "user123"]
        for username in usernames:
            result = anonymizer.pseudonymize(username)
            assert isinstance(result, uuid.UUID)

    def test_pseudonymize_user_id(self, anonymizer: Anonymizer) -> None:
        """Numeric user IDs are converted to valid UUIDs."""
        user_ids = ["12345", "999999999", "0"]
        for user_id in user_ids:
            result = anonymizer.pseudonymize(user_id)
            assert isinstance(result, uuid.UUID)


class TestAnonymizerEdgeCases:
    """Edge case tests for Anonymizer.pseudonymize()."""

    @pytest.fixture
    def anonymizer(self, tmp_path: Path) -> Anonymizer:
        vault_path = tmp_path / "vault.bin"
        return Anonymizer(org_salt="test-salt", vault_path=vault_path, vault_key="key")

    def test_empty_string_produces_uuid(self, anonymizer: Anonymizer) -> None:
        """Empty string input produces a valid (but deterministic) UUID."""
        result = anonymizer.pseudonymize("")
        assert isinstance(result, uuid.UUID)

    def test_empty_string_deterministic(self, anonymizer: Anonymizer) -> None:
        """Empty string always produces the same UUID."""
        result1 = anonymizer.pseudonymize("")
        result2 = anonymizer.pseudonymize("")
        assert result1 == result2

    def test_special_characters_in_identifier(self, anonymizer: Anonymizer) -> None:
        """Special characters are handled correctly."""
        identifiers = [
            "user.name@company.com",
            "user-name@company.com",
            "user_name@company.com",
            "user@company.com#1",
            "user@company.com!",
        ]
        for identifier in identifiers:
            result = anonymizer.pseudonymize(identifier)
            assert isinstance(result, uuid.UUID)
            # Each should be unique
            results = {anonymizer.pseudonymize(i) for i in identifiers}
            assert len(results) == len(identifiers)

    def test_unicode_characters(self, anonymizer: Anonymizer) -> None:
        """Unicode characters are handled correctly."""
        identifiers = [
            "用户@example.com",
            "üser@company.com",
            "José@company.com",
            "🎉user@company.com",
        ]
        for identifier in identifiers:
            result = anonymizer.pseudonymize(identifier)
            assert isinstance(result, uuid.UUID)

    def test_unicode_deterministic(self, anonymizer: Anonymizer) -> None:
        """Unicode identifiers are deterministic."""
        identifier = "用户@example.com"
        result1 = anonymizer.pseudonymize(identifier)
        result2 = anonymizer.pseudonymize(identifier)
        assert result1 == result2

    def test_very_long_identifier(self, anonymizer: Anonymizer) -> None:
        """Very long identifiers are handled correctly."""
        long_id = "a" * 10000
        result = anonymizer.pseudonymize(long_id)
        assert isinstance(result, uuid.UUID)

    def test_none_raises_error(self, anonymizer: Anonymizer) -> None:
        """None input raises an error (cannot encode None)."""
        with pytest.raises((TypeError, AttributeError)):
            anonymizer.pseudonymize(None)  # type: ignore[arg-type]


class TestAnonymizedSignalsPII:
    """Tests for PII-negative invariants in AnonymizedSignals."""

    def test_no_real_identifiers_in_signals(self) -> None:
        """AnonymizedSignals contains no real identifiers — only feature values."""
        signals = AnonymizedSignals(
            meeting_density=0.5,
            after_hours_meetings=0.2,
            focus_blocks=1.0,
            email_response_latency=30.0,
            meeting_accept_rate=0.8,
            message_volume=10.0,
            after_hours_messages=2.0,
            response_time_slack=5.0,
            mention_frequency=3.0,
            commit_frequency=0.5,
            after_hours_commits=0.1,
            pr_review_load=2.0,
            context_switch_rate=0.3,
            interactions={},
        )
        # Convert to dict and verify no email-like strings
        signal_dict = {
            "meeting_density": signals.meeting_density,
            "after_hours_meetings": signals.after_hours_meetings,
            "focus_blocks": signals.focus_blocks,
            "email_response_latency": signals.email_response_latency,
            "meeting_accept_rate": signals.meeting_accept_rate,
            "message_volume": signals.message_volume,
            "after_hours_messages": signals.after_hours_messages,
            "response_time_slack": signals.response_time_slack,
            "mention_frequency": signals.mention_frequency,
            "commit_frequency": signals.commit_frequency,
            "after_hours_commits": signals.after_hours_commits,
            "pr_review_load": signals.pr_review_load,
            "context_switch_rate": signals.context_switch_rate,
            "interactions": signals.interactions,
        }

        for key, value in signal_dict.items():
            if isinstance(value, str):
                assert "@" not in value, f"PII found in {key}: {value}"
                assert not value.endswith(".com"), f"PII found in {key}: {value}"
                assert not value.endswith(".org"), f"PII found in {key}: {value}"

    def test_interactions_keyed_by_uuid_not_string(self) -> None:
        """Interactions dict uses UUID keys, not string identifiers."""
        pseudo_id = uuid.uuid4()
        signals = AnonymizedSignals(
            meeting_density=0.5,
            after_hours_meetings=0.0,
            focus_blocks=0.0,
            email_response_latency=0.0,
            meeting_accept_rate=0.0,
            message_volume=0.0,
            after_hours_messages=0.0,
            response_time_slack=0.0,
            mention_frequency=0.0,
            commit_frequency=0.0,
            after_hours_commits=0.0,
            pr_review_load=0.0,
            context_switch_rate=0.0,
            interactions={pseudo_id: 0.5},
        )
        # Verify interaction keys are UUIDs
        for key in signals.interactions.keys():
            assert isinstance(key, uuid.UUID)


class TestAnonymizedSignalsFromRaw:
    """Tests for AnonymizedSignals.from_raw() method."""

    @pytest.fixture
    def anonymizer(self, tmp_path: Path) -> Anonymizer:
        vault_path = tmp_path / "vault.bin"
        return Anonymizer(org_salt="test-salt", vault_path=vault_path, vault_key="key")

    def test_from_raw_remaps_interaction_keys(self, anonymizer: Anonymizer) -> None:
        """Interaction keys are remapped from real_id to pseudo_id."""
        # Create raw signals with real identifiers in interactions
        raw = RawSignals(
            meeting_density=0.5,
            interactions={"alice@company.com": 0.8, "bob@company.com": 0.3},
        )

        # Build interaction_map (as anonymize_batch does)
        interaction_map = {
            "alice@company.com": anonymizer.pseudonymize("alice@company.com"),
            "bob@company.com": anonymizer.pseudonymize("bob@company.com"),
        }

        # Create anonymized signals
        anon_signals = AnonymizedSignals.from_raw(raw, interaction_map)

        # Verify interaction keys are now UUIDs, not email strings
        for key in anon_signals.interactions.keys():
            assert isinstance(key, uuid.UUID)
            assert not isinstance(key, str)

    def test_from_raw_preserves_feature_values(self, anonymizer: Anonymizer) -> None:
        """Feature values are preserved, only keys are remapped."""
        raw = RawSignals(
            meeting_density=0.75,
            after_hours_meetings=0.25,
            focus_blocks=2.0,
            email_response_latency=45.0,
            meeting_accept_rate=0.9,
            message_volume=15.0,
            after_hours_messages=3.0,
            response_time_slack=10.0,
            mention_frequency=5.0,
            commit_frequency=1.0,
            after_hours_commits=0.2,
            pr_review_load=3.0,
            context_switch_rate=0.4,
            interactions={},
        )

        interaction_map: dict[str, uuid.UUID] = {}
        anon_signals = AnonymizedSignals.from_raw(raw, interaction_map)

        assert anon_signals.meeting_density == 0.75
        assert anon_signals.after_hours_meetings == 0.25
        assert anon_signals.focus_blocks == 2.0
        assert anon_signals.email_response_latency == 45.0
        assert anon_signals.meeting_accept_rate == 0.9
        assert anon_signals.message_volume == 15.0
        assert anon_signals.after_hours_messages == 3.0
        assert anon_signals.response_time_slack == 10.0
        assert anon_signals.mention_frequency == 5.0
        assert anon_signals.commit_frequency == 1.0
        assert anon_signals.after_hours_commits == 0.2
        assert anon_signals.pr_review_load == 3.0
        assert anon_signals.context_switch_rate == 0.4


class TestAnonymizerBatch:
    """Tests for Anonymizer.anonymize_batch() method."""

    @pytest.fixture
    def anonymizer(self, tmp_path: Path) -> Anonymizer:
        vault_path = tmp_path / "vault.bin"
        return Anonymizer(org_salt="test-salt", vault_path=vault_path, vault_key="key")

    @pytest.fixture
    def raw_signals(self) -> dict[str, RawSignals]:
        return {
            "alice@company.com": RawSignals(
                meeting_density=0.5,
                after_hours_meetings=0.1,
                focus_blocks=1.0,
                email_response_latency=30.0,
                meeting_accept_rate=0.8,
                message_volume=10.0,
                after_hours_messages=2.0,
                response_time_slack=5.0,
                mention_frequency=3.0,
                commit_frequency=0.5,
                after_hours_commits=0.1,
                pr_review_load=2.0,
                context_switch_rate=0.3,
                interactions={"bob@company.com": 0.5},
            ),
            "bob@company.com": RawSignals(
                meeting_density=0.6,
                after_hours_meetings=0.2,
                focus_blocks=0.5,
                email_response_latency=45.0,
                meeting_accept_rate=0.9,
                message_volume=8.0,
                after_hours_messages=1.0,
                response_time_slack=3.0,
                mention_frequency=2.0,
                commit_frequency=0.8,
                after_hours_commits=0.2,
                pr_review_load=1.0,
                context_switch_rate=0.2,
                interactions={"alice@company.com": 0.5},
            ),
        }

    def test_batch_output_keyed_by_uuid(self, anonymizer: Anonymizer, raw_signals: dict[str, RawSignals]) -> None:
        """Output dict is keyed by pseudo_id (UUID), not real identifiers."""
        result = anonymizer.anonymize_batch(raw_signals)

        # All keys must be UUIDs
        for key in result.keys():
            assert isinstance(key, uuid.UUID)
            assert not isinstance(key, str)

    def test_batch_no_real_identifiers_in_output(self, anonymizer: Anonymizer, raw_signals: dict[str, RawSignals]) -> None:
        """No real email addresses appear in the output."""
        result = anonymizer.anonymize_batch(raw_signals)

        # Convert entire result to string and check for email patterns
        result_str = str(result)
        assert "alice@company.com" not in result_str
        assert "bob@company.com" not in result_str

    def test_batch_pseudonymization_is_deterministic(self, anonymizer: Anonymizer, raw_signals: dict[str, RawSignals]) -> None:
        """Running batch twice produces identical pseudo_ids."""
        result1 = anonymizer.anonymize_batch(raw_signals)
        result2 = anonymizer.anonymize_batch(raw_signals)

        # Same keys (pseudo_ids)
        assert set(result1.keys()) == set(result2.keys())

    def test_batch_interactions_use_uuid_keys(self, anonymizer: Anonymizer, raw_signals: dict[str, RawSignals]) -> None:
        """All interaction keys in output are UUIDs."""
        result = anonymizer.anonymize_batch(raw_signals)

        for signals in result.values():
            for interaction_key in signals.interactions.keys():
                assert isinstance(interaction_key, uuid.UUID)

    def test_batch_empty_input(self, anonymizer: Anonymizer) -> None:
        """Empty input produces empty output."""
        result = anonymizer.anonymize_batch({})
        assert result == {}

    def test_batch_single_user(self, anonymizer: Anonymizer) -> None:
        """Single user is anonymized correctly."""
        raw_signals = {
            "single@user.com": RawSignals(meeting_density=0.5),
        }
        result = anonymizer.anonymize_batch(raw_signals)

        assert len(result) == 1
        for key in result.keys():
            assert isinstance(key, uuid.UUID)

    def test_batch_calls_vault_upsert(self, anonymizer: Anonymizer, raw_signals: dict[str, RawSignals]) -> None:
        """Anonymizer calls vault.upsert() for each mapping."""
        with patch.object(anonymizer._store, "upsert") as mock_upsert:
            anonymizer.anonymize_batch(raw_signals)
            # Should be called once per user
            assert mock_upsert.call_count == len(raw_signals)


class TestPIIInvariant:
    """Comprehensive PII-negative invariant tests."""

    @pytest.fixture
    def anonymizer(self, tmp_path: Path) -> Anonymizer:
        vault_path = tmp_path / "vault.bin"
        return Anonymizer(org_salt="test-salt", vault_path=vault_path, vault_key="key")

    @pytest.mark.parametrize(
        "email",
        [
            "alice@company.com",
            "bob.smith@subdomain.company.org",
            "admin@company.co.uk",
            "user+tag@example.com",
            "test@company.io",
        ],
    )
    def test_email_never_appears_in_output(self, anonymizer: Anonymizer, email: str) -> None:
        """Real email addresses never appear in anonymized output."""
        raw_signals = {
            email: RawSignals(meeting_density=0.5, interactions={email: 0.5}),
        }
        result = anonymizer.anonymize_batch(raw_signals)

        result_str = str(result)
        assert email not in result_str
        assert "@" not in result_str or result_str.count("@") == 0

    @pytest.mark.parametrize(
        "username",
        [
            "alice_dev",
            "bob_smith",
            "admin_user",
            "testuser123",
        ],
    )
    def test_username_never_appears_in_output(self, anonymizer: Anonymizer, username: str) -> None:
        """Usernames never appear in anonymized output."""
        raw_signals = {
            username: RawSignals(meeting_density=0.5),
        }
        result = anonymizer.anonymize_batch(raw_signals)

        result_str = str(result)
        # The username should not appear as a string key
        # (it may appear in the vault, but not in the output signals)
        assert username not in result_str or result_str.count(username) == 0

    def test_pseudo_ids_are_valid_uuids(self, anonymizer: Anonymizer) -> None:
        """All pseudo_ids in output are valid UUIDs."""
        raw_signals = {
            "user1@company.com": RawSignals(meeting_density=0.5),
            "user2@company.com": RawSignals(meeting_density=0.6),
            "user3@company.com": RawSignals(meeting_density=0.7),
        }
        result = anonymizer.anonymize_batch(raw_signals)

        for pseudo_id in result.keys():
            # Should not raise
            uuid_obj = uuid.UUID(str(pseudo_id))
            assert uuid_obj == pseudo_id