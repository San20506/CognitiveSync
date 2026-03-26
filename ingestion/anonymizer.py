"""PII anonymization engine — implementation in Phase 4A (T-032 → T-035).

PRIVACY CONTRACT:
- All user identifiers stripped before any data reaches downstream components.
- Raw API payloads never written to any persistent store.
- UUID v5 pseudonyms are deterministic (same input + salt → same UUID).
- ID mapping stored exclusively in encrypted volume (AES-256-GCM), never PostgreSQL.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from config.vault import EncryptedMappingStore
from ingestion.adapters.base import RawSignals

logger = logging.getLogger(__name__)


@dataclass
class AnonymizedSignals:
    """RawSignals with PII stripped — safe for feature extraction and storage."""

    # Mirrors RawSignals but keyed by pseudo_id, never real identifier
    meeting_density: float | None
    after_hours_meetings: float | None
    focus_blocks: float | None
    email_response_latency: float | None
    meeting_accept_rate: float | None
    message_volume: float | None
    after_hours_messages: float | None
    response_time_slack: float | None
    mention_frequency: float | None
    commit_frequency: float | None
    after_hours_commits: float | None
    pr_review_load: float | None
    context_switch_rate: float | None
    # Cross-user interactions keyed by pseudo_id (not real id)
    interactions: dict[uuid.UUID, float]

    @classmethod
    def from_raw(
        cls,
        raw: RawSignals,
        interaction_map: dict[str, uuid.UUID],
    ) -> AnonymizedSignals:
        """Create anonymized signals from raw, remapping interaction keys to pseudo_ids."""
        return cls(
            meeting_density=raw.meeting_density,
            after_hours_meetings=raw.after_hours_meetings,
            focus_blocks=raw.focus_blocks,
            email_response_latency=raw.email_response_latency,
            meeting_accept_rate=raw.meeting_accept_rate,
            message_volume=raw.message_volume,
            after_hours_messages=raw.after_hours_messages,
            response_time_slack=raw.response_time_slack,
            mention_frequency=raw.mention_frequency,
            commit_frequency=raw.commit_frequency,
            after_hours_commits=raw.after_hours_commits,
            pr_review_load=raw.pr_review_load,
            context_switch_rate=raw.context_switch_rate,
            # Remap interaction keys from real_id → pseudo_id
            interactions={
                interaction_map[real_id]: weight
                for real_id, weight in raw.interactions.items()
                if real_id in interaction_map
            },
        )


class Anonymizer:
    """SHA-256 + org salt → UUID v5 pseudonymization engine.

    Phase 4A implementation target: T-032, T-033, T-034, T-035.
    """

    def __init__(self, org_salt: str, vault_path: Path, vault_key: str) -> None:
        self._salt = org_salt.encode()
        self._store = EncryptedMappingStore(vault_path, vault_key)

    def pseudonymize(self, identifier: str) -> uuid.UUID:
        """Compute a deterministic UUID from SHA-256(identifier + org_salt).

        Uses the first 16 bytes of the SHA-256 digest as a stable pseudo-ID.
        Deterministic: same input + salt always yields the same UUID.
        """
        digest = hashlib.sha256(identifier.encode() + self._salt).digest()
        # Truncate SHA-256 to 16 bytes and interpret as UUID (no version kwarg —
        # passing version= alongside bytes= does not correctly set version bits)
        return uuid.UUID(bytes=digest[:16])

    def anonymize_batch(
        self,
        raw_signals: dict[str, RawSignals],
    ) -> dict[uuid.UUID, AnonymizedSignals]:
        """Anonymize a batch of raw signals.

        Steps per WF-02:
        1. Pseudonymize all identifiers
        2. Upsert mappings to encrypted volume
        3. Remap interaction keys to pseudo_ids
        4. Build AnonymizedSignals (no real identifiers survive)
        """
        # Step 1 & 3: Pseudonymize all identifiers and build the interaction_map
        # interaction_map maps real_id → pseudo_id for use when remapping interaction keys
        interaction_map: dict[str, uuid.UUID] = {
            real_id: self.pseudonymize(real_id) for real_id in raw_signals
        }

        # Step 2: Upsert every pseudo_id ↔ real_id pair into the encrypted vault
        for real_id, pseudo_id in interaction_map.items():
            self._store.upsert(pseudo_id, real_id)

        # Step 4: Build AnonymizedSignals for each user, keyed by pseudo_id
        result: dict[uuid.UUID, AnonymizedSignals] = {
            interaction_map[real_id]: AnonymizedSignals.from_raw(raw, interaction_map)
            for real_id, raw in raw_signals.items()
        }

        logger.debug("anonymize_batch: processed %d records", len(result))
        return result
