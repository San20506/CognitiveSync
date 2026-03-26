"""Behavioral feature extractor — implementation in Phase 4A (T-037 → T-039).

Computes 13-dimensional feature vectors from anonymized signals per user per 48h window.

Feature Index | Name                  | Source
------------- | --------------------- | -------
0             | meeting_density       | MS Graph
1             | after_hours_meetings  | MS Graph
2             | focus_blocks          | MS Graph
3             | email_response_latency| MS Graph
4             | meeting_accept_rate   | MS Graph
5             | message_volume        | Slack
6             | after_hours_messages  | Slack
7             | response_time_slack   | Slack
8             | mention_frequency     | Slack
9             | commit_frequency      | GitHub
10            | after_hours_commits   | GitHub
11            | pr_review_load        | GitHub
12            | context_switch_rate   | GitHub
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

import numpy as np

from ingestion.anonymizer import AnonymizedSignals

logger = logging.getLogger(__name__)

FEATURE_NAMES: list[str] = [
    "meeting_density",
    "after_hours_meetings",
    "focus_blocks",
    "email_response_latency",
    "meeting_accept_rate",
    "message_volume",
    "after_hours_messages",
    "response_time_slack",
    "mention_frequency",
    "commit_frequency",
    "after_hours_commits",
    "pr_review_load",
    "context_switch_rate",
]

NEUTRAL_BASELINE = 0.5  # Imputed value for missing source features
FEATURE_DIM = len(FEATURE_NAMES)  # 13


@dataclass
class ExtractedFeatures:
    """Feature vector + metadata for a single user per window."""

    pseudo_id: UUID
    feature_vector: np.ndarray  # shape (13,), all values in [0, 1]
    is_imputed: bool
    data_completeness: float  # fraction of features from real data
    window_start: datetime
    window_end: datetime
    # Cross-user interaction signals for graph edge construction
    interactions: dict[UUID, float] = field(default_factory=dict)


@dataclass
class EdgeSignalBatch:
    """Collection of cross-user interaction signals for graph construction."""

    edges: list[tuple[UUID, UUID, float]]  # (source_pseudo_id, target_pseudo_id, weight)


class FeatureExtractor:
    """Computes normalized 13-dim feature vectors from anonymized signals.

    Phase 4A implementation target: T-037, T-038, T-039.
    """

    def extract_batch(
        self,
        signals: dict[UUID, AnonymizedSignals],
        window_start: datetime,
        window_end: datetime,
        rolling_stats: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[list[ExtractedFeatures], EdgeSignalBatch]:
        """Extract features for all users in a batch.

        Args:
            signals: Anonymized signals keyed by pseudo_id.
            window_start: Window start timestamp.
            window_end: Window end timestamp.
            rolling_stats: Optional {feature_name: (min, max)} for normalization.
                           If None, raw values assumed already in [0,1].

        Returns:
            Tuple of (feature list, edge signal batch).

        TODO T-037, T-038, T-039: Full implementation.
        """
        extracted: list[ExtractedFeatures] = []
        edges: list[tuple[UUID, UUID, float]] = []

        for pseudo_id, anon_signals in signals.items():
            raw_values: list[float | None] = [
                anon_signals.meeting_density,
                anon_signals.after_hours_meetings,
                anon_signals.focus_blocks,
                anon_signals.email_response_latency,
                anon_signals.meeting_accept_rate,
                anon_signals.message_volume,
                anon_signals.after_hours_messages,
                anon_signals.response_time_slack,
                anon_signals.mention_frequency,
                anon_signals.commit_frequency,
                anon_signals.after_hours_commits,
                anon_signals.pr_review_load,
                anon_signals.context_switch_rate,
            ]

            normalized: list[float] = []
            none_count = 0
            for value, feature_name in zip(raw_values, FEATURE_NAMES):
                if value is None:
                    normalized.append(NEUTRAL_BASELINE)
                    none_count += 1
                else:
                    normalized.append(self._normalize(value, feature_name, rolling_stats))

            feature_vector = np.array(normalized, dtype=np.float32)
            is_imputed = none_count > 0
            data_completeness = (FEATURE_DIM - none_count) / FEATURE_DIM

            extracted.append(
                ExtractedFeatures(
                    pseudo_id=pseudo_id,
                    feature_vector=feature_vector,
                    is_imputed=is_imputed,
                    data_completeness=data_completeness,
                    window_start=window_start,
                    window_end=window_end,
                    interactions=dict(anon_signals.interactions),
                )
            )

            for target_id, weight in anon_signals.interactions.items():
                edges.append((pseudo_id, target_id, weight))

        logger.debug(
            "extract_batch: processed %d users, %d edges",
            len(extracted),
            len(edges),
        )
        return extracted, EdgeSignalBatch(edges=edges)

    def _normalize(
        self,
        value: float,
        feature_name: str,
        stats: dict[str, tuple[float, float]] | None,
    ) -> float:
        """Min-max normalize a feature value using rolling org statistics.

        TODO T-038: Implement rolling stats management.
        """
        if stats is None or feature_name not in stats:
            return min(max(value, 0.0), 1.0)  # Clamp to [0,1] if no stats
        min_val, max_val = stats[feature_name]
        if max_val - min_val < 1e-8:
            return 0.5  # Degenerate range → neutral
        return (value - min_val) / (max_val - min_val)
