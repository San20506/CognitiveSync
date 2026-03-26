"""Synthetic org graph and feature vector generator.

Implementation in Phase 4A (T-022 → T-025).

Design spec (T-014):
- N configurable employees (default 100, demo 200, perf test 500)
- Team structure: 5-10 teams, each with 1 manager
- Burnout fraction: ~15% of employees (configurable)
- Feature distributions:
    * Normal-risk employees: features ~ N(0.3, 0.1), clipped to [0,1]
    * At-risk employees (burnout=1): 3+ features elevated to N(0.8, 0.1)
    * Specifically elevated: after_hours_*, meeting_density, pr_review_load
    * Specifically depressed: focus_blocks, meeting_accept_rate
- Edge density: within-team ~40%, cross-team ~5%
- Edge weights: interaction_score ~ U[0.1, 1.0]
- Labels: burnout=1 if 3+ features in 90th percentile of the generated distribution
- Seed: configurable for reproducibility

All identifiers are synthetic UUID4 — no real PII used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import numpy as np

from ingestion.adapters.base import RawSignals

logger = logging.getLogger(__name__)

# Feature names must match FEATURE_NAMES in feature_extractor.py
FEATURE_NAMES = [
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

# Features elevated in burnout-positive employees
BURNOUT_HIGH_FEATURES = {
    "meeting_density", "after_hours_meetings", "email_response_latency",
    "message_volume", "after_hours_messages", "mention_frequency",
    "after_hours_commits", "pr_review_load", "context_switch_rate",
}

# Features depressed in burnout-positive employees
BURNOUT_LOW_FEATURES = {"focus_blocks", "meeting_accept_rate"}


@dataclass
class SyntheticEmployee:
    pseudo_id: UUID
    team_id: UUID
    is_manager: bool = False
    burnout_label: int = 0  # 0 or 1


@dataclass
class SyntheticOrgGraph:
    """Complete synthetic org graph ready for pipeline input."""

    employees: list[SyntheticEmployee]
    raw_signals: dict[str, RawSignals]     # keyed by str(pseudo_id) — simulates adapter output
    feature_matrix: np.ndarray             # shape (N, 13)
    labels: np.ndarray                     # shape (N,) binary
    edges: list[tuple[UUID, UUID, float]]  # (src, dst, weight)
    team_map: dict[UUID, list[UUID]]       # team_id → list of pseudo_ids


class SyntheticOrgGenerator:
    """Generates synthetic org graphs for development and testing.

    Phase 4A implementation target: T-022, T-023, T-024, T-025.
    """

    def __init__(
        self,
        n_employees: int = 100,
        burnout_fraction: float = 0.15,
        n_teams: int | None = None,
        seed: int = 42,
    ) -> None:
        self.n_employees = n_employees
        self.burnout_fraction = burnout_fraction
        self.n_teams = n_teams or max(3, n_employees // 15)
        self.rng = np.random.default_rng(seed)

    def generate(self) -> SyntheticOrgGraph:
        """Generate a complete synthetic org graph.

        Returns SyntheticOrgGraph with all components for pipeline testing.

        Implements T-022 → T-025.
        """
        employees, team_map = self._generate_employees()
        feature_matrix = self._generate_feature_matrix(employees)
        labels = self._generate_labels(feature_matrix)

        # Assign burnout labels back to employee objects
        for i, employee in enumerate(employees):
            employee.burnout_label = int(labels[i])

        edges = self._generate_edges(employees, team_map)

        # Build raw_signals dict keyed by str(pseudo_id)
        raw_signals: dict[str, RawSignals] = {}
        for i, employee in enumerate(employees):
            feature_vec = feature_matrix[i]
            feature_kwargs = {
                FEATURE_NAMES[j]: float(feature_vec[j])
                for j in range(len(FEATURE_NAMES))
            }
            raw_signals[str(employee.pseudo_id)] = RawSignals(
                **feature_kwargs,
                interactions={},
            )

        logger.debug(
            "Generated synthetic org: %d employees, %d teams, %d edges, %d burnout",
            len(employees),
            len(team_map),
            len(edges),
            int(labels.sum()),
        )

        return SyntheticOrgGraph(
            employees=employees,
            raw_signals=raw_signals,
            feature_matrix=feature_matrix,
            labels=labels,
            edges=edges,
            team_map=team_map,
        )

    def _generate_employees(self) -> tuple[list[SyntheticEmployee], dict[UUID, list[UUID]]]:
        """T-022: Generate employee list with team assignments.

        Creates N employees distributed across n_teams.  Each team gets exactly
        one manager.  Returns the employee list and a team_id → [pseudo_id] map.
        """
        # Assign each employee to a team (roughly even distribution)
        team_ids: list[UUID] = [uuid4() for _ in range(self.n_teams)]
        team_assignments = self.rng.integers(0, self.n_teams, size=self.n_employees)

        # Build team_map first so we know which index is the first per team
        team_map: dict[UUID, list[UUID]] = {tid: [] for tid in team_ids}
        employees: list[SyntheticEmployee] = []
        team_has_manager: dict[UUID, bool] = {tid: False for tid in team_ids}

        for i in range(self.n_employees):
            pseudo_id = uuid4()
            team_id = team_ids[int(team_assignments[i])]

            # First employee assigned to each team becomes the manager
            is_manager = not team_has_manager[team_id]
            if is_manager:
                team_has_manager[team_id] = True

            employee = SyntheticEmployee(
                pseudo_id=pseudo_id,
                team_id=team_id,
                is_manager=is_manager,
            )
            employees.append(employee)
            team_map[team_id].append(pseudo_id)

        logger.debug(
            "Generated %d employees across %d teams", len(employees), self.n_teams
        )
        return employees, team_map

    def _generate_feature_matrix(
        self, employees: list[SyntheticEmployee]
    ) -> np.ndarray:
        """T-023: Generate (N, 13) feature matrix with burnout-correlated distributions.

        Normal-risk employees: each feature ~ N(0.3, 0.1), clipped to [0, 1].
        At-risk employees (burnout_label=1):
          - BURNOUT_HIGH_FEATURES → N(0.8, 0.1), clipped to [0, 1]
          - BURNOUT_LOW_FEATURES  → N(0.2, 0.1), clipped to [0, 1]
          - Other features        → N(0.3, 0.1), clipped to [0, 1]

        Burnout fraction is determined by self.burnout_fraction applied to the
        employee list in order (first burnout_fraction * N employees are at-risk).
        """
        n = len(employees)
        n_features = len(FEATURE_NAMES)
        n_burnout = int(round(n * self.burnout_fraction))

        # Mark employees as at-risk (first n_burnout in population for reproducibility)
        at_risk_flags = np.zeros(n, dtype=bool)
        at_risk_flags[:n_burnout] = True
        # Shuffle so at-risk employees are spread across the list
        self.rng.shuffle(at_risk_flags)

        # Write burnout_label onto employees so _generate_labels has a cross-check,
        # but labels are officially determined by _generate_labels (90th-percentile rule).
        # We use the flags here only to shape the distribution.
        for i, employee in enumerate(employees):
            employee.burnout_label = int(at_risk_flags[i])

        # Base matrix: all normal-risk distribution
        matrix = self.rng.normal(loc=0.3, scale=0.1, size=(n, n_features))

        # Overwrite at-risk rows feature-by-feature
        for j, fname in enumerate(FEATURE_NAMES):
            at_risk_rows = at_risk_flags

            if fname in BURNOUT_HIGH_FEATURES:
                elevated = self.rng.normal(loc=0.8, scale=0.1, size=at_risk_rows.sum())
                matrix[at_risk_rows, j] = elevated
            elif fname in BURNOUT_LOW_FEATURES:
                depressed = self.rng.normal(loc=0.2, scale=0.1, size=at_risk_rows.sum())
                matrix[at_risk_rows, j] = depressed
            # else: already set to N(0.3, 0.1) from base draw

        # Clip entire matrix to [0, 1]
        matrix = np.clip(matrix, 0.0, 1.0)

        logger.debug("Generated feature matrix shape %s", matrix.shape)
        return matrix

    def _generate_labels(self, feature_matrix: np.ndarray) -> np.ndarray:
        """T-024: Rule-based burnout labels — 1 if 3+ features in 90th percentile.

        Computes the 90th percentile threshold per feature column across all rows.
        A row is labelled burnout=1 if at least 3 of its feature values exceed
        their respective column thresholds.
        """
        # Per-column 90th percentile thresholds — shape (13,)
        thresholds = np.percentile(feature_matrix, 90, axis=0)

        # Boolean mask: which (row, col) pairs are above threshold — shape (N, 13)
        above = feature_matrix > thresholds  # type: ignore[operator]

        # Count features above threshold per row — shape (N,)
        counts = above.sum(axis=1)

        labels = (counts >= 3).astype(np.int32)
        logger.debug(
            "Generated labels: %d burnout / %d total (%.1f%%)",
            int(labels.sum()),
            len(labels),
            100.0 * labels.mean(),
        )
        return labels

    def _generate_edges(
        self, employees: list[SyntheticEmployee], team_map: dict[UUID, list[UUID]]
    ) -> list[tuple[UUID, UUID, float]]:
        """T-025: Generate directed interaction edges with within/cross-team density.

        Within-team pairs: ~40% edge probability per directed direction.
        Cross-team pairs:  ~5% edge probability per directed direction.
        Edge weight: interaction_score ~ U[0.1, 1.0].
        """
        WITHIN_TEAM_DENSITY = 0.40
        CROSS_TEAM_DENSITY = 0.05

        # Build a lookup: pseudo_id → team_id
        id_to_team: dict[UUID, UUID] = {
            emp.pseudo_id: emp.team_id for emp in employees
        }

        edges: list[tuple[UUID, UUID, float]] = []
        pseudo_ids = [emp.pseudo_id for emp in employees]
        n = len(pseudo_ids)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                src = pseudo_ids[i]
                dst = pseudo_ids[j]

                same_team = id_to_team[src] == id_to_team[dst]
                density = WITHIN_TEAM_DENSITY if same_team else CROSS_TEAM_DENSITY

                if self.rng.random() < density:
                    weight = float(self.rng.uniform(0.1, 1.0))
                    edges.append((src, dst, weight))

        logger.debug("Generated %d directed edges", len(edges))
        return edges
