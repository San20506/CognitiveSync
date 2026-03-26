"""Mock adapter — returns synthetic data. Primary adapter for dev, test, and demo.

Used when ADAPTER_MODE=mock (default). Does not make any external API calls.
Implementation in Phase 4A (T-030) after SyntheticOrgGenerator is ready (T-022).
"""

from __future__ import annotations

import logging
from datetime import datetime

from data.synthetic import SyntheticOrgGenerator
from ingestion.adapters.base import BaseAdapter, RawSignals

logger = logging.getLogger(__name__)


class MockAdapter(BaseAdapter):
    """Phase 4A implementation target — T-030.

    Wraps SyntheticOrgGenerator to serve synthetic signals
    through the same interface as real adapters.
    """

    def __init__(self, n_employees: int = 100, seed: int = 42) -> None:
        self._n_employees = n_employees
        self._seed = seed

    async def fetch_signals(
        self, window_start: datetime, window_end: datetime
    ) -> dict[str, RawSignals]:
        """Return synthetic signals from SyntheticOrgGenerator.

        window_start and window_end are intentionally ignored — synthetic data
        has no time dependency.
        """
        generator = SyntheticOrgGenerator(
            n_employees=self._n_employees, seed=self._seed
        )
        graph = generator.generate()
        signals: dict[str, RawSignals] = graph.raw_signals
        logger.debug("MockAdapter returning %d synthetic signals", len(signals))
        return signals

    async def health_check(self) -> bool:
        return True  # Mock is always healthy
