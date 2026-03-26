"""Power BI REST API connector — implementation in Phase 4C (T-060).

Pushes scored data to Power BI streaming dataset after each pipeline run.
No Power BI SDK — standard HTTP POST via httpx async client.

Payload schema (per ARCHITECTURE.md §5.3):
    pseudo_id, team_id, burnout_score, cascade_risk, top_features, window_end
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import httpx
import msal  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


@dataclass
class PowerBIRow:
    pseudo_id: UUID
    team_id: UUID | None
    burnout_score: float
    cascade_risk: float
    top_features: dict[str, float]
    window_end: datetime


class PowerBIConnector:
    """Pushes scoring results to Power BI streaming dataset.

    Phase 4C implementation target: T-060.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        tenant_id: str,
        dataset_id: str,
        workspace_id: str,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._tenant_id = tenant_id
        self._dataset_id = dataset_id
        self._workspace_id = workspace_id

    def _acquire_token(self) -> str:
        """Acquire an Azure AD bearer token via service principal client credentials flow."""
        app = msal.ConfidentialClientApplication(
            client_id=self._client_id,
            client_credential=self._client_secret,
            authority=f"https://login.microsoftonline.com/{self._tenant_id}",
        )
        result = app.acquire_token_for_client(
            scopes=["https://analysis.windows.net/powerbi/api/.default"]
        )
        if "access_token" not in result:
            raise RuntimeError(
                f"Power BI token acquisition failed: {result.get('error_description', 'unknown')}"
            )
        return str(result["access_token"])

    async def push_scores(self, rows: list[PowerBIRow]) -> None:
        """POST rows to Power BI streaming dataset endpoint."""
        if not rows:
            return

        token = self._acquire_token()

        payload = {
            "rows": [
                {
                    "pseudo_id": str(row.pseudo_id),
                    "team_id": str(row.team_id) if row.team_id else None,
                    "burnout_score": row.burnout_score,
                    "cascade_risk": row.cascade_risk,
                    "top_features": str(row.top_features),  # JSON string for Power BI
                    "window_end": row.window_end.isoformat(),
                }
                for row in rows
            ]
        }

        url = (
            f"https://api.powerbi.com/v1.0/myorg/groups/{self._workspace_id}"
            f"/datasets/{self._dataset_id}/tables/BurnoutScores/rows"
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()

        logger.info("Pushed %d rows to Power BI dataset %s", len(rows), self._dataset_id)

    async def trigger_refresh(self) -> None:
        """Trigger Power BI dataset refresh to update report visuals."""
        token = self._acquire_token()

        url = (
            f"https://api.powerbi.com/v1.0/myorg/groups/{self._workspace_id}"
            f"/datasets/{self._dataset_id}/refreshes"
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url,
                json={"notifyOption": "NoNotification"},
                headers={"Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()

        logger.info(
            "Power BI dataset refresh triggered for dataset %s", self._dataset_id
        )
