"""MS Teams Bot — proactive alerting via Bot Framework SDK.

Implementation in Phase 4C (T-062).

Bot Framework pattern: ActivityHandler with proactive messaging.
Webhook endpoint: /api/messages on port 3978.
Trigger: WF-01 Step 12 — burnout_score > ALERT_THRESHOLD after pipeline run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)

_BOT_FRAMEWORK_TOKEN_URL = (
    "https://login.microsoftonline.com/botframework.com/oauth2/v2.0/token"
)
_BOT_FRAMEWORK_SCOPE = "https://api.botframework.com/.default"
_SERVICE_URL = "https://smba.trafficmanager.net/apis"


@dataclass
class RiskCluster:
    """Aggregated risk cluster — no individual pseudo_ids exposed in alert."""

    team_count: int
    risk_level: str  # "HIGH" / "MEDIUM"
    top_signals: list[str]  # Signal names only, no individual attribution
    recommendations: list[str]
    cascade_summary: str | None = None


class CognitiveSyncBot:
    """Bot Framework activity handler for proactive HR and Manager alerts.

    Phase 4C implementation target: T-062.

    PRIVACY RULE:
    - HR card: risk cluster summary, top signals, recommendations (no names/IDs)
    - Manager card: redistribution recommendations ONLY — no scores, no pseudo_ids
    """

    def __init__(self, app_id: str, app_password: str) -> None:
        self._app_id = app_id
        self._app_password = app_password

    async def _get_bot_token(self) -> str:
        """Obtain an OAuth2 access token from Bot Framework using client credentials."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                _BOT_FRAMEWORK_TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._app_id,
                    "client_secret": self._app_password,
                    "scope": _BOT_FRAMEWORK_SCOPE,
                },
            )
            resp.raise_for_status()
            return str(resp.json()["access_token"])

    async def send_hr_alert(
        self,
        cluster: RiskCluster,
        hr_channel_id: str,
    ) -> None:
        """Send HR Adaptive Card to configured HR Teams channel.

        The card contains only aggregated cluster-level data — no individual
        names, IDs, or per-person scores are included.
        """
        from config.settings import settings
        from output.teams_bot.cards import build_hr_adaptive_card

        card = build_hr_adaptive_card(cluster, settings.powerbi_dashboard_url)
        token = await self._get_bot_token()

        activity: dict[str, object] = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": card,
                }
            ],
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{_SERVICE_URL}/v3/conversations/{hr_channel_id}/activities",
                json=activity,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
            if resp.status_code not in (200, 201):
                logger.warning(
                    "HR alert send failed: %d %s",
                    resp.status_code,
                    resp.text,
                )
            else:
                logger.info("HR alert sent to channel %s", hr_channel_id)

    async def send_manager_alert(
        self,
        team_id: UUID,
        recommendations: list[str],
        manager_channel_id: str,
    ) -> None:
        """Send Manager Adaptive Card — redistribution guidance only.

        NO individual names, scores, or pseudo_ids in manager card.
        """
        from output.teams_bot.cards import build_manager_adaptive_card

        card = build_manager_adaptive_card(recommendations)
        token = await self._get_bot_token()

        activity: dict[str, object] = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": card,
                }
            ],
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{_SERVICE_URL}/v3/conversations/{manager_channel_id}/activities",
                json=activity,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
            if resp.status_code not in (200, 201):
                logger.warning(
                    "Manager alert failed for team %s: %d",
                    team_id,
                    resp.status_code,
                )
            else:
                logger.info("Manager alert sent for team %s", team_id)
