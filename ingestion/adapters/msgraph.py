"""Microsoft Graph API connector — Phase 4A implementation (T-027).

Signals: calendar density, after-hours activity, email response latency, focus blocks.
Auth: MSAL client credentials flow (Azure AD — requires tenant admin consent).
Scopes: Calendars.Read, Mail.ReadBasic, User.Read.All
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import httpx
import msal  # type: ignore[import-untyped]

from ingestion.adapters.base import BaseAdapter, RawSignals

logger = logging.getLogger(__name__)

_GRAPH_BASE = "https://graph.microsoft.com"
_GRAPH_SCOPE = "https://graph.microsoft.com/.default"

# Response statuses that count as "accepted"
_ACCEPT_WEIGHT: dict[str, float] = {
    "accepted": 1.0,
    "tentativelyAccepted": 0.5,
    "declined": 0.0,
    "none": 0.0,
    "notResponded": 0.0,
    "organizer": 1.0,  # organiser's own meeting always counts
}


class MSGraphAdapter(BaseAdapter):
    """MS Graph connector — fetches calendar and mail signals for all org users."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        tenant_id: str,
        work_hours_start: str = "09:00",
        work_hours_end: str = "18:00",
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._tenant_id = tenant_id
        self._work_hours_start = work_hours_start
        self._work_hours_end = work_hours_end

    # ------------------------------------------------------------------
    # Token acquisition
    # ------------------------------------------------------------------

    def _acquire_token(self) -> str:
        """Acquire an application-level access token via MSAL client credentials."""
        app = msal.ConfidentialClientApplication(
            client_id=self._client_id,
            client_credential=self._client_secret,
            authority=f"https://login.microsoftonline.com/{self._tenant_id}",
        )
        result: dict[str, Any] = app.acquire_token_for_client(scopes=[_GRAPH_SCOPE])
        if "access_token" not in result:
            error = result.get("error_description") or result.get("error") or "unknown"
            raise RuntimeError(f"MSAL token acquisition failed: {error}")
        token: str = result["access_token"]
        return token

    # ------------------------------------------------------------------
    # Paged Graph fetch helper
    # ------------------------------------------------------------------

    @staticmethod
    async def _get_all_pages(
        client: httpx.AsyncClient,
        url: str,
        params: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch all pages following @odata.nextLink until exhausted."""
        items: list[dict[str, Any]] = []
        next_url: str | None = url
        current_params: dict[str, str] | None = params

        while next_url is not None:
            resp = await client.get(next_url, params=current_params)
            resp.raise_for_status()
            body: dict[str, Any] = resp.json()
            items.extend(body.get("value", []))
            next_url = body.get("@odata.nextLink")
            # nextLink already contains query params — don't pass params again
            current_params = None

        return items

    # ------------------------------------------------------------------
    # Work-hour helpers
    # ------------------------------------------------------------------

    def _parse_work_hours(self) -> tuple[int, int]:
        """Return (start_hour, end_hour) as integers parsed from "HH:MM" strings."""
        start_h = int(self._work_hours_start.split(":")[0])
        end_h = int(self._work_hours_end.split(":")[0])
        return start_h, end_h

    # ------------------------------------------------------------------
    # Per-user signal computation
    # ------------------------------------------------------------------

    def _compute_signals(
        self,
        events: list[dict[str, Any]],
        sent_messages: list[dict[str, Any]],
        window_start: datetime,
        window_end: datetime,
    ) -> RawSignals:
        """Derive RawSignals from raw calendar events and sent-mail items."""
        window_days = max((window_end - window_start).days, 1)
        work_start_h, work_end_h = self._parse_work_hours()

        # --- Parse event datetimes ---------------------------------------------------
        parsed_events: list[tuple[datetime, datetime, list[str], str]] = []
        # Each entry: (start_dt, end_dt, attendee_emails, user_response_status)
        for ev in events:
            raw_start = ev.get("start", {})
            raw_end = ev.get("end", {})
            start_str: str = raw_start.get("dateTime", "")
            end_str: str = raw_end.get("dateTime", "")
            if not start_str or not end_str:
                continue

            try:
                # Graph returns datetimes with "Z" suffix or explicit offset
                ev_start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                ev_end = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            except ValueError:
                logger.warning("Unparseable event datetime: %s / %s", start_str, end_str)
                continue

            attendee_emails: list[str] = [
                (a.get("emailAddress") or {}).get("address", "").lower()
                for a in ev.get("attendees", [])
                if (a.get("emailAddress") or {}).get("address")
            ]

            # Determine this user's own response status
            response_status_obj: dict[str, Any] = ev.get("responseStatus") or {}
            if ev.get("isOrganizer"):
                user_response = "organizer"
            else:
                user_response = response_status_obj.get("response", "none").strip()

            parsed_events.append((ev_start, ev_end, attendee_emails, user_response))

        # --- meeting_density ---------------------------------------------------------
        meeting_density = len(parsed_events) / window_days

        # --- after_hours_meetings ----------------------------------------------------
        after_hours_count = 0
        for ev_start, ev_end, _, _ in parsed_events:
            # Use local (UTC) hour — Graph normalises to per-mailbox timezone but we
            # only have UTC here; callers operating in a single-TZ org get correct counts.
            if ev_start.hour < work_start_h or ev_end.hour > work_end_h:
                after_hours_count += 1

        # --- meeting_accept_rate -----------------------------------------------------
        accept_weights: list[float] = []
        for _, _, _, user_response in parsed_events:
            weight = _ACCEPT_WEIGHT.get(user_response, 0.0)
            accept_weights.append(weight)
        meeting_accept_rate: float | None = (
            sum(accept_weights) / len(accept_weights) if accept_weights else None
        )

        # --- focus_blocks ------------------------------------------------------------
        # Sort events by start time; find intra-working-day gaps > 90 minutes
        if parsed_events:
            sorted_events = sorted(parsed_events, key=lambda t: t[0])
            focus_count = 0
            # We iterate over consecutive pairs and measure gap between end of one
            # and start of the next, clamped to working hours.
            for i in range(len(sorted_events) - 1):
                prev_end = sorted_events[i][1]
                next_start = sorted_events[i + 1][0]
                gap_minutes = (next_start - prev_end).total_seconds() / 60.0
                # Only count gaps that fall within working hours on the same day
                if (
                    gap_minutes > 90
                    and prev_end.date() == next_start.date()
                    and work_start_h <= prev_end.hour < work_end_h
                    and work_start_h <= next_start.hour <= work_end_h
                ):
                    focus_count += 1

            # Also count the gap from work-day start to the first meeting
            first_start = sorted_events[0][0]
            day_start_minutes = first_start.hour * 60 + first_start.minute
            work_start_minutes = work_start_h * 60
            if (day_start_minutes - work_start_minutes) > 90:
                focus_count += 1

            focus_blocks: float | None = float(focus_count)
        else:
            # No meetings at all — count working hours as one large focus block per day
            focus_blocks = float(window_days)

        # --- email_response_latency --------------------------------------------------
        # We only have sent mail here; compute latency only when conversation chain
        # gives us multiple sent messages in the same thread (reply latency proxy).
        # For a proper latency calculation we would need received mail too — we set
        # to None when insufficient data rather than break the adapter.
        email_response_latency: float | None = None
        if sent_messages:
            conv_timestamps: dict[str, list[datetime]] = defaultdict(list)
            for msg in sent_messages:
                conv_id: str = msg.get("conversationId", "")
                sent_str: str = msg.get("sentDateTime", "")
                if not conv_id or not sent_str:
                    continue
                try:
                    sent_dt = datetime.fromisoformat(sent_str.replace("Z", "+00:00"))
                    conv_timestamps[conv_id].append(sent_dt)
                except ValueError:
                    logger.warning("Unparseable sentDateTime: %s", sent_str)

            latencies: list[float] = []
            for timestamps in conv_timestamps.values():
                if len(timestamps) < 2:
                    continue
                sorted_ts = sorted(timestamps)
                for j in range(1, len(sorted_ts)):
                    delta = (sorted_ts[j] - sorted_ts[j - 1]).total_seconds() / 60.0
                    if delta > 0:
                        latencies.append(delta)

            if latencies:
                email_response_latency = sum(latencies) / len(latencies)

        # --- interactions ------------------------------------------------------------
        raw_interactions: dict[str, int] = defaultdict(int)
        for _, _, attendees, _ in parsed_events:
            for email in attendees:
                if email:
                    raw_interactions[email] += 1

        total_meetings = len(parsed_events)
        if total_meetings > 0 and raw_interactions:
            interactions: dict[str, float] = {
                email: count / total_meetings
                for email, count in raw_interactions.items()
            }
        else:
            interactions = {}

        return RawSignals(
            meeting_density=meeting_density,
            after_hours_meetings=float(after_hours_count),
            focus_blocks=focus_blocks,
            email_response_latency=email_response_latency,
            meeting_accept_rate=meeting_accept_rate,
            interactions=interactions,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_signals(
        self, window_start: datetime, window_end: datetime
    ) -> dict[str, RawSignals]:
        """Fetch behavioral signals for all org users within the given window.

        Args:
            window_start: Start of rolling window (timezone-aware).
            window_end: End of rolling window (timezone-aware).

        Returns:
            Dict mapping user email to RawSignals.  Users whose Graph calls fail
            are skipped with a warning rather than aborting the entire batch.
        """
        token = self._acquire_token()

        # Format datetimes as ISO 8601 (Graph requires UTC with Z suffix)
        def _fmt(dt: datetime) -> str:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        start_iso = _fmt(window_start)
        end_iso = _fmt(window_end)

        results: dict[str, RawSignals] = {}

        async with httpx.AsyncClient(
            base_url=_GRAPH_BASE,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        ) as client:
            # 1. Enumerate all org users
            try:
                users = await self._get_all_pages(
                    client,
                    "/v1.0/users",
                    params={
                        "$select": "id,mail,displayName",
                        "$top": "999",
                    },
                )
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"Failed to enumerate org users: {exc.response.status_code}"
                ) from exc

            logger.debug("MSGraphAdapter: processing %d users", len(users))

            for user in users:
                user_id: str = user.get("id", "")
                user_email: str = (user.get("mail") or "").lower().strip()

                if not user_id or not user_email:
                    logger.warning(
                        "Skipping user with missing id/mail: %s", user.get("displayName")
                    )
                    continue

                # 2. Calendar events in window
                try:
                    events = await self._get_all_pages(
                        client,
                        f"/v1.0/users/{user_id}/calendarView",
                        params={
                            "startDateTime": start_iso,
                            "endDateTime": end_iso,
                            "$select": (
                                "start,end,attendees,responseStatus,isOrganizer"
                            ),
                        },
                    )
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    logger.warning(
                        "Calendar fetch failed for user %s (HTTP %d) — skipping",
                        user_email,
                        status,
                    )
                    continue

                # 3. Sent mail in window
                try:
                    sent_messages = await self._get_all_pages(
                        client,
                        f"/v1.0/users/{user_id}/messages",
                        params={
                            "$filter": (
                                f"sentDateTime ge {start_iso}"
                                f" and sentDateTime le {end_iso}"
                            ),
                            "$select": "sentDateTime,conversationId,toRecipients",
                        },
                    )
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    logger.warning(
                        "Mail fetch failed for user %s (HTTP %d) — using empty mail list",
                        user_email,
                        status,
                    )
                    sent_messages = []

                results[user_email] = self._compute_signals(
                    events, sent_messages, window_start, window_end
                )

        logger.debug("MSGraphAdapter: signals computed for %d users", len(results))
        return results

    async def health_check(self) -> bool:
        """Verify token acquisition and basic Graph connectivity.

        Returns True if a valid token is obtained and /v1.0/organization responds
        with HTTP 200; False for any connectivity or auth failure (never raises).
        """
        try:
            token = self._acquire_token()
        except RuntimeError as exc:
            logger.warning("MSGraphAdapter health_check: token error — %s", exc)
            return False

        try:
            async with httpx.AsyncClient(
                base_url=_GRAPH_BASE,
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0,
            ) as client:
                resp = await client.get("/v1.0/organization")
                return resp.status_code == 200
        except Exception as exc:  # noqa: BLE001
            logger.warning("MSGraphAdapter health_check: request error — %s", exc)
            return False
