"""Slack Events API connector — T-028.

Signals: message volume, after-hours messages, response times, mention frequency.
Auth: Slack workspace OAuth bot token (AsyncWebClient).
Scopes: channels:history, im:history, users:read, team:info
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from ingestion.adapters.base import BaseAdapter, RawSignals

logger = logging.getLogger(__name__)


class SlackAdapter(BaseAdapter):
    """Slack behavioral signal adapter — T-028."""

    def __init__(
        self,
        bot_token: str,
        signing_secret: str,
        work_hours_start: str = "09:00",
        work_hours_end: str = "18:00",
    ) -> None:
        self._bot_token = bot_token
        self._signing_secret = signing_secret
        self._work_hours_start = work_hours_start
        self._work_hours_end = work_hours_end

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def fetch_signals(
        self, window_start: datetime, window_end: datetime
    ) -> dict[str, RawSignals]:
        """Fetch behavioral signals for all org users in the given window."""
        client = AsyncWebClient(token=self._bot_token)

        start_hour, start_minute = self._parse_hour_minute(self._work_hours_start)
        end_hour, end_minute = self._parse_hour_minute(self._work_hours_end)

        window_days = max(
            (window_end - window_start).total_seconds() / 86400.0, 1.0
        )
        oldest = str(window_start.timestamp())
        latest = str(window_end.timestamp())

        # --- 1. Fetch all non-bot users ---
        uid_to_email: dict[str, str] = await self._fetch_users(client)
        if not uid_to_email:
            logger.warning("SlackAdapter: no users returned from users_list")
            return {}

        # Reverse map for convenience
        all_user_ids: set[str] = set(uid_to_email.keys())

        # --- 2. Accumulators keyed by Slack user ID ---
        msg_count: dict[str, int] = defaultdict(int)
        after_hours_count: dict[str, int] = defaultdict(int)
        mention_count: dict[str, int] = defaultdict(int)
        # interactions: sender_uid -> {other_uid: count}
        interactions_raw: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # --- 3. Public / private channels ---
        channels = await self._fetch_conversations(client, types="public_channel,private_channel")
        for ch in channels:
            ch_id: str = ch["id"]
            messages = await self._fetch_history(client, ch_id, oldest, latest)
            await asyncio.sleep(1)  # rate-limit courtesy pause

            for msg in messages:
                sender_uid: str | None = msg.get("user")
                if not sender_uid or sender_uid not in all_user_ids:
                    continue

                msg_count[sender_uid] += 1

                ts_float = float(msg.get("ts", "0"))
                if self._is_after_hours(ts_float, start_hour, start_minute, end_hour, end_minute):
                    after_hours_count[sender_uid] += 1

                # Mentions: scan message text for <@UID> patterns
                text: str = msg.get("text", "")
                for target_uid in all_user_ids:
                    if target_uid == sender_uid:
                        continue
                    if f"<@{target_uid}>" in text:
                        mention_count[target_uid] += 1

                # Interactions: count messages from sender toward channel members
                # We record sender -> other members present in the channel.
                # We proxy "members in channel" via the members field if small, or
                # simply record per-message that sender interacted with mentioned users.
                # For weight we count explicit mentions + general channel presence.
                if text:
                    for target_uid in all_user_ids:
                        if target_uid != sender_uid and f"<@{target_uid}>" in text:
                            interactions_raw[sender_uid][target_uid] += 2  # weighted higher for mention
                            interactions_raw[target_uid][sender_uid] += 1  # implicit awareness

        # --- 4. DM (IM) channels — response time ---
        dm_response_times: dict[str, list[float]] = defaultdict(list)

        im_channels = await self._fetch_conversations(client, types="im")
        for ch in im_channels:
            ch_id = ch["id"]
            # ch["user"] is the *other* participant (not the bot, but the human)
            other_uid: str | None = ch.get("user")
            messages = await self._fetch_history(client, ch_id, oldest, latest)
            await asyncio.sleep(1)

            if not messages or other_uid is None:
                continue

            # Sort messages ascending by timestamp
            messages_sorted = sorted(messages, key=lambda m: float(m.get("ts", "0")))

            # Build response-time pairs for each user in the DM
            self._extract_dm_response_times(
                messages_sorted, all_user_ids, dm_response_times
            )

            # Also count volume + after-hours for DMs
            for msg in messages_sorted:
                sender_uid = msg.get("user")
                if not sender_uid or sender_uid not in all_user_ids:
                    continue

                msg_count[sender_uid] += 1
                ts_float = float(msg.get("ts", "0"))
                if self._is_after_hours(ts_float, start_hour, start_minute, end_hour, end_minute):
                    after_hours_count[sender_uid] += 1

                # Interaction weight for DMs
                if other_uid and other_uid in all_user_ids and other_uid != sender_uid:
                    interactions_raw[sender_uid][other_uid] += 1

        # --- 5. Assemble RawSignals per user (keyed by email) ---
        result: dict[str, RawSignals] = {}

        for uid, email in uid_to_email.items():
            total_msgs = msg_count.get(uid, 0)
            total_after = after_hours_count.get(uid, 0)
            total_mentions = mention_count.get(uid, 0)

            # Compute avg DM response time in minutes
            rt_list = dm_response_times.get(uid)
            response_time: float | None = (
                sum(rt_list) / len(rt_list) if rt_list else None
            )

            # Build normalized interactions dict keyed by email
            raw_inter = interactions_raw.get(uid, {})
            interactions_email: dict[str, float] = {}
            if raw_inter:
                total_inter = sum(raw_inter.values())
                if total_inter > 0:
                    for other_uid, count in raw_inter.items():
                        other_email = uid_to_email.get(other_uid)
                        if other_email:
                            interactions_email[other_email] = count / total_inter

            result[email] = RawSignals(
                message_volume=total_msgs / window_days,
                after_hours_messages=float(total_after),
                response_time_slack=response_time,
                mention_frequency=total_mentions / window_days,
                interactions=interactions_email,
            )

        return result

    async def health_check(self) -> bool:
        """Verify bot token and workspace connectivity via auth.test."""
        client = AsyncWebClient(token=self._bot_token)
        try:
            resp = await client.auth_test()
            return bool(resp.get("ok", False))
        except SlackApiError as exc:
            logger.warning("SlackAdapter health_check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _fetch_users(self, client: AsyncWebClient) -> dict[str, str]:
        """Return {slack_uid: email} for all non-bot org members."""
        uid_to_email: dict[str, str] = {}
        cursor: str | None = None

        while True:
            try:
                kwargs: dict[str, Any] = {"limit": 200}
                if cursor:
                    kwargs["cursor"] = cursor
                resp = await client.users_list(**kwargs)
            except SlackApiError as exc:
                raise RuntimeError(f"SlackAdapter: users_list failed — {exc}") from exc

            for member in resp.get("members", []):
                if member.get("is_bot", False) or member.get("deleted", False):
                    continue
                if member.get("id") == "USLACKBOT":
                    continue
                uid: str = member["id"]
                email: str | None = (
                    member.get("profile", {}).get("email")
                )
                if uid and email:
                    uid_to_email[uid] = email

            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        return uid_to_email

    async def _fetch_conversations(
        self, client: AsyncWebClient, types: str
    ) -> list[dict[str, Any]]:
        """Return all conversations of the given types, handling pagination."""
        channels: list[dict[str, Any]] = []
        cursor: str | None = None

        while True:
            try:
                kwargs: dict[str, Any] = {"types": types, "limit": 200}
                if cursor:
                    kwargs["cursor"] = cursor
                resp = await client.conversations_list(**kwargs)
            except SlackApiError as exc:
                logger.warning("SlackAdapter: conversations_list(%s) failed: %s", types, exc)
                break

            channels.extend(resp.get("channels", []))
            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        return channels

    async def _fetch_history(
        self,
        client: AsyncWebClient,
        channel_id: str,
        oldest: str,
        latest: str,
    ) -> list[dict[str, Any]]:
        """Return all messages in a channel within the time window."""
        messages: list[dict[str, Any]] = []
        cursor: str | None = None

        while True:
            try:
                kwargs: dict[str, Any] = {
                    "channel": channel_id,
                    "oldest": oldest,
                    "latest": latest,
                    "limit": 200,
                }
                if cursor:
                    kwargs["cursor"] = cursor
                resp = await client.conversations_history(**kwargs)
            except SlackApiError as exc:
                logger.warning(
                    "SlackAdapter: conversations_history(%s) failed: %s", channel_id, exc
                )
                break

            messages.extend(resp.get("messages", []))
            if not resp.get("has_more", False):
                break
            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        return messages

    @staticmethod
    def _parse_hour_minute(hhmm: str) -> tuple[int, int]:
        """Parse 'HH:MM' string into (hour, minute) ints."""
        parts = hhmm.split(":")
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0

    @staticmethod
    def _is_after_hours(
        ts: float,
        start_hour: int,
        start_minute: int,
        end_hour: int,
        end_minute: int,
    ) -> bool:
        """Return True if the timestamp falls outside work hours (UTC)."""
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        work_start_minutes = start_hour * 60 + start_minute
        work_end_minutes = end_hour * 60 + end_minute
        current_minutes = dt.hour * 60 + dt.minute
        return current_minutes < work_start_minutes or current_minutes >= work_end_minutes

    @staticmethod
    def _extract_dm_response_times(
        messages: list[dict[str, Any]],
        valid_user_ids: set[str],
        dm_response_times: dict[str, list[float]],
    ) -> None:
        """Populate dm_response_times with per-user avg DM response deltas.

        For each pair (incoming message from A → reply from B), record the
        time delta in minutes against B's response-time list.
        """
        # Track last incoming message per sender so we can detect a reply
        # last_incoming[uid] = (sender_uid, ts_float)
        last_incoming: dict[str, tuple[str, float]] = {}

        for msg in messages:
            sender: str | None = msg.get("user")
            if not sender or sender not in valid_user_ids:
                continue

            ts = float(msg.get("ts", "0"))

            # Check if this message is a reply to a previous message from someone else
            if sender in last_incoming:
                incoming_sender, incoming_ts = last_incoming[sender]
                if incoming_sender != sender:
                    # sender received a message from incoming_sender and now replied
                    delta_minutes = (ts - incoming_ts) / 60.0
                    if delta_minutes >= 0:
                        dm_response_times[sender].append(delta_minutes)
                # Clear after consuming the pair
                del last_incoming[sender]

            # Record this message as the latest incoming for the other party
            # In a DM, every message is "incoming" for all other participants
            for uid in valid_user_ids:
                if uid != sender:
                    last_incoming[uid] = (sender, ts)
