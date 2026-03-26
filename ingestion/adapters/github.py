"""GitHub REST API connector — Phase 4A (T-029).

Signals: commit frequency, after-hours commits, PR review load, context-switch rate.
Auth: Organization-level OAuth token (read:org, read:user, repo read-only).

PyGithub is synchronous; fetch_signals offloads blocking I/O to a thread pool via
asyncio.to_thread so the event loop remains unblocked.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime

from github import Auth, Github
from github.GithubException import GithubException

from ingestion.adapters.base import BaseAdapter, RawSignals

logger = logging.getLogger(__name__)


class GitHubAdapter(BaseAdapter):
    """Fetches behavioral signals from GitHub for all org members in a time window."""

    def __init__(
        self,
        org_token: str,
        org_name: str,
        work_hours_start: str = "09:00",
        work_hours_end: str = "18:00",
    ) -> None:
        self._org_token = org_token
        self._org_name = org_name
        self._work_hours_start = work_hours_start
        self._work_hours_end = work_hours_end

    # ------------------------------------------------------------------
    # Public async interface
    # ------------------------------------------------------------------

    async def fetch_signals(
        self, window_start: datetime, window_end: datetime
    ) -> dict[str, RawSignals]:
        """Fetch GitHub behavioral signals for the given window.

        Offloads synchronous PyGithub calls to a thread-pool worker so the
        event loop stays responsive.
        """
        return await asyncio.to_thread(self._fetch_sync, window_start, window_end)

    async def health_check(self) -> bool:
        """Verify that the org token is valid and the org is accessible."""
        return await asyncio.to_thread(self._health_check_sync)

    # ------------------------------------------------------------------
    # Synchronous implementation (runs inside thread-pool worker)
    # ------------------------------------------------------------------

    def _health_check_sync(self) -> bool:
        try:
            g = Github(auth=Auth.Token(self._org_token))
            org = g.get_organization(self._org_name)
            _ = org.name  # trigger the API call
            return True
        except Exception:
            logger.warning(
                "GitHubAdapter health check failed for org '%s'", self._org_name
            )
            return False

    def _fetch_sync(
        self, window_start: datetime, window_end: datetime
    ) -> dict[str, RawSignals]:
        """Synchronous implementation — called inside asyncio.to_thread."""
        work_start_hour, work_end_hour = self._parse_work_hours()
        weeks = max((window_end - window_start).days / 7, 1.0)

        g = Github(auth=Auth.Token(self._org_token))
        org = g.get_organization(self._org_name)

        # Accumulators keyed by user email
        commit_count: dict[str, int] = defaultdict(int)
        after_hours_count: dict[str, int] = defaultdict(int)
        distinct_repos: dict[str, set[str]] = defaultdict(set)
        pr_review_load: dict[str, int] = defaultdict(int)
        # interaction_count[commenter][pr_author] = number of review comments
        interaction_count: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        repos = org.get_repos(type="all")

        for repo in repos:
            repo_name = repo.full_name

            # ---- Commit signals ----
            try:
                commits = repo.get_commits(since=window_start, until=window_end)
                for commit in commits:
                    email = self._commit_email(commit)
                    if not email:
                        continue
                    commit_count[email] += 1
                    distinct_repos[email].add(repo_name)
                    commit_hour = commit.commit.author.date.hour
                    if commit_hour < work_start_hour or commit_hour >= work_end_hour:
                        after_hours_count[email] += 1
            except GithubException as exc:
                logger.debug(
                    "Skipping commits for repo '%s': %s", repo_name, exc.status
                )

            # ---- PR review load (open PRs with outstanding review requests) ----
            try:
                open_pulls = repo.get_pulls(state="open")
                for pr in open_pulls:
                    try:
                        review_users, _teams = pr.get_review_requests()
                        for reviewer in review_users:
                            email = reviewer.email if reviewer else None
                            if not email:
                                continue
                            pr_review_load[email] += 1
                    except GithubException as exc:
                        logger.debug(
                            "Skipping review requests for PR #%d in '%s': %s",
                            pr.number,
                            repo_name,
                            exc.status,
                        )
            except GithubException as exc:
                logger.debug(
                    "Skipping open PRs for repo '%s': %s", repo_name, exc.status
                )

            # ---- Interaction signals (PR review comments within window) ----
            try:
                all_pulls = repo.get_pulls(
                    state="all", sort="updated", direction="desc"
                )
                for pr in all_pulls:
                    if pr.updated_at < window_start:
                        # PRs are sorted by updated_at descending; stop early
                        break
                    author_email: str | None = pr.user.email if pr.user else None
                    if not author_email:
                        continue
                    try:
                        for comment in pr.get_review_comments():
                            commenter_email = (
                                comment.user.email if comment.user else None
                            )
                            if not commenter_email:
                                continue
                            if commenter_email == author_email:
                                # skip self-interactions
                                continue
                            interaction_count[commenter_email][author_email] += 1
                    except GithubException as exc:
                        logger.debug(
                            "Skipping review comments for PR #%d in '%s': %s",
                            pr.number,
                            repo_name,
                            exc.status,
                        )
            except GithubException as exc:
                logger.debug(
                    "Skipping PRs for interaction scan in repo '%s': %s",
                    repo_name,
                    exc.status,
                )

        # ---- Normalise interaction weights to [0.0, 1.0] ----
        normalised_interactions = self._normalise_interactions(interaction_count)

        # ---- Assemble per-user RawSignals ----
        all_emails: set[str] = (
            set(commit_count)
            | set(pr_review_load)
            | set(interaction_count)
        )

        window_days = max((window_end - window_start).days, 1)

        result: dict[str, RawSignals] = {}
        for email in all_emails:
            commits_total = commit_count.get(email, 0)
            result[email] = RawSignals(
                commit_frequency=commits_total / window_days,
                after_hours_commits=float(after_hours_count.get(email, 0)),
                pr_review_load=float(pr_review_load.get(email, 0)),
                context_switch_rate=(
                    len(distinct_repos[email]) / weeks
                    if email in distinct_repos
                    else None
                ),
                interactions=normalised_interactions.get(email, {}),
            )

        logger.debug(
            "GitHubAdapter collected signals for %d users (window %s → %s)",
            len(result),
            window_start.isoformat(),
            window_end.isoformat(),
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_work_hours(self) -> tuple[int, int]:
        """Parse 'HH:MM' strings into integer hours."""
        start_hour = int(self._work_hours_start.split(":")[0])
        end_hour = int(self._work_hours_end.split(":")[0])
        return start_hour, end_hour

    @staticmethod
    def _commit_email(commit: object) -> str | None:
        """Extract author email from a PyGithub Commit object.

        Tries the GitHub user profile email first (more reliable), then falls
        back to the commit metadata email.
        """
        try:
            # commit.author is the GitHub User object (may be None for unlinked authors)
            if commit.author and commit.author.email:  # type: ignore[union-attr]
                return commit.author.email  # type: ignore[union-attr]
            # Fallback: raw commit metadata
            return commit.commit.author.email or None  # type: ignore[union-attr]
        except AttributeError:
            return None

    @staticmethod
    def _normalise_interactions(
        interaction_count: dict[str, dict[str, int]],
    ) -> dict[str, dict[str, float]]:
        """Normalise raw interaction counts to [0.0, 1.0].

        Finds the global maximum count across all pairs and divides every
        count by it.  If there are no interactions the result is an empty dict.
        """
        if not interaction_count:
            return {}

        global_max = max(
            count
            for targets in interaction_count.values()
            for count in targets.values()
        )

        if global_max == 0:
            return {}

        return {
            commenter: {
                author: count / global_max
                for author, count in targets.items()
            }
            for commenter, targets in interaction_count.items()
        }
