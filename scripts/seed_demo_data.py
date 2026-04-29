"""Seed realistic demo data into CognitiveSync for stakeholder demos.

Run: python scripts/seed_demo_data.py
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4, uuid5, NAMESPACE_DNS

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from config.settings import settings
from ingestion.db.models import (
    BurnoutScore,
    EdgeSignal,
    Employee,
    EmployeeProfile,
    FeatureVector,
)

# ── Demo org structure ──────────────────────────────────────────────────────

TEAMS = {
    "eng-platform":   "Platform Engineering",
    "eng-frontend":   "Frontend Engineering",
    "eng-backend":    "Backend Engineering",
    "product":        "Product Management",
    "design":         "UX Design",
}

ROLES = {
    "eng-platform":  ["Staff Engineer", "Senior SRE", "DevOps Engineer", "SRE"],
    "eng-frontend":  ["Senior FE Engineer", "FE Engineer", "FE Lead", "FE Engineer"],
    "eng-backend":   ["Backend Engineer", "Senior BE Engineer", "Tech Lead", "BE Engineer", "Junior BE"],
    "product":       ["Product Manager", "Senior PM", "Associate PM"],
    "design":        ["UX Designer", "Senior UX Designer", "Design Lead"],
}

SENIORITY = ["junior", "mid", "senior", "staff", "lead"]

FEATURES = [
    "meeting_density",
    "after_hours_meetings",
    "message_volume_spike",
    "focus_time_ratio",
    "weekend_activity",
    "response_latency_drop",
    "collaboration_breadth",
    "pr_review_load",
    "context_switching",
    "vacation_days_unused",
    "late_night_commits",
    "oncall_incidents",
    "slack_after_hours",
]


def _team_uuid(team_id: str) -> UUID:
    return uuid5(NAMESPACE_DNS, f"demo.cognitivesyncs.internal/team/{team_id}")


def _pseudo(name: str) -> UUID:
    return uuid5(NAMESPACE_DNS, f"demo.cognitivesyncs.internal/{name}")


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:64]


def _feature_vec(risk_profile: str) -> dict[str, float]:
    """Generate a plausible 13-dim feature vector keyed by risk profile."""
    base = {f: round(random.uniform(0.1, 0.3), 3) for f in FEATURES}
    if risk_profile == "high":
        base["after_hours_meetings"] = round(random.uniform(0.7, 0.95), 3)
        base["meeting_density"] = round(random.uniform(0.6, 0.85), 3)
        base["weekend_activity"] = round(random.uniform(0.5, 0.8), 3)
        base["focus_time_ratio"] = round(random.uniform(0.05, 0.2), 3)
        base["late_night_commits"] = round(random.uniform(0.5, 0.9), 3)
        base["oncall_incidents"] = round(random.uniform(0.4, 0.8), 3)
    elif risk_profile == "medium":
        base["after_hours_meetings"] = round(random.uniform(0.35, 0.55), 3)
        base["meeting_density"] = round(random.uniform(0.45, 0.65), 3)
        base["focus_time_ratio"] = round(random.uniform(0.2, 0.4), 3)
        base["message_volume_spike"] = round(random.uniform(0.3, 0.55), 3)
    return base


def _top_features(fv: dict[str, float], n: int = 5) -> dict[str, float]:
    return dict(sorted(fv.items(), key=lambda x: x[1], reverse=True)[:n])


# ── Employee pool ───────────────────────────────────────────────────────────

def build_employees() -> list[dict]:
    members = []
    counter = 1
    for team_id, roles in ROLES.items():
        for i, role in enumerate(roles):
            name_key = f"emp-{team_id}-{i+1}"
            pseudo = _pseudo(name_key)
            risk = "high" if counter in {2, 5, 8, 11, 14} else (
                "medium" if counter in {1, 4, 7, 10, 13, 16} else "low"
            )
            members.append({
                "name_key": name_key,
                "pseudo_id": pseudo,
                "display_name_hash": _hash(name_key),
                "team_id": team_id,
                "role": role,
                "seniority": random.choice(SENIORITY),
                "timezone": random.choice(["UTC", "America/New_York", "Europe/London", "Asia/Kolkata"]),
                "risk_profile": risk,
            })
            counter += 1
    return members


# ── Main seed ───────────────────────────────────────────────────────────────

async def seed(db: AsyncSession) -> None:
    random.seed(42)

    now = datetime.now(UTC)
    window_end = now.replace(minute=0, second=0, microsecond=0)
    window_start = window_end - timedelta(hours=48)
    run_id = _pseudo("demo-run-2026-04-29")

    employees = build_employees()
    print(f"Seeding {len(employees)} employees...")

    # ── Employees + profiles ────────────────────────────────────────────────
    for e in employees:
        emp = Employee(
            pseudo_id=e["pseudo_id"],
            display_name_hash=e["display_name_hash"],
            team_id=e["team_id"],
            role=e["role"],
            seniority=e["seniority"],
            timezone=e["timezone"],
            work_hours_start="09:00",
            work_hours_end="18:00",
            is_active=True,
            enrolled_at=now - timedelta(days=90),
            updated_at=now,
        )
        db.add(emp)

    # ── Feature vectors ─────────────────────────────────────────────────────
    for e in employees:
        fv = _feature_vec(e["risk_profile"])
        db.add(FeatureVector(
            pseudo_id=e["pseudo_id"],
            window_start=window_start,
            window_end=window_end,
            feature_json=fv,
            is_imputed=False,
            data_completeness=round(random.uniform(0.85, 1.0), 3),
            created_at=now,
        ))

    # ── Burnout scores ──────────────────────────────────────────────────────
    score_map: dict[UUID, float] = {}
    for e in employees:
        if e["risk_profile"] == "high":
            score = round(random.uniform(0.72, 0.91), 4)
        elif e["risk_profile"] == "medium":
            score = round(random.uniform(0.42, 0.68), 4)
        else:
            score = round(random.uniform(0.08, 0.38), 4)
        score_map[e["pseudo_id"]] = score

    # Identify high-risk sources for cascade
    high_risk_ids = [e["pseudo_id"] for e in employees if e["risk_profile"] == "high"]

    for e in employees:
        pseudo = e["pseudo_id"]
        score = score_map[pseudo]

        # Cascade risk — non-source members near high-risk sources inherit some risk
        cascade_sources_list = []
        cascade_risk = 0.0
        if e["risk_profile"] != "high":
            nearby = [h for h in high_risk_ids if h != pseudo][:2]
            if nearby and random.random() < 0.4:
                cascade_sources_list = [str(h) for h in nearby[:1]]
                cascade_risk = round(score * 0.3, 4)

        fv = _feature_vec(e["risk_profile"])
        top_feat = _top_features(fv)

        db.add(BurnoutScore(
            run_id=run_id,
            pseudo_id=pseudo,
            burnout_score=score,
            confidence_low=round(max(0.0, score - 0.12), 4),
            confidence_high=round(min(1.0, score + 0.12), 4),
            cascade_risk=cascade_risk,
            cascade_sources={"sources": cascade_sources_list} if cascade_sources_list else None,
            top_features=top_feat,
            team_id=_team_uuid(e["team_id"]),
            window_end=window_end,
            created_at=now,
        ))

        trend = []
        for w in range(8, 0, -1):
            past_ts = window_end - timedelta(days=w * 7)
            drift = random.uniform(-0.08, 0.08)
            past_score = round(min(1.0, max(0.0, score + drift)), 4)
            trend.append({"run_id": str(_pseudo(f"run-w{w}")), "score": past_score, "ts": past_ts.isoformat()})

        db.add(EmployeeProfile(
            pseudo_id=pseudo,
            team_id=e["team_id"],
            latest_score=score,
            avg_score_30d=round(sum(t["score"] for t in trend[-4:]) / 4, 4),
            score_trend=trend,
            top_features=top_feat,
            cascade_exposure_count=len(cascade_sources_list),
            run_count=8,
            updated_at=now,
            created_at=now - timedelta(days=90),
        ))

    # ── Edge signals (interaction graph) ────────────────────────────────────
    pseudo_ids = [e["pseudo_id"] for e in employees]
    edges_added = set()
    edge_count = 0

    # Intra-team dense connections
    team_map: dict[str, list[UUID]] = {}
    for e in employees:
        team_map.setdefault(e["team_id"], []).append(e["pseudo_id"])

    for team_members in team_map.values():
        for i, src in enumerate(team_members):
            for tgt in team_members[i+1:]:
                key = (min(src, tgt), max(src, tgt))
                if key not in edges_added:
                    edges_added.add(key)
                    db.add(EdgeSignal(
                        source_pseudo_id=src,
                        target_pseudo_id=tgt,
                        weight=round(random.uniform(0.4, 0.9), 3),
                        window_start=window_start,
                        window_end=window_end,
                    ))
                    edge_count += 1

    # Cross-team sparse connections
    for _ in range(15):
        src, tgt = random.sample(pseudo_ids, 2)
        key = (min(src, tgt), max(src, tgt))
        if key not in edges_added:
            edges_added.add(key)
            db.add(EdgeSignal(
                source_pseudo_id=src,
                target_pseudo_id=tgt,
                weight=round(random.uniform(0.1, 0.4), 3),
                window_start=window_start,
                window_end=window_end,
            ))
            edge_count += 1

    await db.commit()

    # ── Summary ─────────────────────────────────────────────────────────────
    high = sum(1 for s in score_map.values() if s >= 0.70)
    medium = sum(1 for s in score_map.values() if 0.40 <= s < 0.70)
    low = sum(1 for s in score_map.values() if s < 0.40)
    print(f"  Burnout scores: {high} high / {medium} medium / {low} low")
    print(f"  Edge signals: {edge_count}")
    print(f"  Run ID: {run_id}")
    print(f"  Window: {window_start.date()} → {window_end.date()}")
    print("Done. Demo data ready.")


async def main() -> None:
    engine = create_async_engine(str(settings.database_url), echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)  # type: ignore[call-overload]
    async with async_session() as db:
        await seed(db)
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
