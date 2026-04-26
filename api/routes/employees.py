"""Employee enrollment endpoints — T-060, T-062."""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import TokenPayload
from api.middleware.rbac import require_role
from api.schemas.common import UserRole
from api.schemas.enrollment import (
    EmployeeResponse,
    EnrollRequest,
    EnrollResponse,
    InitialProfileSeed,
)
from ingestion.db.models import Employee, EmployeeProfile
from ingestion.db.session import get_db

logger = logging.getLogger(__name__)
router = APIRouter()

ENROLLMENT_TEMPLATE_PATH = "docs/enrollment_template.json"


def _hash_stressors(stressors: list[str]) -> list[str]:
    return [hashlib.sha256(s.encode()).hexdigest()[:16] for s in stressors]


def _seed_to_dict(seed: InitialProfileSeed) -> dict:  # type: ignore[type-arg]
    return {
        "role_risk_modifier": seed.role_risk_modifier,
        "work_hours_start": seed.work_hours_start,
        "work_hours_end": seed.work_hours_end,
        "expected_after_hours_ratio": seed.expected_after_hours_ratio,
        "known_stressors": _hash_stressors(seed.known_stressors),
        "notes_hash": seed.notes_hash,
        "seeded_at": datetime.now(UTC).isoformat(),
    }


@router.post(
    "/employees/enroll",
    response_model=EnrollResponse,
    status_code=status.HTTP_201_CREATED,
)
async def enroll_employee(
    payload: EnrollRequest,
    token: TokenPayload = require_role(UserRole.IT_ADMIN, UserRole.HR_ADMIN),
    db: AsyncSession = Depends(get_db),
) -> EnrollResponse:
    """Register an employee in the system before their data enters the pipeline.

    Idempotent — re-enrolling an existing pseudo_id updates their record
    rather than creating a duplicate.

    Access: IT Admin, HR Admin.
    """
    now = datetime.now(UTC)

    # Upsert Employee record
    result = await db.execute(select(Employee).where(Employee.pseudo_id == payload.pseudo_id))
    emp = result.scalar_one_or_none()

    if emp is None:
        emp = Employee(
            pseudo_id=payload.pseudo_id,
            display_name_hash=payload.display_name_hash,
            team_id=payload.team_id,
            role=payload.role,
            seniority=payload.seniority,
            timezone=payload.timezone,
            work_hours_start=payload.initial_profile.work_hours_start
            if payload.initial_profile
            else "09:00",
            work_hours_end=payload.initial_profile.work_hours_end
            if payload.initial_profile
            else "18:00",
            is_active=True,
            enrolled_at=now,
            updated_at=now,
        )
        db.add(emp)
        logger.info("Enrolled new employee pseudo_id=%s", payload.pseudo_id)
    else:
        emp.team_id = payload.team_id or emp.team_id
        emp.role = payload.role or emp.role
        emp.seniority = payload.seniority or emp.seniority
        emp.timezone = payload.timezone
        emp.updated_at = now
        logger.info("Updated existing employee pseudo_id=%s", payload.pseudo_id)

    # Upsert EmployeeProfile — create if absent, seed if template provided
    prof_result = await db.execute(
        select(EmployeeProfile).where(EmployeeProfile.pseudo_id == payload.pseudo_id)
    )
    prof = prof_result.scalar_one_or_none()
    profile_seeded = False

    if prof is None:
        seed_data = _seed_to_dict(payload.initial_profile) if payload.initial_profile else None
        prof = EmployeeProfile(
            pseudo_id=payload.pseudo_id,
            team_id=payload.team_id,
            seed_data=seed_data,
            score_trend=[],
            cascade_exposure_count=0,
            run_count=0,
            updated_at=now,
            created_at=now,
        )
        db.add(prof)
        profile_seeded = payload.initial_profile is not None
    elif payload.initial_profile and prof.seed_data is None:
        prof.seed_data = _seed_to_dict(payload.initial_profile)
        prof.updated_at = now
        profile_seeded = True

    await db.commit()

    return EnrollResponse(
        pseudo_id=payload.pseudo_id,
        enrolled_at=emp.enrolled_at,
        profile_seeded=profile_seeded,
        message="enrolled" if emp.run_count == 0 else "updated",  # type: ignore[attr-defined]
    )


@router.get("/employees/{pseudo_id}", response_model=EmployeeResponse)
async def get_employee(
    pseudo_id: UUID,
    token: TokenPayload = require_role(UserRole.IT_ADMIN, UserRole.HR_ADMIN),
    db: AsyncSession = Depends(get_db),
) -> EmployeeResponse:
    """Return employee enrollment record.

    Access: IT Admin, HR Admin.
    """
    result = await db.execute(select(Employee).where(Employee.pseudo_id == pseudo_id))
    emp = result.scalar_one_or_none()
    if emp is None:
        raise HTTPException(status_code=404, detail="Employee not found")

    return EmployeeResponse(
        pseudo_id=emp.pseudo_id,
        team_id=emp.team_id,
        role=emp.role,
        seniority=emp.seniority,
        timezone=emp.timezone,
        work_hours_start=emp.work_hours_start,
        work_hours_end=emp.work_hours_end,
        is_active=emp.is_active,
        enrolled_at=emp.enrolled_at,
        updated_at=emp.updated_at,
    )


@router.get("/employees", response_model=list[EmployeeResponse])
async def list_employees(
    team_id: str | None = Query(default=None),
    is_active: bool = Query(default=True),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    token: TokenPayload = require_role(UserRole.IT_ADMIN, UserRole.HR_ADMIN),
    db: AsyncSession = Depends(get_db),
) -> list[EmployeeResponse]:
    """List enrolled employees with optional team filter.

    Access: IT Admin, HR Admin.
    """
    query = select(Employee).where(Employee.is_active == is_active)
    if team_id:
        query = query.where(Employee.team_id == team_id)
    query = query.offset(offset).limit(limit)

    result = await db.execute(query)
    employees = result.scalars().all()

    return [
        EmployeeResponse(
            pseudo_id=e.pseudo_id,
            team_id=e.team_id,
            role=e.role,
            seniority=e.seniority,
            timezone=e.timezone,
            work_hours_start=e.work_hours_start,
            work_hours_end=e.work_hours_end,
            is_active=e.is_active,
            enrolled_at=e.enrolled_at,
            updated_at=e.updated_at,
        )
        for e in employees
    ]
