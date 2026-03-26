"""Role-Based Access Control middleware.

Enforced at the API layer — NOT only at the UI layer.
Manager role MUST NEVER receive individual burnout scores or pseudo_ids.
"""

from __future__ import annotations

import logging

from fastapi import Depends, HTTPException, status

from api.middleware.auth import TokenPayload, verify_jwt
from api.schemas.common import UserRole

logger = logging.getLogger(__name__)


def require_role(*roles: UserRole) -> object:
    """FastAPI dependency factory — raises HTTP 403 if token role not in allowed set.

    Usage:
        @router.get("/scores")
        async def get_scores(token = Depends(require_role(UserRole.HR_ADMIN, UserRole.HR_ANALYST))):
            ...
    """

    async def _check_role(token: TokenPayload = Depends(verify_jwt)) -> TokenPayload:
        if token.role not in roles:
            logger.warning(
                "Access denied: role=%s tried to access endpoint requiring %s",
                token.role,
                roles,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions.",
            )
        return token

    return Depends(_check_role)
