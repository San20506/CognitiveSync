"""JWT RS256 authentication middleware.

Validates Bearer tokens on every protected endpoint.
Token claims: sub (pseudo_id or service_id), role, org_id, exp.
"""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from api.schemas.common import UserRole
from config.settings import settings

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class TokenPayload(BaseModel):
    sub: UUID
    role: UserRole
    org_id: UUID
    exp: int


async def verify_jwt(token: str = Depends(oauth2_scheme)) -> TokenPayload:
    """FastAPI dependency — validates JWT and returns parsed payload.

    Raises HTTP 401 if token is missing, expired, or invalid.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token,
            settings.jwt_public_key,
            algorithms=[settings.jwt_algorithm],
        )
        return TokenPayload(**payload)
    except JWTError:
        logger.warning("JWT validation failed")
        raise credentials_exception
