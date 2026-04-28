"""Mint a short-lived JWT for demo/dev use.

Usage:
    uv run python scripts/mint_demo_token.py [--role ROLE] [--ttl MINUTES]

Roles: it_admin (default), hr_admin, hr_analyst, manager
Prints the Bearer token to stdout so you can paste it into curl / Swagger.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jose import jwt

from config.settings import settings

ROLE_MAP = {
    "it_admin": "it_admin",
    "hr_admin": "hr_admin",
    "hr_analyst": "hr_analyst",
    "manager": "manager",
}


def mint(role: str = "it_admin", ttl_minutes: int = 480) -> str:
    now = int(time.time())
    payload = {
        "sub": str(uuid.uuid4()),
        "role": ROLE_MAP[role],
        "org_id": str(uuid.uuid4()),
        "exp": now + ttl_minutes * 60,
        "iat": now,
    }
    return jwt.encode(payload, settings.jwt_private_key, algorithm=settings.jwt_algorithm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mint a demo JWT")
    parser.add_argument("--role", default="it_admin", choices=list(ROLE_MAP))
    parser.add_argument("--ttl", type=int, default=480, help="Token TTL in minutes")
    args = parser.parse_args()

    token = mint(args.role, args.ttl)
    print(token)
    print(f"\n# curl example ({args.role}, {args.ttl}min TTL):")
    print(f'curl -H "Authorization: Bearer {token}" http://localhost:8000/api/v1/scores')
