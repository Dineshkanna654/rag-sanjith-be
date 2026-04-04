import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Callable

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.engine import get_db
from app.db.models import User

_bearer = HTTPBearer()


def create_access_token(user_id: uuid.UUID, username: str, org_id: uuid.UUID | None, roles: list[str]) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "username": username,
        "org_id": str(org_id) if org_id else None,
        "roles": roles,
        "iat": now,
        "exp": now + timedelta(hours=settings.JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


@dataclass
class CurrentUser:
    id: uuid.UUID
    username: str
    org_id: uuid.UUID | None
    roles: list[str] = field(default_factory=list)


async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
) -> CurrentUser:
    try:
        payload = jwt.decode(creds.credentials, settings.JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")

    role_names = [r.name for r in user.roles]

    return CurrentUser(
        id=user.id,
        username=user.username,
        org_id=user.org_id,
        roles=role_names,
    )


def require_role(*allowed: str) -> Callable:
    """Dependency factory: raises 403 if user lacks all of the allowed roles."""
    async def _check(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if not any(r in allowed for r in current_user.roles):
            raise HTTPException(
                status_code=403,
                detail=f"Requires one of: {', '.join(allowed)}",
            )
        return current_user
    return _check
