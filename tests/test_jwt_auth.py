import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest

from app.api.deps import CurrentUser, create_access_token, get_current_user
from app.config import settings


class TestCreateAccessToken:
    def test_creates_valid_jwt(self):
        user_id = uuid.uuid4()
        org_id = uuid.uuid4()
        token = create_access_token(user_id, "alice", org_id, ["admin", "editor"])

        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        assert payload["sub"] == str(user_id)
        assert payload["username"] == "alice"
        assert payload["org_id"] == str(org_id)
        assert payload["roles"] == ["admin", "editor"]
        assert "exp" in payload
        assert "iat" in payload

    def test_handles_none_org_id(self):
        token = create_access_token(uuid.uuid4(), "bob", None, [])
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        assert payload["org_id"] is None
        assert payload["roles"] == []

    def test_token_expires_in_configured_hours(self):
        token = create_access_token(uuid.uuid4(), "carol", uuid.uuid4(), ["viewer"])
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])

        exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        iat = datetime.fromtimestamp(payload["iat"], tz=timezone.utc)
        delta = exp - iat
        assert abs(delta.total_seconds() - settings.JWT_EXPIRY_HOURS * 3600) < 2


class TestGetCurrentUser:
    def _mock_user(self, user_id, username="alice", org_id=None, is_active=True, roles=None):
        mock_user = MagicMock()
        mock_user.id = user_id
        mock_user.username = username
        mock_user.org_id = org_id or uuid.uuid4()
        mock_user.is_active = is_active
        mock_roles = []
        for name in (roles or []):
            r = MagicMock()
            r.name = name
            mock_roles.append(r)
        mock_user.roles = mock_roles
        return mock_user

    @pytest.mark.asyncio
    async def test_valid_token_returns_user_with_roles(self):
        user_id = uuid.uuid4()
        org_id = uuid.uuid4()
        token = create_access_token(user_id, "alice", org_id, ["admin"])

        creds = MagicMock()
        creds.credentials = token

        mock_user = self._mock_user(user_id, "alice", org_id, roles=["admin"])

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await get_current_user(creds, mock_db)
        assert isinstance(result, CurrentUser)
        assert result.id == user_id
        assert result.username == "alice"
        assert result.org_id == org_id
        assert result.roles == ["admin"]

    @pytest.mark.asyncio
    async def test_expired_token_raises_401(self):
        from fastapi import HTTPException

        payload = {
            "sub": str(uuid.uuid4()),
            "username": "alice",
            "org_id": str(uuid.uuid4()),
            "roles": [],
            "iat": datetime.now(timezone.utc) - timedelta(hours=48),
            "exp": datetime.now(timezone.utc) - timedelta(hours=24),
        }
        token = jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")

        creds = MagicMock()
        creds.credentials = token
        mock_db = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(creds, mock_db)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_invalid_token_raises_401(self):
        from fastapi import HTTPException

        creds = MagicMock()
        creds.credentials = "not-a-valid-jwt"
        mock_db = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(creds, mock_db)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_inactive_user_raises_403(self):
        from fastapi import HTTPException

        user_id = uuid.uuid4()
        token = create_access_token(user_id, "alice", uuid.uuid4(), [])

        creds = MagicMock()
        creds.credentials = token

        mock_user = self._mock_user(user_id, is_active=False)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(creds, mock_db)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_user_not_found_raises_401(self):
        from fastapi import HTTPException

        token = create_access_token(uuid.uuid4(), "ghost", uuid.uuid4(), [])

        creds = MagicMock()
        creds.credentials = token

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(creds, mock_db)
        assert exc_info.value.status_code == 401
