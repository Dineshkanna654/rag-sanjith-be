import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from passlib.hash import bcrypt


def _make_user(username="admin", password="admin123", is_active=True):
    user = MagicMock()
    user.username = username
    user.password_hash = bcrypt.hash(password)
    user.is_active = is_active
    user.id = uuid.uuid4()
    return user


def _mock_session_returning(user):
    """Build a mock AsyncSession whose execute() returns `user`."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = user

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    return mock_session


def _client_with_mocked_db(mock_session):
    """Return a TestClient with DB dependency overridden and lifespan mocked."""
    from app.db.engine import get_db
    from app.main import app

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db

    # Patch the engine so the lifespan doesn't attempt a real DB connection
    mock_conn = AsyncMock()
    mock_conn.run_sync = AsyncMock()
    mock_engine_connect = AsyncMock()
    mock_engine_connect.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine_connect.__aexit__ = AsyncMock(return_value=False)

    with patch("app.main.engine") as mock_engine:
        mock_engine.begin.return_value = mock_engine_connect
        client = TestClient(app)
        yield client

    app.dependency_overrides.clear()


class TestLoginDB:
    def test_login_success(self):
        user = _make_user("admin", "admin123")
        mock_session = _mock_session_returning(user)

        gen = _client_with_mocked_db(mock_session)
        client = next(gen)
        resp = client.post("/login", json={"username": "admin", "password": "admin123"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["username"] == "admin"

        # cleanup
        try:
            next(gen)
        except StopIteration:
            pass

    def test_login_wrong_password(self):
        user = _make_user("admin", "admin123")
        mock_session = _mock_session_returning(user)

        gen = _client_with_mocked_db(mock_session)
        client = next(gen)
        resp = client.post("/login", json={"username": "admin", "password": "wrong"})

        assert resp.status_code == 401

        try:
            next(gen)
        except StopIteration:
            pass

    def test_login_unknown_user(self):
        mock_session = _mock_session_returning(None)

        gen = _client_with_mocked_db(mock_session)
        client = next(gen)
        resp = client.post("/login", json={"username": "nobody", "password": "pass"})

        assert resp.status_code == 401

        try:
            next(gen)
        except StopIteration:
            pass

    def test_login_inactive_user(self):
        user = _make_user("admin", "admin123", is_active=False)
        mock_session = _mock_session_returning(user)

        gen = _client_with_mocked_db(mock_session)
        client = next(gen)
        resp = client.post("/login", json={"username": "admin", "password": "admin123"})

        assert resp.status_code == 403
        assert "disabled" in resp.json()["detail"]

        try:
            next(gen)
        except StopIteration:
            pass
