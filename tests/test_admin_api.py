import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.api.deps import CurrentUser, get_current_user

FAKE_ADMIN = CurrentUser(id=uuid.uuid4(), username="admin", org_id=uuid.uuid4(), roles=["admin"])
FAKE_VIEWER = CurrentUser(id=uuid.uuid4(), username="viewer", org_id=uuid.uuid4(), roles=["viewer"])


def _mock_session():
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.flush = AsyncMock()
    return session


def _client_with_user(mock_session, user):
    from app.db.engine import get_db
    from app.main import app

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = lambda: user

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


def _mock_user_obj(username="newuser", email=None, org_id=None, is_active=True, roles=None):
    user = MagicMock()
    user.id = uuid.uuid4()
    user.username = username
    user.email = email
    user.org_id = org_id or uuid.uuid4()
    user.is_active = is_active
    mock_roles = []
    for name in (roles or []):
        r = MagicMock()
        r.name = name
        mock_roles.append(r)
    user.roles = mock_roles
    return user


class TestAdminAccessControl:
    def test_viewer_cannot_list_users(self):
        session = _mock_session()
        gen = _client_with_user(session, FAKE_VIEWER)
        client = next(gen)

        resp = client.get("/admin/users")
        assert resp.status_code == 403

        try:
            next(gen)
        except StopIteration:
            pass

    def test_admin_can_list_users(self):
        session = _mock_session()

        user_obj = _mock_user_obj("alice", roles=["editor"])
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [user_obj]
        session.execute = AsyncMock(return_value=mock_result)

        gen = _client_with_user(session, FAKE_ADMIN)
        client = next(gen)

        resp = client.get("/admin/users")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["username"] == "alice"
        assert data[0]["roles"] == ["editor"]

        try:
            next(gen)
        except StopIteration:
            pass


class TestCreateUser:
    def test_create_user_success(self):
        session = _mock_session()

        # First execute: check username uniqueness (returns None)
        # Second execute: re-fetch user with roles
        created_user = _mock_user_obj("newuser", roles=["viewer"])
        check_result = MagicMock()
        check_result.scalar_one_or_none.return_value = None

        roles_result = MagicMock()
        roles_result.scalars.return_value.all.return_value = []

        refetch_result = MagicMock()
        refetch_result.scalar_one.return_value = created_user

        session.execute = AsyncMock(side_effect=[check_result, roles_result, refetch_result])

        gen = _client_with_user(session, FAKE_ADMIN)
        client = next(gen)

        resp = client.post("/admin/users", json={
            "username": "newuser",
            "password": "secret123",
            "role_names": ["viewer"],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["username"] == "newuser"

        try:
            next(gen)
        except StopIteration:
            pass

    def test_create_duplicate_username_returns_409(self):
        session = _mock_session()

        existing_result = MagicMock()
        existing_result.scalar_one_or_none.return_value = _mock_user_obj("taken")
        session.execute = AsyncMock(return_value=existing_result)

        gen = _client_with_user(session, FAKE_ADMIN)
        client = next(gen)

        resp = client.post("/admin/users", json={
            "username": "taken",
            "password": "secret",
        })
        assert resp.status_code == 409
        assert "already exists" in resp.json()["detail"]

        try:
            next(gen)
        except StopIteration:
            pass


class TestListRoles:
    def test_admin_can_list_roles(self):
        session = _mock_session()

        role1 = MagicMock()
        role1.id = uuid.uuid4()
        role1.name = "admin"
        role1.description = "Full access"

        role2 = MagicMock()
        role2.id = uuid.uuid4()
        role2.name = "viewer"
        role2.description = "Read-only"

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [role1, role2]
        session.execute = AsyncMock(return_value=mock_result)

        gen = _client_with_user(session, FAKE_ADMIN)
        client = next(gen)

        resp = client.get("/admin/roles")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["name"] == "admin"

        try:
            next(gen)
        except StopIteration:
            pass

    def test_viewer_cannot_list_roles(self):
        session = _mock_session()
        gen = _client_with_user(session, FAKE_VIEWER)
        client = next(gen)

        resp = client.get("/admin/roles")
        assert resp.status_code == 403

        try:
            next(gen)
        except StopIteration:
            pass


class TestAssignRole:
    def test_assign_role_to_user(self):
        session = _mock_session()
        user_obj = _mock_user_obj("alice", roles=[])
        role_obj = MagicMock()
        role_obj.id = uuid.uuid4()
        role_obj.name = "editor"

        updated_user = _mock_user_obj("alice", roles=["editor"])

        # execute calls: find user, find role, check existing, re-fetch
        user_result = MagicMock()
        user_result.scalar_one_or_none.return_value = user_obj

        role_result = MagicMock()
        role_result.scalar_one_or_none.return_value = role_obj

        existing_result = MagicMock()
        existing_result.scalar_one_or_none.return_value = None

        refetch_result = MagicMock()
        refetch_result.scalar_one.return_value = updated_user

        session.execute = AsyncMock(side_effect=[user_result, role_result, existing_result, refetch_result])

        gen = _client_with_user(session, FAKE_ADMIN)
        client = next(gen)

        resp = client.post(f"/admin/users/{user_obj.id}/roles", json={"role_name": "editor"})
        assert resp.status_code == 200
        assert "editor" in resp.json()["roles"]

        try:
            next(gen)
        except StopIteration:
            pass
