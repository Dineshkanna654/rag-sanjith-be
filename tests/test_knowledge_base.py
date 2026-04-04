import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.api.deps import CurrentUser, get_current_user

FAKE_ORG_ID = uuid.uuid4()
FAKE_USER = CurrentUser(id=uuid.uuid4(), username="testuser", org_id=FAKE_ORG_ID)


def _mock_kb(name="Test KB", org_id=None, doc_count=0):
    kb = MagicMock()
    kb.id = uuid.uuid4()
    kb.name = name
    kb.description = "A test knowledge base"
    kb.org_id = org_id or FAKE_ORG_ID
    kb.document_count = doc_count
    kb.status = "active"
    kb.documents = []
    return kb


def _mock_session():
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session


def _client_with_mocked_db(mock_session):
    from app.db.engine import get_db
    from app.main import app

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = lambda: FAKE_USER

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


class TestCreateKB:
    def test_create_success(self):
        session = _mock_session()

        # After refresh, the KB object needs its defaults populated
        async def fake_refresh(obj, **kw):
            obj.id = uuid.uuid4()
            obj.document_count = 0
            obj.status = "active"
            obj.org_id = FAKE_ORG_ID

        session.refresh = AsyncMock(side_effect=fake_refresh)

        gen = _client_with_mocked_db(session)
        client = next(gen)

        resp = client.post(
            "/knowledge-bases",
            json={"name": "My KB"},
        )

        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "My KB"
        assert data["document_count"] == 0
        assert session.add.called
        assert session.commit.called

        try:
            next(gen)
        except StopIteration:
            pass


class TestListKBs:
    def test_list_returns_kbs(self):
        session = _mock_session()
        kb = _mock_kb()

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [kb]
        session.execute = AsyncMock(return_value=mock_result)

        gen = _client_with_mocked_db(session)
        client = next(gen)

        resp = client.get("/knowledge-bases")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "Test KB"

        try:
            next(gen)
        except StopIteration:
            pass


class TestGetKB:
    def test_get_existing_kb(self):
        session = _mock_session()
        kb = _mock_kb()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = kb
        session.execute = AsyncMock(return_value=mock_result)

        gen = _client_with_mocked_db(session)
        client = next(gen)

        resp = client.get(f"/knowledge-bases/{kb.id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Test KB"
        assert resp.json()["documents"] == []

        try:
            next(gen)
        except StopIteration:
            pass

    def test_get_nonexistent_kb_returns_404(self):
        session = _mock_session()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=mock_result)

        gen = _client_with_mocked_db(session)
        client = next(gen)

        resp = client.get(f"/knowledge-bases/{uuid.uuid4()}")
        assert resp.status_code == 404

        try:
            next(gen)
        except StopIteration:
            pass
