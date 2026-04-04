import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.api.deps import CurrentUser, get_current_user
from app.main import app

FIXTURES = Path(__file__).parent / "fixtures"

FAKE_USER = CurrentUser(id=uuid.uuid4(), username="testuser", org_id=uuid.uuid4())


def _mock_session_with_kb(kb=None):
    """Build a mock AsyncSession for ingest tests."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = kb

    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session


def _make_kb(doc_count=0):
    kb = MagicMock()
    kb.id = uuid.uuid4()
    kb.document_count = doc_count
    return kb


def _client_with_mocked_db(mock_session):
    from app.db.engine import get_db

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


def test_ingest_txt_file():
    session = _mock_session_with_kb()
    gen = _client_with_mocked_db(session)
    client = next(gen)

    with patch("app.api.ingest.load_and_split_file") as mock_load, \
         patch("app.api.ingest.add_documents") as mock_add:
        from langchain_core.documents import Document
        mock_load.return_value = [Document(page_content="test content")]
        mock_add.return_value = None

        with open(FIXTURES / "sample.txt", "rb") as f:
            response = client.post("/ingest", files={"file": ("sample.txt", f, "text/plain")})

    assert response.status_code == 200
    assert "Ingested" in response.json()["message"]
    assert "sample.txt" in response.json()["message"]

    try:
        next(gen)
    except StopIteration:
        pass


def test_ingest_unsupported_file_returns_400():
    session = _mock_session_with_kb()
    gen = _client_with_mocked_db(session)
    client = next(gen)

    with patch("app.api.ingest.load_and_split_file", side_effect=ValueError("Unsupported file type: .csv")):
        response = client.post(
            "/ingest",
            files={"file": ("data.csv", b"col1,col2\n1,2", "text/csv")}
        )
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]

    try:
        next(gen)
    except StopIteration:
        pass


def test_ingest_with_kb_id():
    kb = _make_kb(doc_count=0)
    session = _mock_session_with_kb(kb)
    gen = _client_with_mocked_db(session)
    client = next(gen)

    with patch("app.api.ingest.load_and_split_file") as mock_load, \
         patch("app.api.ingest.add_documents") as mock_add:
        from langchain_core.documents import Document
        mock_load.return_value = [Document(page_content="chunk1"), Document(page_content="chunk2")]
        mock_add.return_value = None

        with open(FIXTURES / "sample.txt", "rb") as f:
            response = client.post(
                "/ingest",
                files={"file": ("sample.txt", f, "text/plain")},
                data={"kb_id": str(kb.id)},
            )

    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert data["kb_id"] == str(kb.id)
    assert session.add.called
    assert session.commit.called
    assert kb.document_count == 1

    try:
        next(gen)
    except StopIteration:
        pass


def test_ingest_with_invalid_kb_id_returns_404():
    session = _mock_session_with_kb(None)  # KB not found
    gen = _client_with_mocked_db(session)
    client = next(gen)

    with patch("app.api.ingest.load_and_split_file") as mock_load, \
         patch("app.api.ingest.add_documents") as mock_add:
        from langchain_core.documents import Document
        mock_load.return_value = [Document(page_content="test")]
        mock_add.return_value = None

        with open(FIXTURES / "sample.txt", "rb") as f:
            response = client.post(
                "/ingest",
                files={"file": ("sample.txt", f, "text/plain")},
                data={"kb_id": str(uuid.uuid4())},
            )

    assert response.status_code == 404
    assert "Knowledge base not found" in response.json()["detail"]

    try:
        next(gen)
    except StopIteration:
        pass
