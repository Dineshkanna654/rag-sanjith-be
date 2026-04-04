import json
import uuid
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.api.deps import CurrentUser, get_current_user
from app.main import app

FAKE_USER = CurrentUser(id=uuid.uuid4(), username="testuser", org_id=uuid.uuid4())


def _client_with_auth():
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


def test_query_streams_json_events():
    mock_sources = [{"id": 1, "score": 0.9, "content": "test content", "metadata": {"source": "test.pdf", "page": 1}}]

    def mock_stream(question):
        return mock_sources, iter(["Hello", " world"])

    gen = _client_with_auth()
    client = next(gen)

    with patch("app.api.query.stream_rag_response", side_effect=mock_stream):
        response = client.get("/query?q=test+question")

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = [line.removeprefix("data: ") for line in response.text.strip().split("\n") if line.startswith("data: ")]

    # First event is sources
    sources_event = json.loads(events[0])
    assert sources_event["type"] == "sources"
    assert len(sources_event["data"]) == 1
    assert sources_event["data"][0]["score"] == 0.9

    # Token events
    token1 = json.loads(events[1])
    assert token1 == {"type": "token", "data": "Hello"}
    token2 = json.loads(events[2])
    assert token2 == {"type": "token", "data": " world"}

    # DONE sentinel
    assert events[-1] == "[DONE]"

    try:
        next(gen)
    except StopIteration:
        pass


def test_query_missing_param_returns_422():
    gen = _client_with_auth()
    client = next(gen)

    response = client.get("/query")
    assert response.status_code == 422

    try:
        next(gen)
    except StopIteration:
        pass
