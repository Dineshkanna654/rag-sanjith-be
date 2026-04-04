import json
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_query_streams_json_events():
    mock_sources = [{"id": 1, "score": 0.9, "content": "test content", "metadata": {"source": "test.pdf", "page": 1}}]

    def mock_stream(question):
        return mock_sources, iter(["Hello", " world"])

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


def test_query_missing_param_returns_422():
    response = client.get("/query")
    assert response.status_code == 422
