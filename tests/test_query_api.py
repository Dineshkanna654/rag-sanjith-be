from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_query_streams_response():
    with patch("app.api.query.stream_rag_response", return_value=iter(["Hello", " world"])):
        response = client.get("/query?q=test+question")

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    body = response.text
    assert "Hello" in body
    assert "world" in body
    assert "[DONE]" in body


def test_query_missing_param_returns_422():
    response = client.get("/query")
    assert response.status_code == 422
