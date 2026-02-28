from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app

client = TestClient(app)
FIXTURES = Path(__file__).parent / "fixtures"


def test_ingest_txt_file():
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


def test_ingest_unsupported_file_returns_400():
    with patch("app.api.ingest.load_and_split_file", side_effect=ValueError("Unsupported file type: .csv")):
        response = client.post(
            "/ingest",
            files={"file": ("data.csv", b"col1,col2\n1,2", "text/csv")}
        )
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]
