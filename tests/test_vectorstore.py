import pytest
from langchain_core.documents import Document
from app.services.vectorstore import add_documents, similarity_search


@pytest.mark.integration
def test_add_and_search_documents(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    docs = [Document(page_content="FastAPI is a modern web framework", metadata={"source": "test"})]
    add_documents(docs)
    results = similarity_search("web framework")
    assert len(results) >= 1
    assert "FastAPI" in results[0].page_content
