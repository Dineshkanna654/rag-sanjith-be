from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from app.services.rag_chain import stream_rag_response


def test_stream_rag_response_yields_tokens():
    mock_docs = [Document(page_content="FastAPI is fast", metadata={})]

    with patch("app.services.rag_chain.similarity_search", return_value=mock_docs), \
         patch("app.services.rag_chain.OllamaLLM") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.stream.return_value = iter(["FastAPI", " is", " great"])
        mock_llm_cls.return_value = mock_llm

        tokens = list(stream_rag_response("What is FastAPI?"))

    assert tokens == ["FastAPI", " is", " great"]


def test_stream_rag_response_no_context():
    with patch("app.services.rag_chain.similarity_search", return_value=[]), \
         patch("app.services.rag_chain.OllamaLLM") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.stream.return_value = iter(["I don't know."])
        mock_llm_cls.return_value = mock_llm

        tokens = list(stream_rag_response("Unknown question"))

    assert len(tokens) >= 1
