from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from app.services.rag_chain import stream_rag_response


def test_stream_rag_response_returns_sources_and_tokens():
    mock_scored_docs = [
        (Document(page_content="FastAPI is fast", metadata={"source": "doc.pdf", "page": 1}), 0.92),
        (Document(page_content="Flask is also good", metadata={"source": "doc2.pdf"}), 0.75),
    ]

    with patch("app.services.rag_chain.similarity_search_with_scores", return_value=mock_scored_docs), \
         patch("app.services.rag_chain.OllamaLLM") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.stream.return_value = iter(["FastAPI", " is", " great"])
        mock_llm_cls.return_value = mock_llm

        sources, tokens = stream_rag_response("What is FastAPI?")

    assert len(sources) == 2
    assert sources[0]["id"] == 1
    assert sources[0]["score"] == 0.92
    assert sources[0]["content"] == "FastAPI is fast"
    assert sources[0]["metadata"]["source"] == "doc.pdf"
    assert sources[0]["metadata"]["page"] == 1
    assert sources[1]["id"] == 2
    assert sources[1]["metadata"]["page"] is None

    assert list(tokens) == ["FastAPI", " is", " great"]


def test_stream_rag_response_no_context():
    with patch("app.services.rag_chain.similarity_search_with_scores", return_value=[]), \
         patch("app.services.rag_chain.OllamaLLM") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.stream.return_value = iter(["I don't know."])
        mock_llm_cls.return_value = mock_llm

        sources, tokens = stream_rag_response("Unknown question")

    assert sources == []
    assert list(tokens) == ["I don't know."]
