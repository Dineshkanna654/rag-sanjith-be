from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from app.config import settings


def _get_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.EMBEDDING_MODEL,
    )
    return Chroma(
        collection_name="rag_documents",
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )


def add_documents(docs: list[Document]) -> None:
    vs = _get_vectorstore()
    vs.add_documents(docs)


def similarity_search(query: str, k: int = 5) -> list[Document]:
    vs = _get_vectorstore()
    return vs.similarity_search(query, k=k)


def similarity_search_with_scores(query: str, k: int = 5) -> list[tuple[Document, float]]:
    vs = _get_vectorstore()
    return vs.similarity_search_with_relevance_scores(query, k=k)


def get_collection_count() -> int:
    vs = _get_vectorstore()
    return vs._collection.count()


def get_all_chunk_metadata() -> list[dict]:
    vs = _get_vectorstore()
    result = vs._collection.get(include=["metadatas"])
    return result["metadatas"] or []


def delete_chunks_by_filter(where: dict) -> None:
    vs = _get_vectorstore()
    vs._collection.delete(where=where)
