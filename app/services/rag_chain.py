from typing import Iterator
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from app.config import settings
from app.services.vectorstore import similarity_search_with_scores
from app.services.trust_score import compute_trust_score

_PROMPT_TEMPLATE = """You are a helpful assistant. Use the following context to answer the question.
If the context doesn't contain relevant information, say so clearly.
When referencing information from the context, cite the source using [Source N] notation.

Context:
{context}

Question: {question}

Answer:"""


def stream_rag_response(question: str) -> tuple[list[dict], dict, Iterator[str]]:
    scored_docs: list[tuple[Document, float]] = similarity_search_with_scores(question, k=5)

    sources = []
    context_parts = []
    for i, (doc, score) in enumerate(scored_docs, start=1):
        sources.append({
            "id": i,
            "score": round(float(score), 4),
            "content": doc.page_content[:500],
            "metadata": {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
            },
        })
        context_parts.append(f"[Source {i}]\n{doc.page_content}")

    context = "\n\n".join(context_parts)
    prompt = _PROMPT_TEMPLATE.format(context=context, question=question)

    llm = OllamaLLM(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.MODEL_NAME,
    )

    raw_scores = [float(score) for _, score in scored_docs]
    trust = compute_trust_score(raw_scores)

    def token_iterator() -> Iterator[str]:
        yield from llm.stream(prompt)

    return sources, trust.to_dict(), token_iterator()
