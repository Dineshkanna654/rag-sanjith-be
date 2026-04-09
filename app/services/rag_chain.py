import queue
import re
import threading
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


# ─── Multi-LLM Arbitration ───────────────────────────────────────────────────

_STOPWORDS = {
    "what", "when", "where", "which", "that", "this", "with", "from",
    "have", "been", "will", "does", "into", "more", "than", "about",
    "some", "they", "their", "there", "would", "could", "should",
}


def compute_arbitration_scores(
    responses: dict[str, str],
    question: str,
    weights: dict[str, float] | None = None,
) -> dict:
    """Score each model response and pick a winner."""
    w = {"citation": 0.35, "length": 0.35, "keyword": 0.30}
    if weights:
        w.update(weights)

    # Extract significant question words for keyword coverage
    q_words = {
        word.lower() for word in re.findall(r'\b\w+\b', question)
        if len(word) > 3 and word.lower() not in _STOPWORDS
    }

    scores: dict[str, dict] = {}
    for model, response in responses.items():
        # Citation score: count [Source N] references
        citations = len(re.findall(r'\[Source\s+\d+\]', response))
        citation_score = min(citations / 5.0, 1.0)

        # Length score: optimal 300-600 words
        words = len(response.split())
        if 300 <= words <= 600:
            length_score = 1.0
        elif words < 300:
            length_score = max(words / 300.0, 0.0)
        else:
            length_score = max(1.0 - (words - 600) / 600.0, 0.0)

        # Keyword coverage
        if q_words:
            response_lower = response.lower()
            covered = sum(1 for word in q_words if word in response_lower)
            keyword_score = covered / len(q_words)
        else:
            keyword_score = 1.0

        total = (
            w["citation"] * citation_score
            + w["length"] * length_score
            + w["keyword"] * keyword_score
        )
        scores[model] = {
            "citation_score": round(citation_score, 3),
            "length_score": round(length_score, 3),
            "keyword_score": round(keyword_score, 3),
            "total": round(total, 3),
        }

    # Pick winner (highest total; alphabetical tie-break)
    winner = max(scores, key=lambda m: (scores[m]["total"], m))
    winner_scores = scores[winner]

    # Build reason string
    sub_scores = {
        "citation usage": winner_scores["citation_score"],
        "response length": winner_scores["length_score"],
        "keyword coverage": winner_scores["keyword_score"],
    }
    best_sub = max(sub_scores, key=sub_scores.__getitem__)
    reason = (
        f"{winner} scored highest overall ({winner_scores['total']:.0%}) "
        f"with strong {best_sub} ({sub_scores[best_sub]:.0%})."
    )

    return {"scores": scores, "winner": winner, "reason": reason}


def stream_multi_rag_response(
    question: str,
    models: list[str],
) -> tuple[list[dict], dict, Iterator[dict]]:
    """Retrieve context once, then run each model in parallel threads."""
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

    raw_scores = [float(score) for _, score in scored_docs]
    trust = compute_trust_score(raw_scores)

    # Shared state for worker threads
    token_queue: queue.Queue = queue.Queue()
    full_responses: dict[str, list[str]] = {m: [] for m in models}
    finished = [0]
    lock = threading.Lock()

    def _model_worker(model_name: str) -> None:
        token_queue.put({"type": "model_start", "model": model_name})
        try:
            llm = OllamaLLM(base_url=settings.OLLAMA_BASE_URL, model=model_name)
            for token in llm.stream(prompt):
                full_responses[model_name].append(token)
                token_queue.put({"type": "model_token", "model": model_name, "data": token})
            token_queue.put({"type": "model_done", "model": model_name})
        except Exception as exc:
            token_queue.put({"type": "model_error", "model": model_name, "data": str(exc)})
        finally:
            with lock:
                finished[0] += 1
                if finished[0] == len(models):
                    token_queue.put(None)  # sentinel

    for model in models:
        t = threading.Thread(target=_model_worker, args=(model,), daemon=True)
        t.start()

    def _event_iterator() -> Iterator[dict]:
        while True:
            item = token_queue.get()
            if item is None:
                break
            yield item
        # All models done — compute arbitration
        responses_text = {"".join(full_responses[m]): m for m in models}
        arb = compute_arbitration_scores(
            {m: "".join(full_responses[m]) for m in models},
            question,
        )
        yield {"type": "arbitration", "data": arb}
        yield {"type": "done"}

    return sources, trust.to_dict(), _event_iterator()
