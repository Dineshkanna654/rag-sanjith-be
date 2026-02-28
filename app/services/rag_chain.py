from typing import Iterator
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from app.config import settings
from app.services.vectorstore import similarity_search

_PROMPT_TEMPLATE = """You are a helpful assistant. Use the following context to answer the question.
If the context doesn't contain relevant information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""


def stream_rag_response(question: str) -> Iterator[str]:
    docs: list[Document] = similarity_search(question, k=5)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = _PROMPT_TEMPLATE.format(context=context, question=question)

    llm = OllamaLLM(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.MODEL_NAME,
    )
    yield from llm.stream(prompt)
