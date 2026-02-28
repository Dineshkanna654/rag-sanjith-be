from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.services.rag_chain import stream_rag_response

router = APIRouter()


def _event_generator(question: str):
    try:
        for token in stream_rag_response(question):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: [ERROR] {e}\n\n"


@router.get("/query")
def query_documents(q: str = Query(..., min_length=1)):
    return StreamingResponse(
        _event_generator(q),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
