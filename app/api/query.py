import json

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from app.api.deps import CurrentUser, get_current_user
from app.services.rag_chain import stream_rag_response

router = APIRouter()


def _event_generator(question: str):
    try:
        sources, tokens = stream_rag_response(question)
        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
        for token in tokens:
            yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"


@router.get("/query")
def query_documents(
    q: str = Query(..., min_length=1),
    current_user: CurrentUser = Depends(get_current_user),
):
    return StreamingResponse(
        _event_generator(q),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
