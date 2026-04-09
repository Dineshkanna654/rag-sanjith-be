import json

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from app.api.deps import CurrentUser, get_current_user
from app.services.rag_chain import stream_rag_response, stream_multi_rag_response

router = APIRouter()


def _event_generator(question: str):
    try:
        sources, trust_score, tokens = stream_rag_response(question)
        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
        yield f"data: {json.dumps({'type': 'trust_score', 'data': trust_score})}\n\n"
        for token in tokens:
            yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"


def _multi_event_generator(question: str, models: list[str]):
    try:
        sources, trust_score, events = stream_multi_rag_response(question, models)
        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
        yield f"data: {json.dumps({'type': 'trust_score', 'data': trust_score})}\n\n"
        for event in events:
            yield f"data: {json.dumps(event)}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"


@router.get("/query/multi")
def query_documents_multi(
    q: str = Query(..., min_length=1),
    models: str = Query(..., description="Comma-separated model names (2-3)"),
    current_user: CurrentUser = Depends(get_current_user),
):
    from fastapi import HTTPException
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not (2 <= len(model_list) <= 3):
        raise HTTPException(status_code=400, detail="Provide 2 or 3 model names")
    return StreamingResponse(
        _multi_event_generator(q, model_list),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
