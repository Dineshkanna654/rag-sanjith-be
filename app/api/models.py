import json
import urllib.error
import urllib.request

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import CurrentUser, get_current_user
from app.config import settings

router = APIRouter()


@router.get("/models")
async def list_models(current_user: CurrentUser = Depends(get_current_user)) -> dict:
    """List available Ollama models by querying the local Ollama API."""
    try:
        with urllib.request.urlopen(
            f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5
        ) as resp:
            data = json.loads(resp.read())
        model_names = [
            m["name"] for m in data.get("models", [])
            if "embed" not in m["name"].lower()
        ]
        return {"models": model_names}
    except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail=f"Ollama unreachable: {exc}")
