from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, require_role
from app.db.engine import get_db
from app.services.kb_health import run_health_check, run_healing

router = APIRouter()

_admin_dep = require_role("admin")


@router.get("/knowledge-bases/health")
async def kb_health(
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(_admin_dep),
):
    return await run_health_check(db)


@router.post("/knowledge-bases/heal")
async def kb_heal(
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(_admin_dep),
):
    return await run_healing(db)
