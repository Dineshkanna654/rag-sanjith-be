import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import CurrentUser, get_current_user, require_role
from app.db.engine import get_db
from app.db.models import KnowledgeBase

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])

_editor_dep = require_role("admin", "editor")


class KBCreate(BaseModel):
    name: str
    description: str | None = None


class KBOut(BaseModel):
    id: str
    name: str
    description: str | None
    org_id: str
    document_count: int
    status: str

    model_config = {"from_attributes": True}


class DocumentOut(BaseModel):
    id: str
    filename: str
    file_type: str
    chunk_count: int
    created_at: str | None = None

    model_config = {"from_attributes": True}


class KBDetail(KBOut):
    documents: list[DocumentOut] = []


@router.post("", response_model=KBOut, status_code=201)
async def create_knowledge_base(
    body: KBCreate,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(_editor_dep),
):
    kb = KnowledgeBase(
        name=body.name,
        description=body.description,
        org_id=current_user.org_id,
        created_by=current_user.id,
    )
    db.add(kb)
    await db.commit()
    await db.refresh(kb)
    return _kb_to_out(kb)


@router.get("", response_model=list[KBOut])
async def list_knowledge_bases(
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
):
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.org_id == current_user.org_id)
    )
    kbs = result.scalars().all()
    return [_kb_to_out(kb) for kb in kbs]


@router.get("/{kb_id}", response_model=KBDetail)
async def get_knowledge_base(
    kb_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
):
    result = await db.execute(
        select(KnowledgeBase)
        .where(KnowledgeBase.id == uuid.UUID(kb_id), KnowledgeBase.org_id == current_user.org_id)
        .options(selectinload(KnowledgeBase.documents))
    )
    kb = result.scalar_one_or_none()
    if kb is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return _kb_detail(kb)


def _kb_to_out(kb: KnowledgeBase) -> dict:
    return {
        "id": str(kb.id),
        "name": kb.name,
        "description": kb.description,
        "org_id": str(kb.org_id),
        "document_count": kb.document_count,
        "status": kb.status,
    }


def _kb_detail(kb: KnowledgeBase) -> dict:
    out = _kb_to_out(kb)
    out["documents"] = [
        {
            "id": str(doc.id),
            "filename": doc.filename,
            "file_type": doc.file_type,
            "chunk_count": doc.chunk_count,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
        }
        for doc in kb.documents
    ]
    return out
