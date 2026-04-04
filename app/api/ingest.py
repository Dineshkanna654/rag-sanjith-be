import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, require_role
from app.db.engine import get_db
from app.db.models import Document, KnowledgeBase
from app.services.document_loader import load_and_split_file
from app.services.vectorstore import add_documents

logger = logging.getLogger(__name__)

router = APIRouter()

_editor_dep = require_role("admin", "editor")


@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    kb_id: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(_editor_dep),
):
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        chunks = load_and_split_file(tmp_path, file.filename)
        add_documents(chunks)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Ingestion failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    response = {"message": f"Ingested {len(chunks)} chunks from {file.filename}"}

    if kb_id:
        result = await db.execute(
            select(KnowledgeBase).where(KnowledgeBase.id == uuid.UUID(kb_id))
        )
        kb = result.scalar_one_or_none()
        if kb is None:
            raise HTTPException(status_code=404, detail="Knowledge base not found")
    else:
        # Auto-create or reuse a default knowledge base scoped to user's org
        result = await db.execute(
            select(KnowledgeBase).where(
                KnowledgeBase.name == "Default",
                KnowledgeBase.org_id == current_user.org_id,
            )
        )
        kb = result.scalar_one_or_none()
        if kb is None:
            kb = KnowledgeBase(
                name="Default",
                description="Auto-created default knowledge base",
                org_id=current_user.org_id,
                created_by=current_user.id,
            )
            db.add(kb)
            await db.flush()

    doc = Document(
        filename=file.filename,
        file_type=suffix,
        chunk_count=len(chunks),
        kb_id=kb.id,
        uploaded_by=current_user.id,
    )
    db.add(doc)
    kb.document_count = (kb.document_count or 0) + 1
    await db.commit()
    await db.refresh(doc)
    response["document_id"] = str(doc.id)
    response["kb_id"] = str(kb.id)

    return response
