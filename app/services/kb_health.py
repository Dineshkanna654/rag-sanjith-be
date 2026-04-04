import logging
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, KnowledgeBase
from app.services.vectorstore import delete_chunks_by_filter, get_all_chunk_metadata, get_collection_count

logger = logging.getLogger(__name__)


async def run_health_check(db: AsyncSession) -> dict:
    """Cross-reference PostgreSQL and ChromaDB to detect inconsistencies."""

    now = datetime.now(timezone.utc).isoformat()

    # --- Gather PostgreSQL data ---
    docs_result = await db.execute(
        select(Document).where(Document.status == "active")
    )
    pg_docs = docs_result.scalars().all()
    pg_doc_ids = {str(d.id) for d in pg_docs}
    total_chunks_pg = sum(d.chunk_count for d in pg_docs)

    # --- Gather ChromaDB data ---
    try:
        chroma_count = get_collection_count()
        all_metadata = get_all_chunk_metadata()
    except Exception as e:
        logger.exception("Failed to read ChromaDB")
        return {
            "status": "unhealthy",
            "checked_at": now,
            "summary": {"total_documents": len(pg_docs), "total_chunks_pg": total_chunks_pg, "total_chunks_chroma": 0},
            "checks": {
                "chroma_sync": {"status": "error", "message": f"ChromaDB unreachable: {e}"},
            },
            "healable_count": 0,
        }

    # --- Check 1: ChromaDB sync ---
    difference = abs(total_chunks_pg - chroma_count)
    if difference == 0:
        chroma_sync = {"status": "ok", "pg_chunks": total_chunks_pg, "chroma_chunks": chroma_count, "difference": 0}
    elif difference <= 5:
        chroma_sync = {"status": "warning", "pg_chunks": total_chunks_pg, "chroma_chunks": chroma_count, "difference": difference}
    else:
        chroma_sync = {"status": "error", "pg_chunks": total_chunks_pg, "chroma_chunks": chroma_count, "difference": difference}

    # --- Categorize chunks ---
    chunks_by_doc_id: dict[str, int] = {}
    legacy_chunk_count = 0
    orphaned_doc_ids: set[str] = set()

    for meta in all_metadata:
        doc_id = meta.get("document_id") if meta else None
        if not doc_id:
            legacy_chunk_count += 1
            continue
        chunks_by_doc_id[doc_id] = chunks_by_doc_id.get(doc_id, 0) + 1
        if doc_id not in pg_doc_ids:
            orphaned_doc_ids.add(doc_id)

    # --- Check 2: Ghost documents (in PG but no chunks in ChromaDB) ---
    ghost_docs = []
    for doc in pg_docs:
        doc_id_str = str(doc.id)
        if doc_id_str not in chunks_by_doc_id:
            ghost_docs.append({"id": doc_id_str, "filename": doc.filename})

    ghost_status = "ok" if len(ghost_docs) == 0 else ("warning" if len(ghost_docs) <= 2 else "error")
    ghost_check = {"status": ghost_status, "count": len(ghost_docs), "documents": ghost_docs}

    # --- Check 3: Orphaned chunks (in ChromaDB but not in PG) ---
    orphaned_chunk_count = sum(chunks_by_doc_id.get(did, 0) for did in orphaned_doc_ids)
    orphan_status = "ok" if orphaned_chunk_count == 0 else ("warning" if orphaned_chunk_count <= 10 else "error")
    orphan_check = {"status": orphan_status, "count": orphaned_chunk_count}

    # --- Check 4: Legacy chunks (no document_id metadata) ---
    legacy_status = "ok" if legacy_chunk_count == 0 else "warning"
    legacy_check = {"status": legacy_status, "count": legacy_chunk_count}

    # --- Check 5: Count drift ---
    kbs_result = await db.execute(select(KnowledgeBase))
    kbs = kbs_result.scalars().all()
    drifted_kbs = []
    for kb in kbs:
        actual_result = await db.execute(
            select(func.count()).select_from(Document).where(
                Document.kb_id == kb.id,
                Document.status == "active",
            )
        )
        actual = actual_result.scalar() or 0
        if kb.document_count != actual:
            drifted_kbs.append({
                "id": str(kb.id),
                "name": kb.name,
                "recorded": kb.document_count,
                "actual": actual,
            })

    drift_status = "ok" if len(drifted_kbs) == 0 else "warning"
    drift_check = {"status": drift_status, "drifted_kbs": drifted_kbs}

    # --- Aggregate ---
    all_checks = {
        "chroma_sync": chroma_sync,
        "ghost_documents": ghost_check,
        "orphaned_chunks": orphan_check,
        "legacy_chunks": legacy_check,
        "count_drift": drift_check,
    }

    statuses = [c["status"] for c in all_checks.values()]
    if "error" in statuses:
        overall = "unhealthy"
    elif "warning" in statuses:
        overall = "degraded"
    else:
        overall = "healthy"

    healable = len(drifted_kbs) + (1 if orphaned_chunk_count > 0 else 0) + len(ghost_docs)

    return {
        "status": overall,
        "checked_at": now,
        "summary": {
            "total_documents": len(pg_docs),
            "total_chunks_pg": total_chunks_pg,
            "total_chunks_chroma": chroma_count,
        },
        "checks": all_checks,
        "healable_count": healable,
    }


async def run_healing(db: AsyncSession) -> dict:
    """Auto-repair detected inconsistencies between PostgreSQL and ChromaDB."""

    now = datetime.now(timezone.utc).isoformat()
    actions: list[dict] = []

    # --- Gather data ---
    docs_result = await db.execute(
        select(Document).where(Document.status == "active")
    )
    pg_docs = docs_result.scalars().all()
    pg_doc_ids = {str(d.id) for d in pg_docs}

    all_metadata = get_all_chunk_metadata()

    chunks_by_doc_id: dict[str, int] = {}
    legacy_chunk_count = 0
    orphaned_doc_ids: set[str] = set()

    for meta in all_metadata:
        doc_id = meta.get("document_id") if meta else None
        if not doc_id:
            legacy_chunk_count += 1
            continue
        chunks_by_doc_id[doc_id] = chunks_by_doc_id.get(doc_id, 0) + 1
        if doc_id not in pg_doc_ids:
            orphaned_doc_ids.add(doc_id)

    # --- Action 1: Reconcile KB document counts ---
    kbs_result = await db.execute(select(KnowledgeBase))
    kbs = kbs_result.scalars().all()
    for kb in kbs:
        actual_result = await db.execute(
            select(func.count()).select_from(Document).where(
                Document.kb_id == kb.id,
                Document.status == "active",
            )
        )
        actual = actual_result.scalar() or 0
        if kb.document_count != actual:
            old_count = kb.document_count
            kb.document_count = actual
            actions.append({
                "type": "reconcile_count",
                "details": f"Updated KB '{kb.name}': {old_count} → {actual}",
            })

    # --- Action 2: Purge orphaned chunks (skip legacy — don't delete useful data) ---
    orphaned_purged = 0
    for doc_id in orphaned_doc_ids:
        try:
            delete_chunks_by_filter({"document_id": doc_id})
            orphaned_purged += chunks_by_doc_id.get(doc_id, 0)
        except Exception:
            logger.exception("Failed to purge orphaned chunks for doc_id=%s", doc_id)

    if orphaned_purged > 0:
        actions.append({"type": "purge_orphaned", "chunks_removed": orphaned_purged})

    # --- Action 3: Mark ghost documents as stale ---
    stale_count = 0
    for doc in pg_docs:
        if str(doc.id) not in chunks_by_doc_id:
            doc.status = "stale"
            stale_count += 1

    if stale_count > 0:
        actions.append({"type": "mark_stale", "documents_marked": stale_count})

    await db.commit()

    issues_fixed = len(actions)
    remaining = []
    if legacy_chunk_count > 0:
        remaining.append({
            "type": "legacy_chunks",
            "count": legacy_chunk_count,
            "message": "Re-upload documents to track chunks with document_id metadata",
        })

    return {
        "healed_at": now,
        "actions": actions,
        "issues_fixed": issues_fixed,
        "issues_remaining": len(remaining),
        "remaining": remaining,
    }
