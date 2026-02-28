import tempfile
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.document_loader import load_and_split_file
from app.services.vectorstore import add_documents

router = APIRouter()


@router.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
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
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return {"message": f"Ingested {len(chunks)} chunks from {file.filename}"}
