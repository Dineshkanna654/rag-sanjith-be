from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.ingest import router as ingest_router
from app.api.knowledge_base import router as kb_router
from app.api.login import router as login_router
from app.api.query import router as query_router
from app.db.engine import engine
from app.db.models import Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title="RAG Sanjith Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(login_router)
app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(kb_router)


@app.get("/health")
def health():
    return {"status": "ok"}
