from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from app.api.admin import router as admin_router
from app.api.health import router as health_router
from app.api.ingest import router as ingest_router
from app.api.knowledge_base import router as kb_router
from app.api.login import router as login_router
from app.api.query import router as query_router
from app.db.engine import engine
from app.db.models import Base, Role

DEFAULT_ROLES = [
    ("admin", "Full access: manage users, roles, and all resources"),
    ("editor", "Can upload documents, create knowledge bases, and query"),
    ("viewer", "Read-only: can query and view knowledge bases"),
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed default roles
    from app.db.engine import async_session

    async with async_session() as session:
        for name, description in DEFAULT_ROLES:
            result = await session.execute(select(Role).where(Role.name == name))
            if result.scalar_one_or_none() is None:
                session.add(Role(name=name, description=description))
        await session.commit()

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
app.include_router(health_router)
app.include_router(kb_router)
app.include_router(admin_router)


@app.get("/health")
def health():
    return {"status": "ok"}
