import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.deps import CurrentUser, require_role


class TestRequireRole:
    @pytest.mark.asyncio
    async def test_user_with_allowed_role_passes(self):
        check = require_role("admin", "editor")
        user = CurrentUser(id=uuid.uuid4(), username="alice", org_id=uuid.uuid4(), roles=["editor"])
        result = await check(current_user=user)
        assert result is user

    @pytest.mark.asyncio
    async def test_admin_passes_admin_check(self):
        check = require_role("admin")
        user = CurrentUser(id=uuid.uuid4(), username="alice", org_id=uuid.uuid4(), roles=["admin"])
        result = await check(current_user=user)
        assert result is user

    @pytest.mark.asyncio
    async def test_viewer_blocked_from_editor_endpoint(self):
        check = require_role("admin", "editor")
        user = CurrentUser(id=uuid.uuid4(), username="viewer_user", org_id=uuid.uuid4(), roles=["viewer"])
        with pytest.raises(HTTPException) as exc_info:
            await check(current_user=user)
        assert exc_info.value.status_code == 403
        assert "admin" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_no_roles_blocked(self):
        check = require_role("admin")
        user = CurrentUser(id=uuid.uuid4(), username="noroles", org_id=uuid.uuid4(), roles=[])
        with pytest.raises(HTTPException) as exc_info:
            await check(current_user=user)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_multi_role_user_passes_if_any_match(self):
        check = require_role("editor")
        user = CurrentUser(id=uuid.uuid4(), username="multi", org_id=uuid.uuid4(), roles=["viewer", "editor"])
        result = await check(current_user=user)
        assert result is user


class TestIngestRequiresEditorRole:
    """Integration-level: verify /ingest returns 403 for viewer-only users."""

    def test_viewer_cannot_ingest(self):
        from app.api.deps import get_current_user
        from app.main import app
        from fastapi.testclient import TestClient

        viewer = CurrentUser(id=uuid.uuid4(), username="viewer", org_id=uuid.uuid4(), roles=["viewer"])
        app.dependency_overrides[get_current_user] = lambda: viewer

        mock_conn = AsyncMock()
        mock_conn.run_sync = AsyncMock()
        mock_engine_connect = AsyncMock()
        mock_engine_connect.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine_connect.__aexit__ = AsyncMock(return_value=False)

        with patch("app.main.engine") as mock_engine:
            mock_engine.begin.return_value = mock_engine_connect
            client = TestClient(app)
            resp = client.post("/ingest", files={"file": ("a.txt", b"hi", "text/plain")})

        assert resp.status_code == 403
        assert "editor" in resp.json()["detail"]

        app.dependency_overrides.clear()

    def test_editor_can_ingest(self):
        from pathlib import Path
        from app.api.deps import get_current_user
        from app.db.engine import get_db
        from app.main import app
        from fastapi.testclient import TestClient

        editor = CurrentUser(id=uuid.uuid4(), username="editor", org_id=uuid.uuid4(), roles=["editor"])
        app.dependency_overrides[get_current_user] = lambda: editor

        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=mock_result)
        session.add = MagicMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.flush = AsyncMock()

        async def override_get_db():
            yield session

        app.dependency_overrides[get_db] = override_get_db

        mock_conn = AsyncMock()
        mock_conn.run_sync = AsyncMock()
        mock_engine_connect = AsyncMock()
        mock_engine_connect.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine_connect.__aexit__ = AsyncMock(return_value=False)

        fixtures = Path(__file__).parent / "fixtures"

        with patch("app.main.engine") as mock_engine:
            mock_engine.begin.return_value = mock_engine_connect
            client = TestClient(app)

            with patch("app.api.ingest.load_and_split_file") as mock_load, \
                 patch("app.api.ingest.add_documents"):
                from langchain_core.documents import Document
                mock_load.return_value = [Document(page_content="test")]
                with open(fixtures / "sample.txt", "rb") as f:
                    resp = client.post("/ingest", files={"file": ("sample.txt", f, "text/plain")})

        assert resp.status_code == 200

        app.dependency_overrides.clear()


class TestKBCreateRequiresEditor:
    def test_viewer_cannot_create_kb(self):
        from app.api.deps import get_current_user
        from app.db.engine import get_db
        from app.main import app
        from fastapi.testclient import TestClient

        viewer = CurrentUser(id=uuid.uuid4(), username="viewer", org_id=uuid.uuid4(), roles=["viewer"])
        app.dependency_overrides[get_current_user] = lambda: viewer

        session = AsyncMock()
        async def override_get_db():
            yield session
        app.dependency_overrides[get_db] = override_get_db

        mock_conn = AsyncMock()
        mock_conn.run_sync = AsyncMock()
        mock_engine_connect = AsyncMock()
        mock_engine_connect.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine_connect.__aexit__ = AsyncMock(return_value=False)

        with patch("app.main.engine") as mock_engine:
            mock_engine.begin.return_value = mock_engine_connect
            client = TestClient(app)
            resp = client.post("/knowledge-bases", json={"name": "Test"})

        assert resp.status_code == 403

        app.dependency_overrides.clear()
