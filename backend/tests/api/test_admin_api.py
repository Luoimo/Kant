from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.admin import router as admin_router
from api.deps import require_admin


def test_member_cannot_access_admin_route():
    app = FastAPI()
    app.include_router(admin_router)
    client = TestClient(app)
    resp = client.get("/api/admin/users")
    assert resp.status_code in {401, 403}


def test_admin_route_reachable_with_override():
    import api.admin as admin_module

    app = FastAPI()
    app.include_router(admin_router)
    app.dependency_overrides[require_admin] = lambda: {"user_id": "admin-1", "role": "admin"}
    app.dependency_overrides[admin_module.AuditLogCatalog] = lambda: None
    app.dependency_overrides[admin_module.UserCatalog] = lambda: None
    client = TestClient(app)
    # 使用 monkeypatch 避免真实数据库依赖
    class _FakeAudit:
        def write(self, **kwargs):
            return None

    class _FakeUsers:
        def list_all(self):
            return [{"user_id": "u1", "email": "u1@test.com", "role": "member", "status": "active"}]

    admin_module.AuditLogCatalog = lambda: _FakeAudit()
    admin_module.UserCatalog = lambda: _FakeUsers()
    resp = client.get("/api/admin/users")
    assert resp.status_code == 200
