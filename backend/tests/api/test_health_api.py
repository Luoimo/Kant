from fastapi.testclient import TestClient

from config import Settings
from main import app


def test_healthz_returns_ok():
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_responses_include_security_headers():
    response = TestClient(app).get("/healthz")

    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cross-origin-resource-policy"] == "same-origin"
    assert response.headers["permissions-policy"] == (
        "camera=(), microphone=(), geolocation=()"
    )
    assert response.headers["content-security-policy"] == (
        "default-src 'none'; frame-ancestors 'none'"
    )


def test_settings_parses_cors_allow_origins():
    settings = Settings(
        cors_allow_origins="https://kant.example.com, https://api.kant.example.com"
    )

    assert settings.cors_origins == [
        "https://kant.example.com",
        "https://api.kant.example.com",
    ]


def test_settings_uses_localhost_cors_defaults():
    settings = Settings(cors_allow_origins="")

    assert settings.cors_origins == [
        "http://localhost:5173",
        "http://localhost:3000",
    ]


def test_knowledge_graph_is_disabled_by_default():
    settings = Settings()

    assert settings.enable_knowledge_graph is False
