from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.security_headers import (
    PRODUCTION_CONTENT_SECURITY_POLICY,
    install_security_headers,
)


EXPECTED_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; "
    "img-src 'self'; font-src 'self'; object-src 'none'; base-uri 'self'; "
    "form-action 'self'; frame-src 'none'; frame-ancestors 'none'; worker-src 'none'"
)


def _test_app() -> FastAPI:
    app = FastAPI()
    install_security_headers(app)

    @app.get("/")
    async def index():
        return {"status": "ok"}

    return app


def test_production_csp_is_enforcing_and_exact() -> None:
    assert PRODUCTION_CONTENT_SECURITY_POLICY == EXPECTED_CSP
    assert "'unsafe-inline'" not in PRODUCTION_CONTENT_SECURITY_POLICY
    assert "'unsafe-eval'" not in PRODUCTION_CONTENT_SECURITY_POLICY
    assert "*" not in PRODUCTION_CONTENT_SECURITY_POLICY

    response = TestClient(_test_app()).get("/")

    assert response.status_code == 200
    assert response.headers["Content-Security-Policy"] == EXPECTED_CSP
    assert "Content-Security-Policy-Report-Only" not in response.headers


def test_nosniff_header_is_enabled() -> None:
    response = TestClient(_test_app()).get("/")

    assert response.headers["X-Content-Type-Options"] == "nosniff"
