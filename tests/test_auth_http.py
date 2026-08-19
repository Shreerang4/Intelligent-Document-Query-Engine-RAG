from __future__ import annotations

import uuid
import importlib
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import backend.app.auth.router as auth_router_module
from backend.app.auth.config import ACCESS_JWT_AUDIENCE, ACCESS_JWT_ISSUER
from backend.app.auth.refresh_sessions import create_refresh_session, hash_refresh_token
from backend.app.auth.http_errors import install_auth_http_error_handlers
from backend.app.auth.router import router
from backend.app.auth.services import register_user
from backend.app.auth.tokens import create_access_token
from persistence.db import Base, get_session
from persistence.models import RefreshSession, User


TEST_SECRET = "phase-4b-test-secret-that-is-never-used-in-production"
SAME_ORIGIN = "http://testserver"
ORIGIN_HEADERS = {"Origin": SAME_ORIGIN}
PASSWORD = "fifteen chars ok!"


@pytest.fixture(autouse=True)
def auth_environment(monkeypatch):
    monkeypatch.setenv("ACCESS_JWT_SECRET", TEST_SECRET)
    monkeypatch.delenv("ACCESS_TOKEN_TTL_SECONDS", raising=False)
    monkeypatch.setenv("REFRESH_COOKIE_SECURE", "false")
    monkeypatch.delenv("AUTH_ALLOWED_ORIGINS", raising=False)


@pytest.fixture
def sqlite_engine():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def app(sqlite_engine) -> FastAPI:
    test_app = FastAPI()
    install_auth_http_error_handlers(test_app)

    def override_get_session():
        with Session(sqlite_engine) as session:
            yield session

    test_app.dependency_overrides[get_session] = override_get_session
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app):
    with TestClient(app, base_url=SAME_ORIGIN) as test_client:
        yield test_client


def _register(client: TestClient, *, email: str = "person@EXAMPLE.COM"):
    return client.post(
        "/auth/register",
        headers=ORIGIN_HEADERS,
        json={"email": email, "password": PASSWORD, "display_name": "Person"},
    )


def _seed_user(engine, *, email: str = "person@example.com", password: str = PASSWORD) -> str:
    with Session(engine) as session:
        return register_user(session, email=email, password=password).id


def _auth_claims(token: str) -> dict:
    return jwt.decode(
        token,
        TEST_SECRET,
        algorithms=["HS256"],
        issuer=ACCESS_JWT_ISSUER,
        audience=ACCESS_JWT_AUDIENCE,
    )


def test_register_returns_201_normalized_public_user_and_credentials(client, sqlite_engine) -> None:
    response = _register(client)

    assert response.status_code == 201
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 600
    assert _auth_claims(body["access_token"])["sub"] == body["user"]["id"]
    assert body["user"]["email"] == "person@example.com"
    assert set(body["user"]) == {"id", "email", "display_name", "auth_provider", "created_at"}
    assert "password_hash" not in response.text
    assert "refresh" not in body

    with Session(sqlite_engine) as session:
        user = session.get(User, body["user"]["id"])
        assert user is not None
        assert user.email == "person@example.com"
        assert user.password_hash is not None


@pytest.mark.parametrize(
    "password",
    ["x" * 14, "x" * 129],
)
def test_register_enforces_password_length_policy(client, password: str) -> None:
    response = client.post(
        "/auth/register",
        headers=ORIGIN_HEADERS,
        json={"email": "policy@example.com", "password": password},
    )
    assert response.status_code == 422
    assert password not in response.text


def test_register_accepts_spaces_unicode_and_symbols_at_minimum_length(client) -> None:
    response = client.post(
        "/auth/register",
        headers=ORIGIN_HEADERS,
        json={"email": "unicode@example.com", "password": " 密碼 phrase 🔐!!!"},
    )
    assert response.status_code == 201


def test_register_rejects_invalid_email_and_normalized_duplicate(client) -> None:
    invalid = client.post(
        "/auth/register",
        headers=ORIGIN_HEADERS,
        json={"email": "not-an-email", "password": PASSWORD},
    )
    first = _register(client, email="duplicate@EXAMPLE.COM")
    duplicate = _register(client, email="duplicate@example.com")

    assert invalid.status_code == 422
    assert first.status_code == 201
    assert duplicate.status_code == 409


def test_failed_initial_refresh_creation_rolls_back_registered_user(
    client,
    sqlite_engine,
    monkeypatch,
) -> None:
    def fail_refresh_creation(*_args, **_kwargs):
        raise SQLAlchemyError("forced refresh-session failure")

    monkeypatch.setattr(auth_router_module, "add_refresh_session", fail_refresh_creation)
    response = _register(client, email="rollback@example.com")

    assert response.status_code == 503
    assert response.json() == {"detail": "Authentication service is unavailable."}
    with Session(sqlite_engine) as session:
        assert session.scalar(select(func.count()).select_from(User)) == 0
        assert session.scalar(select(func.count()).select_from(RefreshSession)) == 0


def test_login_returns_generic_401_for_wrong_password_and_unknown_email(client, sqlite_engine) -> None:
    _seed_user(sqlite_engine)
    wrong = client.post(
        "/auth/login",
        headers=ORIGIN_HEADERS,
        json={"email": "person@example.com", "password": "incorrect password"},
    )
    unknown = client.post(
        "/auth/login",
        headers=ORIGIN_HEADERS,
        json={"email": "unknown@example.com", "password": "incorrect password"},
    )

    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json() == {"detail": "Invalid email or password."}
    assert wrong.headers["www-authenticate"] == unknown.headers["www-authenticate"] == "Bearer"


def test_valid_login_is_public_and_each_login_creates_independent_session(
    client,
    sqlite_engine,
) -> None:
    user_id = _seed_user(sqlite_engine)
    first = client.post(
        "/auth/login",
        headers=ORIGIN_HEADERS,
        json={"email": "person@EXAMPLE.COM", "password": PASSWORD},
    )
    first_cookie = first.cookies["idqe_refresh"]
    second = client.post(
        "/auth/login",
        headers=ORIGIN_HEADERS,
        json={"email": "person@example.com", "password": PASSWORD},
    )
    second_cookie = second.cookies["idqe_refresh"]

    assert first.status_code == second.status_code == 200
    assert first_cookie != second_cookie
    assert _auth_claims(first.json()["access_token"])["sub"] == user_id
    assert "password" not in first.json()
    assert "password_hash" not in first.json()["user"]
    assert first_cookie not in first.text
    with Session(sqlite_engine) as session:
        sessions = session.scalars(
            select(RefreshSession).where(RefreshSession.user_id == user_id)
        ).all()
        assert len(sessions) == 2
        assert all(item.revoked_at is None for item in sessions)


def test_refresh_cookie_has_shared_attributes_and_matches_database_expiry(
    client,
    sqlite_engine,
) -> None:
    response = _register(client, email="cookie@example.com")
    raw_token = response.cookies["idqe_refresh"]
    set_cookie = response.headers["set-cookie"]

    assert "HttpOnly" in set_cookie
    assert "SameSite=lax" in set_cookie
    assert "Path=/auth" in set_cookie
    assert "Domain=" not in set_cookie
    assert "Secure" not in set_cookie

    with Session(sqlite_engine) as session:
        stored = session.scalar(
            select(RefreshSession).where(
                RefreshSession.token_hash == hash_refresh_token(raw_token)
            )
        )
        assert stored is not None
        expires_at = stored.expires_at.replace(tzinfo=timezone.utc)

    cookie_parts = {
        part.split("=", 1)[0].strip().lower(): part.split("=", 1)[1].strip()
        for part in set_cookie.split(";")
        if "=" in part
    }
    header_expiry = parsedate_to_datetime(cookie_parts["expires"])
    assert abs((header_expiry - expires_at).total_seconds()) < 1
    assert abs(int(cookie_parts["max-age"]) - (expires_at - datetime.now(timezone.utc)).total_seconds()) < 3


def test_refresh_cookie_is_secure_by_default_and_can_be_disabled_for_localhost(
    app,
    monkeypatch,
) -> None:
    monkeypatch.delenv("REFRESH_COOKIE_SECURE", raising=False)
    with TestClient(app, base_url=SAME_ORIGIN) as production_client:
        production = _register(production_client, email="production-cookie@example.com")
    assert "Secure" in production.headers["set-cookie"]

    monkeypatch.setenv("REFRESH_COOKIE_SECURE", "false")
    with TestClient(app, base_url=SAME_ORIGIN) as local_client:
        local = _register(local_client, email="local-cookie@example.com")
    assert "Secure" not in local.headers["set-cookie"]


def test_refresh_rotates_cookie_digest_and_preserves_absolute_expiry(client, sqlite_engine) -> None:
    login = _register(client, email="rotate@example.com")
    first_raw = login.cookies["idqe_refresh"]
    with Session(sqlite_engine) as session:
        first = session.scalar(
            select(RefreshSession).where(RefreshSession.token_hash == hash_refresh_token(first_raw))
        )
        assert first is not None
        first_id = first.id
        absolute_expiry = first.expires_at

    refreshed = client.post("/auth/refresh", headers=ORIGIN_HEADERS)
    second_raw = refreshed.cookies["idqe_refresh"]

    assert refreshed.status_code == 200
    assert refreshed.headers["cache-control"] == "no-store"
    assert second_raw != first_raw
    assert first_raw not in refreshed.text
    assert second_raw not in refreshed.text
    refreshed_cookie_parts = {
        part.split("=", 1)[0].strip().lower(): part.split("=", 1)[1].strip()
        for part in refreshed.headers["set-cookie"].split(";")
        if "=" in part
    }
    inherited_cookie_expiry = parsedate_to_datetime(refreshed_cookie_parts["expires"])
    expected_expiry = absolute_expiry.replace(tzinfo=timezone.utc)
    assert abs((inherited_cookie_expiry - expected_expiry).total_seconds()) < 1
    with Session(sqlite_engine) as session:
        original = session.get(RefreshSession, first_id)
        replacement = session.scalar(
            select(RefreshSession).where(RefreshSession.token_hash == hash_refresh_token(second_raw))
        )
        assert original is not None and replacement is not None
        assert original.revoked_at is not None
        assert original.replaced_by_session_id == replacement.id
        assert replacement.expires_at == absolute_expiry
        assert replacement.token_hash == hash_refresh_token(second_raw)

    with TestClient(client.app, base_url=SAME_ORIGIN) as stale_client:
        stale_client.cookies.set("idqe_refresh", first_raw, path="/auth")
        stale = stale_client.post("/auth/refresh", headers=ORIGIN_HEADERS)
    assert stale.status_code == 401
    assert stale.json() == {"detail": "Invalid refresh credential."}
    assert "set-cookie" not in stale.headers


@pytest.mark.parametrize("credential_kind", ["missing", "random", "expired", "revoked"])
def test_refresh_failures_collapse_to_generic_401(
    app,
    sqlite_engine,
    credential_kind: str,
) -> None:
    raw_token = None
    if credential_kind in {"expired", "revoked"}:
        user_id = _seed_user(sqlite_engine, email=f"{credential_kind}@example.com")
        with Session(sqlite_engine) as session:
            created = create_refresh_session(session, user_id)
            raw_token = created.raw_token
            stored = session.scalar(
                select(RefreshSession).where(
                    RefreshSession.token_hash == hash_refresh_token(raw_token)
                )
            )
            assert stored is not None
            if credential_kind == "expired":
                stored.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            else:
                stored.revoked_at = datetime.now(timezone.utc)
            session.commit()
    elif credential_kind == "random":
        raw_token = "random-refresh-credential"

    with TestClient(app, base_url=SAME_ORIGIN) as refresh_client:
        if raw_token:
            refresh_client.cookies.set("idqe_refresh", raw_token, path="/auth")
        response = refresh_client.post("/auth/refresh", headers=ORIGIN_HEADERS)

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid refresh credential."}
    assert response.headers["www-authenticate"] == "Bearer"
    if credential_kind in {"random", "revoked"}:
        assert "set-cookie" not in response.headers
    if credential_kind == "expired":
        assert "Max-Age=0" in response.headers["set-cookie"]


def test_logout_revokes_only_current_device_clears_cookie_and_is_idempotent(
    app,
    sqlite_engine,
) -> None:
    user_id = _seed_user(sqlite_engine, email="logout@example.com")
    with TestClient(app, base_url=SAME_ORIGIN) as first_client, TestClient(
        app, base_url=SAME_ORIGIN
    ) as second_client:
        first_login = first_client.post(
            "/auth/login",
            headers=ORIGIN_HEADERS,
            json={"email": "logout@example.com", "password": PASSWORD},
        )
        second_login = second_client.post(
            "/auth/login",
            headers=ORIGIN_HEADERS,
            json={"email": "logout@example.com", "password": PASSWORD},
        )
        first_raw = first_login.cookies["idqe_refresh"]
        second_raw = second_login.cookies["idqe_refresh"]

        logout = first_client.post("/auth/logout", headers=ORIGIN_HEADERS)
        repeated = first_client.post("/auth/logout", headers=ORIGIN_HEADERS)

    assert logout.status_code == repeated.status_code == 204
    assert logout.content == b""
    clear_header = logout.headers["set-cookie"]
    assert "idqe_refresh=" in clear_header
    assert "Max-Age=0" in clear_header
    assert "HttpOnly" in clear_header
    assert "SameSite=lax" in clear_header
    assert "Path=/auth" in clear_header
    assert "Domain=" not in clear_header

    with Session(sqlite_engine) as session:
        first_session = session.scalar(
            select(RefreshSession).where(
                RefreshSession.token_hash == hash_refresh_token(first_raw)
            )
        )
        second_session = session.scalar(
            select(RefreshSession).where(
                RefreshSession.token_hash == hash_refresh_token(second_raw)
            )
        )
        assert first_session is not None and first_session.revoked_at is not None
        assert second_session is not None and second_session.revoked_at is None
        assert first_session.user_id == second_session.user_id == user_id


def test_me_returns_public_current_user_and_deleted_user_is_unauthorized(
    client,
    sqlite_engine,
) -> None:
    user_id = _seed_user(sqlite_engine, email="me@example.com")
    token = create_access_token(user_id)
    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json()["email"] == "me@example.com"
    assert "password_hash" not in response.text

    with Session(sqlite_engine) as session:
        user = session.get(User, user_id)
        assert user is not None
        session.delete(user)
        session.commit()

    deleted = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert deleted.status_code == 401
    assert deleted.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("token", ["not-a-jwt", None])
def test_me_rejects_invalid_or_missing_access_token(client, token: str | None) -> None:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = client.get("/auth/me", headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_me_rejects_expired_access_token(client) -> None:
    now = int(datetime.now(timezone.utc).timestamp())
    token = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "iat": now - 700,
            "exp": now - 100,
            "jti": str(uuid.uuid4()),
            "iss": ACCESS_JWT_ISSUER,
            "aud": ACCESS_JWT_AUDIENCE,
        },
        TEST_SECRET,
        algorithm="HS256",
    )
    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_same_origin_posts_succeed_and_cross_origin_auth_posts_are_rejected(
    client,
    sqlite_engine,
) -> None:
    same_origin_register = _register(client, email="origin@example.com")
    assert same_origin_register.status_code == 201

    cross_origin = {"Origin": "https://attacker.example"}
    refresh = client.post("/auth/refresh", headers=cross_origin)
    logout = client.post("/auth/logout", headers=cross_origin)
    login = client.post(
        "/auth/login",
        headers=cross_origin,
        json={"email": "origin@example.com", "password": PASSWORD},
    )
    register = client.post(
        "/auth/register",
        headers=cross_origin,
        json={"email": "attacker@example.com", "password": PASSWORD},
    )

    assert refresh.status_code == logout.status_code == 403
    assert login.status_code == register.status_code == 403
    with Session(sqlite_engine) as session:
        assert session.scalar(select(func.count()).select_from(User)) == 1


def test_explicit_production_origin_does_not_allow_other_origins(client, monkeypatch) -> None:
    monkeypatch.setenv("AUTH_ALLOWED_ORIGINS", "https://app.example.com")

    allowed = client.post(
        "/auth/register",
        headers={"Origin": "https://app.example.com"},
        json={"email": "allowed-origin@example.com", "password": PASSWORD},
    )
    rejected = client.post(
        "/auth/register",
        headers={"Origin": "https://other.example.com"},
        json={"email": "rejected-origin@example.com", "password": PASSWORD},
    )

    assert allowed.status_code == 201
    assert rejected.status_code == 403


def test_root_application_registers_auth_router_before_spa_catch_all() -> None:
    application_module = importlib.import_module("main")
    paths = [route.path for route in application_module.app.routes]

    assert "/auth/register" in paths
    assert "/auth/login" in paths
    assert "/auth/refresh" in paths
    assert "/auth/logout" in paths
    assert "/auth/me" in paths
    assert paths.index("/auth/register") < paths.index("/{full_path:path}")
