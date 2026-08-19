from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime, timezone
from typing import Annotated

import jwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.auth.config import (
    ACCESS_JWT_ALGORITHM,
    ACCESS_JWT_AUDIENCE,
    ACCESS_JWT_ISSUER,
)
from backend.app.auth.dependencies import get_authenticated_user_id, get_current_user
from backend.app.auth.errors import AuthConfigurationError, InvalidAccessTokenError
from backend.app.auth.tokens import create_access_token, decode_access_token
from persistence.db import Base, get_session
from persistence.models import User


TEST_SECRET = "phase-3-test-secret-that-is-not-used-in-production"
EXPECTED_CLAIMS = {"sub", "iat", "exp", "jti", "iss", "aud"}


@pytest.fixture(autouse=True)
def jwt_environment(monkeypatch):
    monkeypatch.setenv("ACCESS_JWT_SECRET", TEST_SECRET)
    monkeypatch.delenv("ACCESS_TOKEN_TTL_SECONDS", raising=False)


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


def _new_user_id() -> str:
    return str(uuid.uuid4())


def _claims(user_id: str | None = None) -> dict:
    issued_at = int(datetime.now(timezone.utc).timestamp())
    return {
        "sub": user_id or _new_user_id(),
        "iat": issued_at,
        "exp": issued_at + 900,
        "jti": str(uuid.uuid4()),
        "iss": ACCESS_JWT_ISSUER,
        "aud": ACCESS_JWT_AUDIENCE,
    }


def _encode(claims: dict, *, secret: str = TEST_SECRET, algorithm: str = "HS256") -> str:
    return jwt.encode(claims, secret, algorithm=algorithm)


def _build_test_app(engine) -> FastAPI:
    app = FastAPI()

    def override_get_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session

    @app.get("/identity")
    def identity(user_id: Annotated[str, Depends(get_authenticated_user_id)]) -> dict[str, str]:
        return {"id": user_id}

    @app.get("/protected")
    def protected(current_user: Annotated[User, Depends(get_current_user)]) -> dict[str, str]:
        return {"id": current_user.id, "email": current_user.email or ""}

    return app


def test_valid_access_token_round_trip_and_subject() -> None:
    user_id = _new_user_id()

    token = create_access_token(user_id)
    payload = decode_access_token(token)

    assert payload["sub"] == user_id


def test_access_token_has_expected_minimal_claims_and_lifetime() -> None:
    payload = decode_access_token(create_access_token(_new_user_id()))

    assert set(payload) == EXPECTED_CLAIMS
    assert payload["exp"] - payload["iat"] == 600
    assert payload["iss"] == ACCESS_JWT_ISSUER
    assert payload["aud"] == ACCESS_JWT_AUDIENCE
    assert str(uuid.UUID(payload["jti"])) == payload["jti"]


def test_expired_access_token_is_rejected() -> None:
    claims = _claims()
    claims["iat"] -= 1_000
    claims["exp"] = claims["iat"] + 1

    with pytest.raises(InvalidAccessTokenError):
        decode_access_token(_encode(claims))


def test_tampered_signature_is_rejected() -> None:
    token = create_access_token(_new_user_id())
    header, payload, signature = token.split(".")
    tampered_signature = ("A" if signature[0] != "A" else "B") + signature[1:]

    with pytest.raises(InvalidAccessTokenError):
        decode_access_token(".".join((header, payload, tampered_signature)))


def test_tampered_payload_is_rejected() -> None:
    token = create_access_token(_new_user_id())
    header, encoded_payload, signature = token.split(".")
    padding = "=" * (-len(encoded_payload) % 4)
    payload = json.loads(base64.urlsafe_b64decode(encoded_payload + padding))
    payload["sub"] = _new_user_id()
    tampered_payload = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).rstrip(b"=").decode("ascii")

    with pytest.raises(InvalidAccessTokenError):
        decode_access_token(".".join((header, tampered_payload, signature)))


def test_wrong_signing_secret_is_rejected(monkeypatch) -> None:
    token = create_access_token(_new_user_id())
    monkeypatch.setenv("ACCESS_JWT_SECRET", "a-different-phase-3-secret-with-32-bytes")

    with pytest.raises(InvalidAccessTokenError):
        decode_access_token(token)


def test_wrong_algorithm_is_rejected() -> None:
    token = _encode(_claims(), algorithm="HS384")

    with pytest.raises(InvalidAccessTokenError):
        decode_access_token(token)


def test_wrong_issuer_is_rejected() -> None:
    claims = _claims()
    claims["iss"] = "wrong-issuer"

    with pytest.raises(InvalidAccessTokenError):
        decode_access_token(_encode(claims))


def test_wrong_audience_is_rejected() -> None:
    claims = _claims()
    claims["aud"] = "wrong-audience"

    with pytest.raises(InvalidAccessTokenError):
        decode_access_token(_encode(claims))


@pytest.mark.parametrize("missing_claim", sorted(EXPECTED_CLAIMS))
def test_missing_required_claim_is_rejected(missing_claim: str) -> None:
    claims = _claims()
    claims.pop(missing_claim)

    with pytest.raises(InvalidAccessTokenError):
        decode_access_token(_encode(claims))


@pytest.mark.parametrize("malformed_token", ["", "not-a-jwt", "one.two.three"])
def test_malformed_token_is_rejected(malformed_token: str) -> None:
    with pytest.raises(InvalidAccessTokenError):
        decode_access_token(malformed_token)


@pytest.mark.parametrize("invalid_user_id", ["", " ", "not-a-uuid"])
def test_malformed_or_empty_user_id_is_rejected(invalid_user_id: str) -> None:
    claims = _claims()
    claims["sub"] = invalid_user_id

    with pytest.raises(InvalidAccessTokenError):
        decode_access_token(_encode(claims))


@pytest.mark.parametrize("path", ["/identity", "/protected"])
def test_missing_authorization_header_returns_bearer_401(sqlite_engine, path: str) -> None:
    with TestClient(_build_test_app(sqlite_engine)) as client:
        response = client.get(path)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_authenticated_user_id_returns_subject_without_database_lookup(sqlite_engine, monkeypatch) -> None:
    user_id = _new_user_id()
    token = create_access_token(user_id)

    def fail_if_user_lookup_occurs(*_args, **_kwargs):
        raise AssertionError("The stateless dependency must not query User.")

    monkeypatch.setattr(Session, "get", fail_if_user_lookup_occurs)
    with TestClient(_build_test_app(sqlite_engine)) as client:
        response = client.get("/identity", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json() == {"id": user_id}


def test_both_dependencies_apply_the_same_jwt_validation_rules(sqlite_engine) -> None:
    expired_claims = _claims()
    expired_claims["iat"] -= 1_000
    expired_claims["exp"] = expired_claims["iat"] + 1
    expired_token = _encode(expired_claims)

    valid_token = create_access_token(_new_user_id())
    header, payload, signature = valid_token.split(".")
    tampered_signature = ("A" if signature[0] != "A" else "B") + signature[1:]
    tampered_token = ".".join((header, payload, tampered_signature))

    with TestClient(_build_test_app(sqlite_engine)) as client:
        for path in ("/identity", "/protected"):
            for token in ("not-a-jwt", expired_token, tampered_token):
                response = client.get(path, headers={"Authorization": f"Bearer {token}"})
                assert response.status_code == 401
                assert response.headers["www-authenticate"] == "Bearer"


def test_nonexistent_user_returns_bearer_401(sqlite_engine) -> None:
    token = create_access_token(_new_user_id())

    with TestClient(_build_test_app(sqlite_engine)) as client:
        response = client.get("/protected", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_current_user_performs_lookup_and_resolves_correct_user(sqlite_engine, monkeypatch) -> None:
    with Session(sqlite_engine) as session:
        user = User(
            email="jwt-user@example.com",
            auth_provider="password",
            password_hash="$argon2id$test-only-placeholder",
        )
        session.add(user)
        session.commit()
        user_id = user.id

    original_session_get = Session.get
    lookup_calls: list[tuple[object, object]] = []

    def tracking_session_get(session, entity, identity, *args, **kwargs):
        lookup_calls.append((entity, identity))
        return original_session_get(session, entity, identity, *args, **kwargs)

    monkeypatch.setattr(Session, "get", tracking_session_get)
    token = create_access_token(user_id)
    with TestClient(_build_test_app(sqlite_engine)) as client:
        response = client.get("/protected", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json() == {"id": user_id, "email": "jwt-user@example.com"}
    assert lookup_calls == [(User, user_id)]


def test_deleted_user_still_has_stateless_identity_but_not_current_user(sqlite_engine) -> None:
    with Session(sqlite_engine) as session:
        user = User(
            email="deleted-jwt-user@example.com",
            auth_provider="password",
            password_hash="$argon2id$test-only-placeholder",
        )
        session.add(user)
        session.commit()
        user_id = user.id
        token = create_access_token(user_id)
        session.delete(user)
        session.commit()

    with TestClient(_build_test_app(sqlite_engine)) as client:
        identity_response = client.get(
            "/identity",
            headers={"Authorization": f"Bearer {token}"},
        )
        current_user_response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert identity_response.status_code == 200
    assert identity_response.json() == {"id": user_id}
    assert current_user_response.status_code == 401
    assert current_user_response.headers["www-authenticate"] == "Bearer"


def test_issued_tokens_have_unique_jti_values() -> None:
    user_id = _new_user_id()
    first = decode_access_token(create_access_token(user_id))
    second = decode_access_token(create_access_token(user_id))

    assert first["jti"] != second["jti"]


def test_jwt_payload_contains_no_user_or_document_data() -> None:
    payload = decode_access_token(create_access_token(_new_user_id()))

    assert set(payload) == EXPECTED_CLAIMS
    assert not {
        "email",
        "display_name",
        "password_hash",
        "auth_provider",
        "documents",
        "document_ids",
        "user",
    }.intersection(payload)


def test_missing_secret_fails_only_when_jwt_functionality_is_used(monkeypatch) -> None:
    monkeypatch.delenv("ACCESS_JWT_SECRET", raising=False)

    with pytest.raises(AuthConfigurationError, match="ACCESS_JWT_SECRET"):
        create_access_token(_new_user_id())


def test_short_secret_fails_clearly_when_jwt_functionality_is_used(monkeypatch) -> None:
    monkeypatch.setenv("ACCESS_JWT_SECRET", "too-short")

    with pytest.raises(AuthConfigurationError, match="at least 32 bytes"):
        create_access_token(_new_user_id())


def test_access_token_ttl_can_be_configured(monkeypatch) -> None:
    monkeypatch.setenv("ACCESS_TOKEN_TTL_SECONDS", "120")

    payload = decode_access_token(create_access_token(_new_user_id()))

    assert payload["exp"] - payload["iat"] == 120
