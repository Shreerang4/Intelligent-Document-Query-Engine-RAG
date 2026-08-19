from __future__ import annotations

import base64
import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

import backend.app.auth.refresh_sessions as refresh_services
from backend.app.auth.errors import (
    ExpiredRefreshSessionError,
    InvalidRefreshTokenError,
    RefreshSessionUserNotFoundError,
    RevokedRefreshSessionError,
)
from backend.app.auth.refresh_sessions import (
    REFRESH_TOKEN_RANDOM_BYTES,
    create_refresh_session,
    generate_refresh_token,
    hash_refresh_token,
    resolve_refresh_session,
    revoke_refresh_session,
    rotate_refresh_session,
)
from persistence.db import Base
from persistence.models import RefreshSession, User


@pytest.fixture
def sqlite_engine():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)

    @event.listens_for(engine, "connect")
    def enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def _create_user(session: Session, email: str = "refresh-user@example.com") -> User:
    user = User(
        email=email,
        auth_provider="password",
        password_hash="$argon2id$test-only-placeholder",
    )
    session.add(user)
    session.commit()
    return user


def _stored_session(session: Session, session_id: str) -> RefreshSession:
    stored = session.get(RefreshSession, session_id)
    assert stored is not None
    return stored


def test_generated_refresh_token_uses_at_least_256_random_bits(monkeypatch) -> None:
    requested_byte_counts: list[int] = []
    original_token_urlsafe = secrets.token_urlsafe

    def recording_token_urlsafe(byte_count: int) -> str:
        requested_byte_counts.append(byte_count)
        return original_token_urlsafe(byte_count)

    monkeypatch.setattr(refresh_services.secrets, "token_urlsafe", recording_token_urlsafe)
    raw_token = generate_refresh_token()
    padding = "=" * (-len(raw_token) % 4)

    assert requested_byte_counts == [REFRESH_TOKEN_RANDOM_BYTES]
    assert REFRESH_TOKEN_RANDOM_BYTES >= 32
    assert len(base64.urlsafe_b64decode(raw_token + padding)) >= 32


def test_creation_stores_only_sha256_digest_and_resolves_correct_user(sqlite_engine) -> None:
    with Session(sqlite_engine) as session:
        user = _create_user(session)
        created = create_refresh_session(session, user.id)
        stored = _stored_session(session, created.session.id)

        assert created.raw_token
        assert created.raw_token not in repr(created)
        assert "raw_token" not in RefreshSession.__table__.c
        assert "token_hash" not in created.session.__dataclass_fields__
        assert stored.token_hash == hashlib.sha256(created.raw_token.encode("utf-8")).digest()
        assert stored.token_hash == hash_refresh_token(created.raw_token)
        assert stored.token_hash != created.raw_token.encode("utf-8")
        assert len(stored.token_hash) == 32
        assert created.session.expires_at - created.session.created_at == timedelta(days=7)
        assert created.session.revoked_at is None
        assert created.session.replaced_by_session_id is None

        resolved = resolve_refresh_session(session, created.raw_token)
        assert resolved.session.id == created.session.id
        assert resolved.session.user_id == user.id
        assert resolved.user.id == user.id


def test_one_user_can_have_multiple_independent_active_sessions(sqlite_engine) -> None:
    with Session(sqlite_engine) as session:
        user = _create_user(session)
        laptop = create_refresh_session(session, user.id)
        phone = create_refresh_session(session, user.id)

        assert laptop.raw_token != phone.raw_token
        assert laptop.session.id != phone.session.id
        assert resolve_refresh_session(session, laptop.raw_token).user.id == user.id
        assert resolve_refresh_session(session, phone.raw_token).user.id == user.id
        active_count = session.scalar(
            select(func.count()).select_from(RefreshSession).where(
                RefreshSession.user_id == user.id,
                RefreshSession.revoked_at.is_(None),
            )
        )
        assert active_count == 2


def test_modified_random_or_digest_credential_does_not_resolve(sqlite_engine) -> None:
    with Session(sqlite_engine) as session:
        user = _create_user(session)
        created = create_refresh_session(session, user.id)
        stored_digest = _stored_session(session, created.session.id).token_hash

        for invalid_token in (
            f"{created.raw_token}modified",
            generate_refresh_token(),
            stored_digest.hex(),
        ):
            with pytest.raises(InvalidRefreshTokenError):
                resolve_refresh_session(session, invalid_token)

        with pytest.raises(InvalidRefreshTokenError):
            resolve_refresh_session(session, stored_digest)  # type: ignore[arg-type]


def test_expired_refresh_session_is_rejected(sqlite_engine) -> None:
    with Session(sqlite_engine) as session:
        user = _create_user(session)
        created = create_refresh_session(session, user.id)
        stored = _stored_session(session, created.session.id)
        stored.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

        with pytest.raises(ExpiredRefreshSessionError):
            resolve_refresh_session(session, created.raw_token)


def test_revoked_refresh_session_is_rejected_and_revocation_is_idempotent(sqlite_engine) -> None:
    with Session(sqlite_engine) as session:
        user = _create_user(session)
        created = create_refresh_session(session, user.id)

        assert revoke_refresh_session(session, created.raw_token) is True
        assert revoke_refresh_session(session, created.raw_token) is False
        with pytest.raises(RevokedRefreshSessionError):
            resolve_refresh_session(session, created.raw_token)


def test_successful_rotation_revokes_old_and_preserves_absolute_expiry(sqlite_engine) -> None:
    with Session(sqlite_engine) as session:
        user = _create_user(session)
        created = create_refresh_session(session, user.id)
        original_expiry = created.session.expires_at

        rotated = rotate_refresh_session(session, created.raw_token)
        original = _stored_session(session, created.session.id)
        replacement = _stored_session(session, rotated.session.id)

        assert rotated.raw_token != created.raw_token
        assert rotated.raw_token not in repr(rotated)
        assert rotated.user.id == user.id
        assert rotated.session.expires_at == original_expiry
        assert replacement.expires_at.replace(tzinfo=timezone.utc) == original_expiry
        assert original.revoked_at is not None
        assert original.replaced_by_session_id == replacement.id

        with pytest.raises(RevokedRefreshSessionError):
            resolve_refresh_session(session, created.raw_token)
        assert resolve_refresh_session(session, rotated.raw_token).session.id == replacement.id


def test_rotating_an_already_rotated_token_fails_and_creates_one_successor(sqlite_engine) -> None:
    with Session(sqlite_engine) as session:
        user = _create_user(session)
        created = create_refresh_session(session, user.id)
        rotate_refresh_session(session, created.raw_token)

        with pytest.raises(RevokedRefreshSessionError):
            rotate_refresh_session(session, created.raw_token)

        rows = session.execute(
            select(RefreshSession).where(RefreshSession.user_id == user.id)
        ).scalars().all()
        assert len(rows) == 2
        assert sum(row.revoked_at is None for row in rows) == 1


def test_failed_rotation_rolls_back_replacement_and_old_revocation(sqlite_engine, monkeypatch) -> None:
    with Session(sqlite_engine) as session:
        user = _create_user(session)
        user_id = user.id
        created = create_refresh_session(session, user_id)

        def fail_commit() -> None:
            raise SQLAlchemyError("forced rotation commit failure")

        monkeypatch.setattr(session, "commit", fail_commit)
        with pytest.raises(SQLAlchemyError, match="forced rotation commit failure"):
            rotate_refresh_session(session, created.raw_token)

    with Session(sqlite_engine) as verification_session:
        rows = verification_session.execute(
            select(RefreshSession).where(RefreshSession.user_id == user_id)
        ).scalars().all()
        assert len(rows) == 1
        assert rows[0].revoked_at is None
        assert rows[0].replaced_by_session_id is None
        assert resolve_refresh_session(verification_session, created.raw_token).user.id == user_id


def test_missing_user_is_rejected_during_creation(sqlite_engine) -> None:
    with Session(sqlite_engine) as session:
        with pytest.raises(RefreshSessionUserNotFoundError):
            create_refresh_session(session, "00000000-0000-0000-0000-000000000000")


def test_deleting_user_removes_all_refresh_sessions(sqlite_engine) -> None:
    with Session(sqlite_engine) as session:
        user = _create_user(session)
        user_id = user.id
        create_refresh_session(session, user_id)
        create_refresh_session(session, user_id)

        session.delete(user)
        session.commit()

        remaining = session.scalar(
            select(func.count()).select_from(RefreshSession).where(
                RefreshSession.user_id == user_id
            )
        )
        assert remaining == 0
