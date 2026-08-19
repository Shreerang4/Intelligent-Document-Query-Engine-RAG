"""Opaque refresh-token and refresh-session service logic.

This module contains no cookie or HTTP behavior. Raw tokens are returned only to
the caller and are never assigned to an ORM model or log record.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.auth.errors import (
    ExpiredRefreshSessionError,
    InvalidRefreshTokenError,
    RefreshSessionError,
    RefreshSessionUserNotFoundError,
    RevokedRefreshSessionError,
)
from persistence.models import RefreshSession, User


REFRESH_TOKEN_RANDOM_BYTES = 32
REFRESH_SESSION_LIFETIME = timedelta(days=7)


@dataclass(frozen=True)
class RefreshSessionMetadata:
    id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    revoked_at: datetime | None
    replaced_by_session_id: str | None


@dataclass(frozen=True)
class CreatedRefreshSession:
    raw_token: str = field(repr=False)
    session: RefreshSessionMetadata


@dataclass(frozen=True)
class ResolvedRefreshSession:
    user: User
    session: RefreshSessionMetadata


@dataclass(frozen=True)
class RotatedRefreshSession:
    raw_token: str = field(repr=False)
    user: User
    session: RefreshSessionMetadata


def generate_refresh_token() -> str:
    """Generate a URL-safe opaque credential with 256 bits of entropy."""
    return secrets.token_urlsafe(REFRESH_TOKEN_RANDOM_BYTES)


def hash_refresh_token(raw_token: str) -> bytes:
    """Return the canonical SHA-256 digest used for every DB operation."""
    if not isinstance(raw_token, str) or not raw_token:
        raise InvalidRefreshTokenError("Invalid refresh token.")
    return hashlib.sha256(raw_token.encode("utf-8")).digest()


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _metadata(refresh_session: RefreshSession) -> RefreshSessionMetadata:
    return RefreshSessionMetadata(
        id=refresh_session.id,
        user_id=refresh_session.user_id,
        created_at=_utc_datetime(refresh_session.created_at),
        expires_at=_utc_datetime(refresh_session.expires_at),
        revoked_at=(
            _utc_datetime(refresh_session.revoked_at)
            if refresh_session.revoked_at is not None
            else None
        ),
        replaced_by_session_id=refresh_session.replaced_by_session_id,
    )


def _find_refresh_session(
    session: Session,
    raw_token: str,
    *,
    for_update: bool,
) -> RefreshSession | None:
    statement = select(RefreshSession).where(
        RefreshSession.token_hash == hash_refresh_token(raw_token)
    )
    if for_update:
        statement = statement.with_for_update()
    return session.execute(statement).scalar_one_or_none()


def _require_active_refresh_session(
    refresh_session: RefreshSession | None,
    *,
    now: datetime,
) -> RefreshSession:
    if refresh_session is None:
        raise InvalidRefreshTokenError("Invalid refresh token.")
    if _utc_datetime(refresh_session.expires_at) <= now:
        raise ExpiredRefreshSessionError("Refresh session has expired.")
    if refresh_session.revoked_at is not None:
        raise RevokedRefreshSessionError("Refresh session has been revoked or rotated.")
    return refresh_session


def _require_refresh_user(session: Session, user_id: str) -> User:
    user = session.get(User, user_id)
    if user is None:
        raise RefreshSessionUserNotFoundError("Refresh session user no longer exists.")
    return user


def create_refresh_session(session: Session, user_id: str) -> CreatedRefreshSession:
    """Create and commit one independent seven-day refresh session."""
    try:
        created = add_refresh_session(session, user_id)
        session.commit()
    except Exception:
        session.rollback()
        raise
    return created


def add_refresh_session(session: Session, user_id: str) -> CreatedRefreshSession:
    """Add and flush a refresh session while leaving commit ownership to the caller."""
    user = session.get(User, user_id)
    if user is None:
        raise RefreshSessionUserNotFoundError("Refresh session user does not exist.")

    raw_token = generate_refresh_token()
    now = datetime.now(timezone.utc)
    refresh_session = RefreshSession(
        id=str(uuid.uuid4()),
        user_id=user.id,
        token_hash=hash_refresh_token(raw_token),
        created_at=now,
        expires_at=now + REFRESH_SESSION_LIFETIME,
        revoked_at=None,
        replaced_by_session_id=None,
    )
    session.add(refresh_session)
    session.flush()
    return CreatedRefreshSession(raw_token=raw_token, session=_metadata(refresh_session))


def resolve_refresh_session(session: Session, raw_token: str) -> ResolvedRefreshSession:
    """Resolve one active raw refresh credential to its session and user."""
    now = datetime.now(timezone.utc)
    refresh_session = _require_active_refresh_session(
        _find_refresh_session(session, raw_token, for_update=False),
        now=now,
    )
    user = _require_refresh_user(session, refresh_session.user_id)
    return ResolvedRefreshSession(user=user, session=_metadata(refresh_session))


def rotate_refresh_session(session: Session, raw_token: str) -> RotatedRefreshSession:
    """Atomically revoke one credential and create its absolute-expiry successor."""
    try:
        rotated = rotate_refresh_session_in_transaction(session, raw_token)
        session.commit()
    except RefreshSessionError:
        session.rollback()
        raise
    except Exception:
        session.rollback()
        raise

    return rotated


def rotate_refresh_session_in_transaction(
    session: Session,
    raw_token: str,
) -> RotatedRefreshSession:
    """Stage a locked rotation while leaving commit ownership to the caller."""
    now = datetime.now(timezone.utc)
    original = _require_active_refresh_session(
        _find_refresh_session(session, raw_token, for_update=True),
        now=now,
    )
    user = _require_refresh_user(session, original.user_id)

    replacement_raw_token = generate_refresh_token()
    replacement = RefreshSession(
        id=str(uuid.uuid4()),
        user_id=original.user_id,
        token_hash=hash_refresh_token(replacement_raw_token),
        created_at=now,
        expires_at=original.expires_at,
        revoked_at=None,
        replaced_by_session_id=None,
    )
    session.add(replacement)
    session.flush()

    original.revoked_at = now
    original.replaced_by_session_id = replacement.id
    session.flush()
    return RotatedRefreshSession(
        raw_token=replacement_raw_token,
        user=user,
        session=_metadata(replacement),
    )


def revoke_refresh_session(session: Session, raw_token: str) -> bool:
    """Idempotently revoke one active refresh session by its raw credential."""
    try:
        now = datetime.now(timezone.utc)
        refresh_session = _find_refresh_session(session, raw_token, for_update=True)
        if (
            refresh_session is None
            or refresh_session.revoked_at is not None
            or _utc_datetime(refresh_session.expires_at) <= now
        ):
            session.rollback()
            return False

        refresh_session.revoked_at = now
        session.commit()
        return True
    except InvalidRefreshTokenError:
        session.rollback()
        raise
    except Exception:
        session.rollback()
        raise
