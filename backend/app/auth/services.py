"""Password registration and login credential services."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.auth.email import normalize_email
from backend.app.auth.errors import DuplicateUserError, InvalidCredentialsError
from backend.app.auth.passwords import (
    hash_password,
    password_needs_rehash,
    verify_password,
)
from persistence.models import User


INVALID_CREDENTIALS_MESSAGE = "Invalid email or password."
_DUMMY_PASSWORD_HASH = hash_password("dummy-password-for-unknown-users")


def _find_user_by_email(session: Session, normalized_email: str) -> Optional[User]:
    return session.execute(
        select(User).where(User.email == normalized_email)
    ).scalar_one_or_none()


def _is_email_uniqueness_conflict(exc: IntegrityError) -> bool:
    error_text = str(exc.orig).lower()
    return "uq_users_email" in error_text or "users.email" in error_text


def register_user(
    session: Session,
    *,
    email: str,
    password: str,
    display_name: Optional[str] = None,
) -> User:
    """Create and commit one password-authenticated user."""
    try:
        user = add_password_user(
            session,
            email=email,
            password=password,
            display_name=display_name,
        )
        session.commit()
    except DuplicateUserError:
        session.rollback()
        raise
    except SQLAlchemyError:
        session.rollback()
        raise

    session.refresh(user)
    return user


def add_password_user(
    session: Session,
    *,
    email: str,
    password: str,
    display_name: Optional[str] = None,
) -> User:
    """Add and flush a password user while leaving commit ownership to the caller."""
    normalized_email = normalize_email(email)
    if _find_user_by_email(session, normalized_email) is not None:
        raise DuplicateUserError("An account with this email already exists.")

    user = User(
        email=normalized_email,
        display_name=display_name,
        auth_provider="password",
        password_hash=hash_password(password),
    )
    session.add(user)
    try:
        session.flush()
    except IntegrityError as exc:
        if _is_email_uniqueness_conflict(exc):
            raise DuplicateUserError("An account with this email already exists.") from exc
        raise
    return user


def authenticate_user(session: Session, *, email: str, password: str) -> User:
    """Verify credentials, opportunistically rehash, and return the user."""
    user, rehashed = verify_user_credentials(session, email=email, password=password)
    if rehashed:
        try:
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            raise
        session.refresh(user)
    return user


def verify_user_credentials(
    session: Session,
    *,
    email: str,
    password: str,
) -> tuple[User, bool]:
    """Verify credentials and stage a needed rehash without committing it."""
    normalized_email = normalize_email(email)
    user = _find_user_by_email(session, normalized_email)

    stored_hash = (
        user.password_hash
        if user is not None and user.auth_provider == "password" and user.password_hash
        else _DUMMY_PASSWORD_HASH
    )
    password_matches = verify_password(stored_hash, password)
    if user is None or user.auth_provider != "password" or not user.password_hash or not password_matches:
        raise InvalidCredentialsError(INVALID_CREDENTIALS_MESSAGE)

    rehashed = False
    if password_needs_rehash(user.password_hash):
        user.password_hash = hash_password(password)
        rehashed = True

    return user, rehashed
