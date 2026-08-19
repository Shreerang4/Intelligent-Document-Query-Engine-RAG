from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.dialects import mysql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateTable

from persistence.db import Base
from persistence.models import RefreshSession, User


REPO_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_PATH = REPO_ROOT / "migrations" / "mysql" / "002_multi_user_auth.sql"
INIT_DB_PATH = REPO_ROOT / "scripts" / "init_db.py"


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


def _password_user(email: str) -> User:
    return User(
        email=email,
        display_name="Test User",
        auth_provider="password",
        password_hash="$argon2id$test-only-placeholder",
    )


def test_password_user_gets_uuid_id_and_auth_fields(sqlite_engine) -> None:
    with Session(sqlite_engine) as session:
        user = _password_user("user@example.com")
        session.add(user)
        session.commit()

        assert str(uuid.UUID(user.id)) == user.id
        assert user.auth_provider == "password"
        assert user.email == "user@example.com"
        assert user.password_hash == "$argon2id$test-only-placeholder"


def test_email_is_unique_but_nullable(sqlite_engine) -> None:
    with Session(sqlite_engine) as session:
        session.add_all(
            [
                _password_user("first@example.com"),
                User(id="transitional-user-1"),
                User(id="transitional-user-2"),
            ]
        )
        session.commit()

        duplicate = _password_user("first@example.com")
        session.add(duplicate)
        with pytest.raises(IntegrityError):
            session.commit()


def test_user_schema_has_no_separate_normalized_email_column() -> None:
    assert "email_normalized" not in User.__table__.c
    unique_constraint_names = {
        constraint.name
        for constraint in User.__table__.constraints
        if constraint.__class__.__name__ == "UniqueConstraint"
    }
    assert "uq_users_email" in unique_constraint_names


def test_refresh_session_supports_hash_uniqueness_and_rotation(sqlite_engine) -> None:
    token_hash = hashlib.sha256(b"first refresh credential").digest()
    replacement_hash = hashlib.sha256(b"replacement refresh credential").digest()
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)

    with Session(sqlite_engine) as session:
        user = _password_user("refresh@example.com")
        original = RefreshSession(
            user=user,
            token_hash=token_hash,
            expires_at=expires_at,
        )
        replacement = RefreshSession(
            user=user,
            token_hash=replacement_hash,
            expires_at=expires_at,
        )
        original.replaced_by = replacement
        original.revoked_at = datetime.now(timezone.utc)
        session.add_all([original, replacement])
        session.commit()

        assert str(uuid.UUID(original.id)) == original.id
        assert str(uuid.UUID(replacement.id)) == replacement.id
        assert original.user_id == user.id
        assert original.replaced_by_session_id == replacement.id
        assert len(original.token_hash) == hashlib.sha256().digest_size

        session.add(
            RefreshSession(
                user=user,
                token_hash=token_hash,
                expires_at=expires_at,
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()


def test_refresh_sessions_are_deleted_with_their_user(sqlite_engine) -> None:
    with Session(sqlite_engine) as session:
        user = _password_user("cascade@example.com")
        refresh_session = RefreshSession(
            user=user,
            token_hash=hashlib.sha256(b"cascade credential").digest(),
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        )
        session.add(refresh_session)
        session.commit()
        refresh_session_id = refresh_session.id

        session.delete(user)
        session.commit()

        assert session.get(RefreshSession, refresh_session_id) is None


def test_refresh_token_hash_compiles_as_fixed_width_mysql_binary() -> None:
    mysql_ddl = str(CreateTable(RefreshSession.__table__).compile(dialect=mysql.dialect()))

    assert "token_hash BINARY(32) NOT NULL" in mysql_ddl
    assert "CONSTRAINT uq_refresh_sessions_token_hash UNIQUE (token_hash)" in mysql_ddl


def test_mysql_auth_migration_is_idempotent_and_cleans_legacy_rows_in_fk_order() -> None:
    migration = MIGRATION_PATH.read_text(encoding="utf-8").lower()

    assert "information_schema.columns" in migration
    assert "information_schema.statistics" in migration
    assert "create table if not exists `refresh_sessions`" in migration
    assert "unique index `uq_users_email`" in migration
    assert "unique key `uq_refresh_sessions_token_hash`" in migration
    assert "`token_hash` binary(32) not null" in migration

    cleanup_markers = [
        "delete citation\n",
        "delete stored_query\n",
        "delete chunk_row\n",
        "delete from `documents`",
        "delete from `refresh_sessions`",
        "delete from `users`",
    ]
    cleanup_positions = [migration.index(marker) for marker in cleanup_markers]
    assert cleanup_positions == sorted(cleanup_positions)
    assert "where id = @legacy_user_id" in migration
    assert "insert into `users`" not in migration
    assert "email_normalized" not in migration


def test_database_initialization_does_not_seed_local_dev_user() -> None:
    init_db_source = INIT_DB_PATH.read_text(encoding="utf-8")

    assert "DEFAULT_USER_ID" not in init_db_source
    assert "_seed_default_user" not in init_db_source
    assert "local-dev-user" not in init_db_source
