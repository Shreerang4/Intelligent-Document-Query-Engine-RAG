from __future__ import annotations

import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import bindparam, create_engine, event, func, inspect, select, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, sessionmaker

from backend.app.auth.errors import RevokedRefreshSessionError
from backend.app.auth.refresh_sessions import (
    create_refresh_session,
    hash_refresh_token,
    rotate_refresh_session,
    rotate_refresh_session_in_transaction,
)
from backend.app.auth.router import router as auth_router
from persistence.db import Base, get_session
from persistence.models import RefreshSession, User


MYSQL_TEST_DATABASE_URL_ENV = "MYSQL_TEST_DATABASE_URL"
MYSQL_TEST_CA_CERT_ENV = "MYSQL_TEST_CA_CERT"
MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "migrations"
    / "mysql"
    / "002_multi_user_auth.sql"
)
PASSWORD = "phase-7a-integration-password"

pytestmark = pytest.mark.mysql_integration


def _disposable_mysql_engine() -> Engine:
    database_url = os.getenv(MYSQL_TEST_DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(
            f"Set {MYSQL_TEST_DATABASE_URL_ENV} to run destructive disposable-MySQL tests."
        )

    parsed = make_url(database_url)
    database_name = (parsed.database or "").lower()
    if not parsed.drivername.startswith("mysql"):
        pytest.skip(f"{MYSQL_TEST_DATABASE_URL_ENV} must use a MySQL driver.")
    if not any(marker in database_name for marker in ("test", "testing", "disposable")):
        pytest.skip(
            f"Refusing destructive integration tests because database name {database_name!r} "
            "does not identify a test/disposable database."
        )

    connect_args = {}
    ca_cert = os.getenv(MYSQL_TEST_CA_CERT_ENV)
    if ca_cert:
        connect_args = {"ssl": {"ca": ca_cert}}
    return create_engine(
        database_url,
        connect_args=connect_args,
        pool_size=5,
        max_overflow=2,
        pool_pre_ping=True,
        future=True,
    )


@pytest.fixture
def mysql_engine():
    engine = _disposable_mysql_engine()
    with engine.connect() as connection:
        assert connection.dialect.name == "mysql"
        assert connection.execute(text("SELECT 1")).scalar_one() == 1
    Base.metadata.drop_all(engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def _assert_innodb(engine: Engine, table_names: set[str]) -> None:
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                """
                SELECT table_name, engine
                FROM information_schema.tables
                WHERE table_schema = DATABASE()
                  AND table_name IN :table_names
                """
            ).bindparams(bindparam("table_names", expanding=True)),
            {"table_names": tuple(table_names)},
        ).all()
    engines = {str(table_name): str(storage_engine).upper() for table_name, storage_engine in rows}
    assert set(engines) == table_names
    assert set(engines.values()) == {"INNODB"}


def test_real_innodb_same_token_rotation_serializes_to_one_successor(mysql_engine) -> None:
    Base.metadata.create_all(mysql_engine)
    _assert_innodb(mysql_engine, {"users", "refresh_sessions"})
    factory = sessionmaker(bind=mysql_engine, autoflush=False, expire_on_commit=False, future=True)

    with factory() as session:
        user = User(
            email="mysql-race@example.com",
            auth_provider="password",
            password_hash="$argon2id$test-only-placeholder",
        )
        session.add(user)
        session.commit()
        user_id = user.id
        original = create_refresh_session(session, user_id)

    first_lock_acquired = threading.Event()
    second_select_started = threading.Event()
    release_first_rotation = threading.Event()
    thread_role = threading.local()
    connection_ids: dict[str, int] = {}

    @event.listens_for(mysql_engine, "before_cursor_execute")
    def observe_second_attempt(_connection, _cursor, statement, _parameters, _context, _many):
        if (
            getattr(thread_role, "value", None) == "second"
            and "FOR UPDATE" in statement.upper()
            and "refresh_sessions" in statement
        ):
            second_select_started.set()

    @event.listens_for(mysql_engine, "after_cursor_execute")
    def hold_first_lock(_connection, _cursor, statement, _parameters, _context, _many):
        if (
            getattr(thread_role, "value", None) == "first"
            and "FOR UPDATE" in statement.upper()
            and "refresh_sessions" in statement
        ):
            first_lock_acquired.set()
            assert release_first_rotation.wait(timeout=10)

    def rotate(role: str):
        thread_role.value = role
        with factory() as session:
            connection_ids[role] = session.execute(text("SELECT CONNECTION_ID()")).scalar_one()
            try:
                return ("success", rotate_refresh_session(session, original.raw_token))
            except RevokedRefreshSessionError as exc:
                return ("revoked", exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(rotate, "first")
        assert first_lock_acquired.wait(timeout=10)
        second_future = executor.submit(rotate, "second")
        assert second_select_started.wait(timeout=10)
        assert not second_future.done(), "Second rotation must wait on R1's InnoDB row lock."
        release_first_rotation.set()
        outcomes = [first_future.result(timeout=10), second_future.result(timeout=10)]

    assert sorted(outcome for outcome, _result in outcomes) == ["revoked", "success"]
    assert connection_ids["first"] != connection_ids["second"]
    successful_rotation = next(result for outcome, result in outcomes if outcome == "success")

    with factory() as session:
        rows = session.execute(
            select(RefreshSession)
            .where(RefreshSession.user_id == user_id)
            .order_by(RefreshSession.created_at.asc())
        ).scalars().all()
        stored_original = session.get(RefreshSession, original.session.id)
        assert stored_original is not None
        assert len(rows) == 2
        assert stored_original.revoked_at is not None
        assert stored_original.replaced_by_session_id == successful_rotation.session.id
        assert {row.id for row in rows if row.id != stored_original.id} == {
            successful_rotation.session.id
        }
        assert sum(row.revoked_at is None for row in rows) == 1
        assert session.execute(text("SELECT 1")).scalar_one() == 1


def test_real_mysql_rotation_staging_can_roll_back_without_partial_successor(mysql_engine) -> None:
    Base.metadata.create_all(mysql_engine)
    factory = sessionmaker(bind=mysql_engine, autoflush=False, expire_on_commit=False, future=True)
    with factory() as session:
        user = User(
            email="mysql-rollback@example.com",
            auth_provider="password",
            password_hash="$argon2id$test-only-placeholder",
        )
        session.add(user)
        session.commit()
        user_id = user.id
        original = create_refresh_session(session, user_id)

    with factory() as session:
        rotate_refresh_session_in_transaction(session, original.raw_token)
        session.rollback()

    with factory() as session:
        rows = session.execute(
            select(RefreshSession).where(RefreshSession.user_id == user_id)
        ).scalars().all()
        assert len(rows) == 1
        assert rows[0].id == original.session.id
        assert rows[0].revoked_at is None
        assert rows[0].replaced_by_session_id is None
        assert session.execute(text("SELECT 1")).scalar_one() == 1


def _execute_migration_script(engine: Engine) -> None:
    statements: list[str] = []
    buffer: list[str] = []
    for line in MIGRATION_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        buffer.append(line)
        if stripped.endswith(";"):
            statements.append("\n".join(buffer))
            buffer = []
    assert not buffer

    with engine.connect() as connection:
        for statement in statements:
            connection.exec_driver_sql(statement)


def _create_representative_pre_auth_state(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE `refresh_sessions`")
        connection.exec_driver_sql("ALTER TABLE `users` DROP INDEX `uq_users_email`")
        connection.exec_driver_sql("ALTER TABLE `users` DROP COLUMN `password_hash`")
        connection.execute(
            text(
                """
                INSERT INTO users (id, email, display_name, auth_provider, created_at, updated_at)
                VALUES
                    ('local-dev-user', NULL, 'Legacy User', 'local', NOW(6), NOW(6)),
                    ('surviving-user', NULL, 'Surviving User', 'external', NOW(6), NOW(6))
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO documents (
                    id, user_id, source_type, filename, source_hash, status,
                    embedding_model, embedding_format, retrieval_mode,
                    reranker_model, k_initial, k_final, created_at, updated_at
                ) VALUES (
                    'legacy-document', 'local-dev-user', 'upload', 'legacy.pdf',
                    'legacy-hash', 'ingested', 'e5', 'format', 'faiss',
                    'reranker', 20, 8, NOW(6), NOW(6)
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO chunks (
                    id, user_id, document_id, chunk_id, chunk_index, page_number,
                    text, created_at
                ) VALUES (
                    'legacy-chunk', 'local-dev-user', 'legacy-document', 0, 0, 1,
                    'legacy chunk', NOW(6)
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO queries (
                    id, user_id, document_id, question, answer, status,
                    is_abstained, embedding_model, retrieval_mode, reranker_model,
                    k_initial, k_final, created_at
                ) VALUES (
                    'legacy-query', 'local-dev-user', 'legacy-document', 'Question?',
                    'Answer.', 'ok', 0, 'e5', 'faiss', 'reranker', 20, 8, NOW(6)
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO citations (
                    id, user_id, query_id, document_id, chunk_db_id, chunk_id,
                    `rank`, page_number, excerpt, created_at
                ) VALUES (
                    'legacy-citation', 'local-dev-user', 'legacy-query',
                    'legacy-document', 'legacy-chunk', 0, 1, 1, 'legacy excerpt', NOW(6)
                )
                """
            )
        )


def test_migration_002_rehearsal_on_disposable_mysql(mysql_engine, monkeypatch) -> None:
    _create_representative_pre_auth_state(mysql_engine)
    _execute_migration_script(mysql_engine)
    _execute_migration_script(mysql_engine)

    inspector = inspect(mysql_engine)
    user_columns = {column["name"] for column in inspector.get_columns("users")}
    refresh_columns = {column["name"] for column in inspector.get_columns("refresh_sessions")}
    user_unique_constraints = {
        constraint["name"] for constraint in inspector.get_unique_constraints("users")
    }
    refresh_unique_constraints = {
        constraint["name"]
        for constraint in inspector.get_unique_constraints("refresh_sessions")
    }
    refresh_indexes = {index["name"] for index in inspector.get_indexes("refresh_sessions")}
    refresh_foreign_keys = {
        foreign_key["name"]: foreign_key for foreign_key in inspector.get_foreign_keys("refresh_sessions")
    }

    assert "password_hash" in user_columns
    assert "uq_users_email" in user_unique_constraints
    assert refresh_columns == {
        "id",
        "user_id",
        "token_hash",
        "expires_at",
        "revoked_at",
        "created_at",
        "replaced_by_session_id",
    }
    assert "uq_refresh_sessions_token_hash" in refresh_unique_constraints
    assert {
        "ix_refresh_sessions_expires_at",
        "ix_refresh_sessions_user_id_revoked_at_expires_at",
    }.issubset(refresh_indexes)
    assert set(refresh_foreign_keys) == {
        "fk_refresh_sessions_user_id_users",
        "fk_refresh_sessions_replaced_by_session_id",
    }
    assert refresh_foreign_keys["fk_refresh_sessions_user_id_users"]["options"]["ondelete"] == "CASCADE"
    assert (
        refresh_foreign_keys["fk_refresh_sessions_replaced_by_session_id"]["options"]["ondelete"]
        == "SET NULL"
    )
    _assert_innodb(
        mysql_engine,
        {"users", "documents", "chunks", "queries", "citations", "refresh_sessions"},
    )

    with Session(mysql_engine) as session:
        assert session.scalar(
            select(func.count()).select_from(User).where(User.id == "local-dev-user")
        ) == 0
        assert session.get(User, "surviving-user") is not None

    monkeypatch.setenv("ACCESS_JWT_SECRET", "phase-7a-mysql-secret-not-used-in-production")
    monkeypatch.setenv("REFRESH_COOKIE_SECURE", "false")
    factory = sessionmaker(bind=mysql_engine, autoflush=False, expire_on_commit=False, future=True)
    application = FastAPI()
    application.include_router(auth_router)

    def override_get_session():
        with factory() as session:
            yield session

    application.dependency_overrides[get_session] = override_get_session
    with TestClient(application) as client:
        registration = client.post(
            "/auth/register",
            json={
                "email": "After.Migration@example.com",
                "password": PASSWORD,
                "display_name": "Migrated User",
            },
        )
        login = client.post(
            "/auth/login",
            json={"email": "after.migration@example.com", "password": PASSWORD},
        )
        registration_refresh_token = registration.cookies["idqe_refresh"]
        login_refresh_token = login.cookies["idqe_refresh"]

    assert registration.status_code == 201
    assert login.status_code == 200
    assert registration.json()["user"]["email"] == "After.Migration@example.com"
    assert login.json()["user"]["id"] == registration.json()["user"]["id"]
    assert registration.json()["access_token"]
    assert login.json()["access_token"]
    assert registration.json()["access_token"] != login.json()["access_token"]
    assert registration_refresh_token != login_refresh_token

    registered_user_id = registration.json()["user"]["id"]
    with Session(mysql_engine) as session:
        registered_user = session.get(User, registered_user_id)
        assert registered_user is not None
        assert registered_user.password_hash is not None
        assert registered_user.password_hash.startswith("$argon2id$")
        assert registered_user.password_hash != PASSWORD

        persisted_refresh_sessions = session.scalars(
            select(RefreshSession).where(RefreshSession.user_id == registered_user_id)
        ).all()
        assert len(persisted_refresh_sessions) == 2
        assert all(row.revoked_at is None for row in persisted_refresh_sessions)
        assert {row.token_hash for row in persisted_refresh_sessions} == {
            hash_refresh_token(registration_refresh_token),
            hash_refresh_token(login_refresh_token),
        }
        assert all(len(row.token_hash) == 32 for row in persisted_refresh_sessions)
        assert all(
            row.token_hash
            not in {
                registration_refresh_token.encode("utf-8"),
                login_refresh_token.encode("utf-8"),
            }
            for row in persisted_refresh_sessions
        )
