from __future__ import annotations

import pytest
from argon2 import PasswordHasher, Type, extract_parameters
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

import backend.app.auth.services as credential_services
from backend.app.auth.email import normalize_email
from backend.app.auth.errors import DuplicateUserError, InvalidCredentialsError, InvalidEmailError
from backend.app.auth.passwords import (
    ARGON2_MEMORY_COST_KIB,
    ARGON2_PARALLELISM,
    ARGON2_TIME_COST,
    hash_password,
    password_needs_rehash,
    verify_password,
)
from backend.app.auth.services import authenticate_user, register_user
from backend.app.schemas.auth import PublicUser
from persistence.db import Base
from persistence.models import User


@pytest.fixture
def sqlite_engine():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_valid_email_is_normalized_without_deliverability_lookup() -> None:
    assert normalize_email("person@EXAMPLE.COM") == "person@example.com"


@pytest.mark.parametrize("raw_email", ["not-an-email", "missing-domain@", "@missing-local.example"])
def test_invalid_email_is_rejected(raw_email: str) -> None:
    with pytest.raises(InvalidEmailError, match="Invalid email address"):
        normalize_email(raw_email)


def test_registration_stores_normalized_email_directly(sqlite_engine) -> None:
    with Session(sqlite_engine) as session:
        user = register_user(
            session,
            email="person@EXAMPLE.COM",
            password="correct horse battery staple",
            display_name="Person",
        )

        assert user.email == "person@example.com"
        assert user.auth_provider == "password"
        assert user.display_name == "Person"
        assert user.password_hash is not None
        assert user.password_hash.startswith("$argon2id$")
        assert not hasattr(user, "email_normalized")


def test_duplicate_registration_after_normalization(sqlite_engine) -> None:
    with Session(sqlite_engine) as session:
        register_user(session, email="person@EXAMPLE.COM", password="first password")

        with pytest.raises(DuplicateUserError, match="already exists"):
            register_user(session, email="person@example.com", password="second password")


def test_database_duplicate_race_is_converted_to_service_error(sqlite_engine, monkeypatch) -> None:
    with Session(sqlite_engine) as session:
        register_user(session, email="race@example.com", password="first password")
        monkeypatch.setattr(credential_services, "_find_user_by_email", lambda *_args: None)

        with pytest.raises(DuplicateUserError, match="already exists"):
            register_user(session, email="race@example.com", password="second password")


def test_argon2id_hashing_verification_parameters_and_random_salts() -> None:
    first_hash = hash_password("same password")
    second_hash = hash_password("same password")
    parameters = extract_parameters(first_hash)

    assert first_hash != second_hash
    assert parameters.type is Type.ID
    assert parameters.memory_cost == ARGON2_MEMORY_COST_KIB
    assert parameters.time_cost == ARGON2_TIME_COST
    assert parameters.parallelism == ARGON2_PARALLELISM
    assert verify_password(first_hash, "same password") is True
    assert verify_password(first_hash, "incorrect password") is False


def test_needs_rehash_detects_old_parameters_and_login_persists_upgrade(sqlite_engine) -> None:
    old_hasher = PasswordHasher(
        memory_cost=8_192,
        time_cost=1,
        parallelism=1,
        type=Type.ID,
    )
    old_hash = old_hasher.hash("rehash password")
    assert password_needs_rehash(old_hash) is True

    with Session(sqlite_engine) as session:
        user = User(
            email="rehash@example.com",
            auth_provider="password",
            password_hash=old_hash,
        )
        session.add(user)
        session.commit()
        user_id = user.id

        authenticated = authenticate_user(
            session,
            email="rehash@EXAMPLE.COM",
            password="rehash password",
        )
        upgraded_hash = authenticated.password_hash

    assert upgraded_hash is not None
    assert upgraded_hash != old_hash
    assert password_needs_rehash(upgraded_hash) is False

    with Session(sqlite_engine) as verification_session:
        persisted_user = verification_session.get(User, user_id)
        assert persisted_user is not None
        assert persisted_user.password_hash == upgraded_hash


def test_incorrect_password_and_unknown_user_share_generic_failure(sqlite_engine) -> None:
    with Session(sqlite_engine) as session:
        register_user(session, email="known@example.com", password="correct password")

        with pytest.raises(InvalidCredentialsError) as incorrect_error:
            authenticate_user(session, email="known@example.com", password="wrong password")
        with pytest.raises(InvalidCredentialsError) as unknown_error:
            authenticate_user(session, email="unknown@example.com", password="wrong password")

        assert str(incorrect_error.value) == str(unknown_error.value) == "Invalid email or password."


def test_unknown_user_login_never_creates_a_user(sqlite_engine) -> None:
    with Session(sqlite_engine) as session:
        before = session.scalar(select(func.count()).select_from(User))

        with pytest.raises(InvalidCredentialsError):
            authenticate_user(session, email="absent@example.com", password="any password")

        after = session.scalar(select(func.count()).select_from(User))
        assert before == after == 0


def test_public_user_representation_excludes_password_hash(sqlite_engine) -> None:
    with Session(sqlite_engine) as session:
        user = register_user(session, email="public@example.com", password="private password")
        public_user = PublicUser.model_validate(user)
        payload = public_user.model_dump(mode="json")

        assert set(payload) == {"id", "email", "display_name", "auth_provider", "created_at"}
        assert payload["id"] == user.id
        assert payload["email"] == "public@example.com"
        assert payload["display_name"] is None
        assert payload["auth_provider"] == "password"
        assert "password_hash" not in payload
