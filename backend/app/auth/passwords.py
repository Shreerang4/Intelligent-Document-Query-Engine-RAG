"""Argon2id password hashing primitives."""

from argon2 import PasswordHasher, Type
from argon2.exceptions import InvalidHashError, VerificationError


ARGON2_MEMORY_COST_KIB = 19_456
ARGON2_TIME_COST = 2
ARGON2_PARALLELISM = 1

_PASSWORD_HASHER = PasswordHasher(
    memory_cost=ARGON2_MEMORY_COST_KIB,
    time_cost=ARGON2_TIME_COST,
    parallelism=ARGON2_PARALLELISM,
    type=Type.ID,
)


def hash_password(password: str) -> str:
    """Hash a password with Argon2id and a library-generated random salt."""
    if not isinstance(password, str):
        raise TypeError("Password must be a string.")
    return _PASSWORD_HASHER.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """Return False for a mismatch or an invalid encoded hash."""
    try:
        return _PASSWORD_HASHER.verify(password_hash, password)
    except (VerificationError, InvalidHashError, TypeError):
        return False


def password_needs_rehash(password_hash: str) -> bool:
    """Return whether a valid encoded hash uses obsolete parameters."""
    try:
        return _PASSWORD_HASHER.check_needs_rehash(password_hash)
    except (InvalidHashError, TypeError):
        return True
