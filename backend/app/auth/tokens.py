"""Creation and strict validation of short-lived access JWTs."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import jwt
from jwt import PyJWTError

from backend.app.auth.config import (
    ACCESS_JWT_ALGORITHM,
    ACCESS_JWT_AUDIENCE,
    ACCESS_JWT_ISSUER,
    get_access_jwt_secret,
    get_access_token_ttl_seconds,
)
from backend.app.auth.errors import InvalidAccessTokenError


REQUIRED_ACCESS_TOKEN_CLAIMS = ("sub", "iat", "exp", "jti", "iss", "aud")


def _canonical_uuid(value: Any, *, claim_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise InvalidAccessTokenError(f"Access token has an invalid {claim_name} claim.")
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError) as exc:
        raise InvalidAccessTokenError(f"Access token has an invalid {claim_name} claim.") from exc


def create_access_token(user_id: str) -> str:
    """Create a signed HS256 access JWT for one canonical UUID user ID."""
    canonical_user_id = _canonical_uuid(user_id, claim_name="sub")
    issued_at = int(datetime.now(timezone.utc).timestamp())
    payload = {
        "sub": canonical_user_id,
        "iat": issued_at,
        "exp": issued_at + get_access_token_ttl_seconds(),
        "jti": str(uuid.uuid4()),
        "iss": ACCESS_JWT_ISSUER,
        "aud": ACCESS_JWT_AUDIENCE,
    }
    return jwt.encode(
        payload,
        get_access_jwt_secret(),
        algorithm=ACCESS_JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> dict[str, Any]:
    """Validate an access JWT and return its minimal trusted claims."""
    if not isinstance(token, str) or not token.strip():
        raise InvalidAccessTokenError("Invalid access token.")

    try:
        payload = jwt.decode(
            token,
            get_access_jwt_secret(),
            algorithms=[ACCESS_JWT_ALGORITHM],
            audience=ACCESS_JWT_AUDIENCE,
            issuer=ACCESS_JWT_ISSUER,
            options={
                "require": list(REQUIRED_ACCESS_TOKEN_CLAIMS),
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
                "verify_iss": True,
                "verify_aud": True,
            },
        )
        payload["sub"] = _canonical_uuid(payload.get("sub"), claim_name="sub")
        payload["jti"] = _canonical_uuid(payload.get("jti"), claim_name="jti")
        return payload
    except InvalidAccessTokenError:
        raise
    except (PyJWTError, TypeError, ValueError) as exc:
        raise InvalidAccessTokenError("Invalid access token.") from exc
