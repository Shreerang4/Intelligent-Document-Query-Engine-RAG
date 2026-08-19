"""Lazy configuration for short-lived access JWTs."""

import os
from urllib.parse import urlsplit

from backend.app.auth.errors import AuthConfigurationError


ACCESS_JWT_SECRET_ENV = "ACCESS_JWT_SECRET"
ACCESS_TOKEN_TTL_SECONDS_ENV = "ACCESS_TOKEN_TTL_SECONDS"
DEFAULT_ACCESS_TOKEN_TTL_SECONDS = 600
MIN_ACCESS_JWT_SECRET_BYTES = 32
ACCESS_JWT_ALGORITHM = "HS256"
ACCESS_JWT_ISSUER = "intelligent-document-query-engine"
ACCESS_JWT_AUDIENCE = "intelligent-document-query-engine-api"
REFRESH_COOKIE_SECURE_ENV = "REFRESH_COOKIE_SECURE"
AUTH_ALLOWED_ORIGINS_ENV = "AUTH_ALLOWED_ORIGINS"
DEFAULT_REFRESH_COOKIE_SECURE = True
LOCAL_DEVELOPMENT_ORIGINS = frozenset(
    {
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    }
)


def get_access_jwt_secret() -> str:
    secret = os.getenv(ACCESS_JWT_SECRET_ENV)
    if secret is None or not secret.strip():
        raise AuthConfigurationError(
            "ACCESS_JWT_SECRET must be configured before access-token functionality is used."
        )
    if len(secret.encode("utf-8")) < MIN_ACCESS_JWT_SECRET_BYTES:
        raise AuthConfigurationError("ACCESS_JWT_SECRET must contain at least 32 bytes.")
    return secret


def get_access_token_ttl_seconds() -> int:
    raw_value = os.getenv(
        ACCESS_TOKEN_TTL_SECONDS_ENV,
        str(DEFAULT_ACCESS_TOKEN_TTL_SECONDS),
    )
    try:
        ttl_seconds = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise AuthConfigurationError("ACCESS_TOKEN_TTL_SECONDS must be a positive integer.") from exc
    if ttl_seconds <= 0:
        raise AuthConfigurationError("ACCESS_TOKEN_TTL_SECONDS must be a positive integer.")
    return ttl_seconds


def get_refresh_cookie_secure() -> bool:
    """Return the explicitly configured cookie security mode; secure by default."""
    raw_value = os.getenv(REFRESH_COOKIE_SECURE_ENV)
    if raw_value is None:
        return DEFAULT_REFRESH_COOKIE_SECURE
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise AuthConfigurationError(
        "REFRESH_COOKIE_SECURE must be a boolean value such as true or false."
    )


def _normalized_origin(value: str) -> str:
    candidate = value.strip().rstrip("/")
    parsed = urlsplit(candidate)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise AuthConfigurationError("AUTH_ALLOWED_ORIGINS contains an invalid origin.")
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def get_auth_allowed_origins() -> frozenset[str]:
    """Return explicit browser origins plus the supported local development origins."""
    configured = os.getenv(AUTH_ALLOWED_ORIGINS_ENV, "")
    extra_origins = {
        _normalized_origin(item)
        for item in configured.split(",")
        if item.strip()
    }
    return LOCAL_DEVELOPMENT_ORIGINS | extra_origins
