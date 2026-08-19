"""Password, access-token, refresh-session, and HTTP authentication support."""

from backend.app.auth.dependencies import get_authenticated_user_id, get_current_user
from backend.app.auth.email import normalize_email
from backend.app.auth.errors import (
    AuthConfigurationError,
    DuplicateUserError,
    ExpiredRefreshSessionError,
    InvalidAccessTokenError,
    InvalidCredentialsError,
    InvalidEmailError,
    InvalidRefreshTokenError,
    RefreshSessionUserNotFoundError,
    RevokedRefreshSessionError,
)
from backend.app.auth.passwords import hash_password, password_needs_rehash, verify_password
from backend.app.auth.refresh_sessions import (
    CreatedRefreshSession,
    RefreshSessionMetadata,
    ResolvedRefreshSession,
    RotatedRefreshSession,
    add_refresh_session,
    create_refresh_session,
    generate_refresh_token,
    hash_refresh_token,
    resolve_refresh_session,
    revoke_refresh_session,
    rotate_refresh_session,
    rotate_refresh_session_in_transaction,
)
from backend.app.auth.services import (
    add_password_user,
    authenticate_user,
    register_user,
    verify_user_credentials,
)
from backend.app.auth.tokens import create_access_token, decode_access_token

__all__ = [
    "AuthConfigurationError",
    "DuplicateUserError",
    "InvalidAccessTokenError",
    "InvalidCredentialsError",
    "InvalidEmailError",
    "InvalidRefreshTokenError",
    "ExpiredRefreshSessionError",
    "RefreshSessionUserNotFoundError",
    "RevokedRefreshSessionError",
    "CreatedRefreshSession",
    "RefreshSessionMetadata",
    "ResolvedRefreshSession",
    "RotatedRefreshSession",
    "authenticate_user",
    "add_password_user",
    "add_refresh_session",
    "create_access_token",
    "decode_access_token",
    "get_authenticated_user_id",
    "get_current_user",
    "create_refresh_session",
    "generate_refresh_token",
    "hash_refresh_token",
    "hash_password",
    "normalize_email",
    "password_needs_rehash",
    "register_user",
    "resolve_refresh_session",
    "revoke_refresh_session",
    "rotate_refresh_session",
    "rotate_refresh_session_in_transaction",
    "verify_user_credentials",
    "verify_password",
]
