"""Service-layer errors for password credentials."""


class AuthServiceError(Exception):
    """Base class for expected credential-service failures."""


class InvalidEmailError(AuthServiceError):
    """Raised when an email address cannot be validated."""


class DuplicateUserError(AuthServiceError):
    """Raised when a normalized email is already registered."""


class InvalidCredentialsError(AuthServiceError):
    """Raised for both unknown users and incorrect passwords."""


class AuthConfigurationError(AuthServiceError):
    """Raised when required authentication configuration is unavailable."""


class InvalidAccessTokenError(AuthServiceError):
    """Raised when an access token cannot be trusted."""


class RefreshSessionError(AuthServiceError):
    """Base class for expected opaque refresh-session failures."""


class InvalidRefreshTokenError(RefreshSessionError):
    """Raised when a raw refresh token is malformed or unknown."""


class ExpiredRefreshSessionError(RefreshSessionError):
    """Raised when a refresh session has passed its absolute expiry."""


class RevokedRefreshSessionError(RefreshSessionError):
    """Raised when a refresh session was revoked or rotated."""


class RefreshSessionUserNotFoundError(RefreshSessionError):
    """Raised when a refresh session has no associated user."""
