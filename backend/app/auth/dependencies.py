"""FastAPI dependencies for access-JWT-authenticated users."""

from typing import Annotated, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.app.auth.errors import InvalidAccessTokenError
from backend.app.auth.tokens import decode_access_token
from persistence.db import get_session
from persistence.models import User


_bearer_scheme = HTTPBearer(auto_error=False)
_UNAUTHORIZED_HEADERS = {"WWW-Authenticate": "Bearer"}


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing access token.",
        headers=_UNAUTHORIZED_HEADERS,
    )


def get_authenticated_user_id(
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials],
        Depends(_bearer_scheme),
    ],
) -> str:
    """Validate an access JWT and return its canonical user UUID without I/O."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized()

    try:
        payload = decode_access_token(credentials.credentials)
    except InvalidAccessTokenError as exc:
        raise _unauthorized() from exc

    return payload["sub"]


def get_current_user(
    user_id: Annotated[str, Depends(get_authenticated_user_id)],
    session: Annotated[Session, Depends(get_session)],
) -> User:
    """Resolve an authenticated user UUID to fresh database-backed state."""

    user = session.get(User, user_id)
    if user is None:
        raise _unauthorized()
    return user
