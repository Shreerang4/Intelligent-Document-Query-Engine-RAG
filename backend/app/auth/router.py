"""HTTP authentication endpoints for password and refresh-session flows."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.auth.config import get_access_token_ttl_seconds, get_refresh_cookie_secure
from backend.app.auth.cookies import (
    REFRESH_COOKIE_NAME,
    clear_refresh_cookie,
    set_refresh_cookie,
)
from backend.app.auth.dependencies import get_current_user
from backend.app.auth.errors import (
    AuthConfigurationError,
    DuplicateUserError,
    ExpiredRefreshSessionError,
    InvalidCredentialsError,
    InvalidEmailError,
    RefreshSessionError,
)
from backend.app.auth.origin import validate_auth_request_origin
from backend.app.auth.refresh_sessions import (
    add_refresh_session,
    revoke_refresh_session,
    rotate_refresh_session_in_transaction,
)
from backend.app.auth.services import add_password_user, verify_user_credentials
from backend.app.auth.tokens import create_access_token
from backend.app.schemas.auth import AuthResponse, LoginRequest, PublicUser, RegisterRequest
from persistence.db import get_session
from persistence.models import User


router = APIRouter(prefix="/auth", tags=["authentication"])
_NO_STORE = {"Cache-Control": "no-store"}
_BEARER_HEADERS = {"WWW-Authenticate": "Bearer"}
_POST_ORIGIN_DEPENDENCY = Depends(validate_auth_request_origin)


def _auth_response(*, access_token: str, user: PublicUser) -> AuthResponse:
    return AuthResponse(
        access_token=access_token,
        expires_in=get_access_token_ttl_seconds(),
        user=user,
    )


def _service_unavailable() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Authentication service is unavailable.",
    )


def _invalid_login() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password.",
        headers=_BEARER_HEADERS,
    )


def _invalid_refresh_response(*, clear_expired_cookie: bool = False) -> JSONResponse:
    response = JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"detail": "Invalid refresh credential."},
        headers={**_BEARER_HEADERS, **_NO_STORE},
    )
    if clear_expired_cookie:
        clear_refresh_cookie(response)
    return response


@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[_POST_ORIGIN_DEPENDENCY],
)
def register(
    payload: RegisterRequest,
    response: Response,
    session: Annotated[Session, Depends(get_session)],
) -> AuthResponse:
    """Atomically create a password user and its first refresh session."""
    try:
        user = add_password_user(
            session,
            email=payload.email,
            password=payload.password,
            display_name=payload.display_name,
        )
        created_refresh = add_refresh_session(session, user.id)
        access_token = create_access_token(user.id)
        public_user = PublicUser.model_validate(user)
        response_payload = _auth_response(access_token=access_token, user=public_user)
        get_refresh_cookie_secure()
        set_refresh_cookie(
            response,
            raw_token=created_refresh.raw_token,
            expires_at=created_refresh.session.expires_at,
        )
        response.headers.update(_NO_STORE)
        session.commit()
    except InvalidEmailError as exc:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid email address.",
        ) from exc
    except DuplicateUserError as exc:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        ) from exc
    except (SQLAlchemyError, RefreshSessionError, AuthConfigurationError) as exc:
        session.rollback()
        raise _service_unavailable() from exc

    return response_payload


@router.post(
    "/login",
    response_model=AuthResponse,
    dependencies=[_POST_ORIGIN_DEPENDENCY],
)
def login(
    payload: LoginRequest,
    response: Response,
    session: Annotated[Session, Depends(get_session)],
) -> AuthResponse:
    """Verify password credentials and create an independent device session."""
    try:
        user, _ = verify_user_credentials(
            session,
            email=payload.email,
            password=payload.password,
        )
        created_refresh = add_refresh_session(session, user.id)
        access_token = create_access_token(user.id)
        public_user = PublicUser.model_validate(user)
        response_payload = _auth_response(access_token=access_token, user=public_user)
        get_refresh_cookie_secure()
        set_refresh_cookie(
            response,
            raw_token=created_refresh.raw_token,
            expires_at=created_refresh.session.expires_at,
        )
        response.headers.update(_NO_STORE)
        session.commit()
    except (InvalidEmailError, InvalidCredentialsError) as exc:
        session.rollback()
        raise _invalid_login() from exc
    except (SQLAlchemyError, RefreshSessionError, AuthConfigurationError) as exc:
        session.rollback()
        raise _service_unavailable() from exc

    return response_payload


@router.post(
    "/refresh",
    response_model=AuthResponse,
    dependencies=[_POST_ORIGIN_DEPENDENCY],
)
def refresh(
    request: Request,
    response: Response,
    session: Annotated[Session, Depends(get_session)],
) -> AuthResponse | Response:
    """Rotate the cookie credential and issue a new access token."""
    raw_token = request.cookies.get(REFRESH_COOKIE_NAME)
    if not raw_token:
        return _invalid_refresh_response()

    try:
        rotated = rotate_refresh_session_in_transaction(session, raw_token)
        access_token = create_access_token(rotated.user.id)
        public_user = PublicUser.model_validate(rotated.user)
        response_payload = _auth_response(access_token=access_token, user=public_user)
        get_refresh_cookie_secure()
        set_refresh_cookie(
            response,
            raw_token=rotated.raw_token,
            expires_at=rotated.session.expires_at,
        )
        response.headers.update(_NO_STORE)
        session.commit()
    except ExpiredRefreshSessionError:
        session.rollback()
        return _invalid_refresh_response(clear_expired_cookie=True)
    except RefreshSessionError:
        session.rollback()
        # Do not clear unknown/revoked credentials: a late response for R1 must
        # not erase an R2 cookie installed by a concurrent successful rotation.
        return _invalid_refresh_response()
    except (SQLAlchemyError, AuthConfigurationError) as exc:
        session.rollback()
        raise _service_unavailable() from exc

    return response_payload


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    dependencies=[_POST_ORIGIN_DEPENDENCY],
)
def logout(
    request: Request,
    session: Annotated[Session, Depends(get_session)],
) -> Response:
    """Idempotently revoke and clear only this browser's refresh session."""
    raw_token = request.cookies.get(REFRESH_COOKIE_NAME)
    try:
        get_refresh_cookie_secure()
        if raw_token:
            revoke_refresh_session(session, raw_token)
    except (SQLAlchemyError, RefreshSessionError, AuthConfigurationError) as exc:
        session.rollback()
        raise _service_unavailable() from exc

    response = Response(status_code=status.HTTP_204_NO_CONTENT, headers=_NO_STORE)
    clear_refresh_cookie(response)
    return response


@router.get("/me", response_model=PublicUser)
def me(
    response: Response,
    current_user: Annotated[User, Depends(get_current_user)],
) -> PublicUser:
    """Return fresh database-backed public account state."""
    response.headers.update(_NO_STORE)
    return PublicUser.model_validate(current_user)
