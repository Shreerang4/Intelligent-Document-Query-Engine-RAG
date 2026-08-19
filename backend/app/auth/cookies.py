"""Central refresh-cookie configuration and response helpers."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from starlette.responses import Response

from backend.app.auth.config import get_refresh_cookie_secure


REFRESH_COOKIE_NAME = "idqe_refresh"
REFRESH_COOKIE_PATH = "/auth"
REFRESH_COOKIE_SAMESITE = "lax"


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def set_refresh_cookie(
    response: Response,
    *,
    raw_token: str,
    expires_at: datetime,
) -> None:
    """Set the host-only refresh cookie through its absolute session expiry."""
    absolute_expiry = _as_utc(expires_at)
    remaining_seconds = max(
        0,
        math.ceil((absolute_expiry - datetime.now(timezone.utc)).total_seconds()),
    )
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=raw_token,
        max_age=remaining_seconds,
        expires=absolute_expiry,
        path=REFRESH_COOKIE_PATH,
        secure=get_refresh_cookie_secure(),
        httponly=True,
        samesite=REFRESH_COOKIE_SAMESITE,
    )


def clear_refresh_cookie(response: Response) -> None:
    """Clear the refresh cookie with attributes matching the setter."""
    response.delete_cookie(
        key=REFRESH_COOKIE_NAME,
        path=REFRESH_COOKIE_PATH,
        secure=get_refresh_cookie_secure(),
        httponly=True,
        samesite=REFRESH_COOKIE_SAMESITE,
    )
