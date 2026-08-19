"""Small browser-origin check for cookie-setting or cookie-authenticated requests."""

from __future__ import annotations

from urllib.parse import urlsplit

from fastapi import HTTPException, Request, status

from backend.app.auth.config import get_auth_allowed_origins


def _origin_from_url(value: str) -> str | None:
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def validate_auth_request_origin(request: Request) -> None:
    """Reject cross-origin browser POSTs while permitting non-browser clients."""
    origin_header = request.headers.get("origin")
    referer_header = request.headers.get("referer")
    fetch_site = request.headers.get("sec-fetch-site", "").lower()

    supplied_origin = None
    if origin_header:
        supplied_origin = _origin_from_url(origin_header)
    elif referer_header:
        supplied_origin = _origin_from_url(referer_header)
    elif fetch_site == "cross-site":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cross-origin authentication request rejected.",
        )
    else:
        # Scripts and other non-browser clients do not necessarily send browser
        # origin metadata. Browsers cannot suppress Origin on cross-site POSTs.
        return

    request_origin = _origin_from_url(str(request.base_url).rstrip("/"))
    if (
        supplied_origin is None
        or (supplied_origin != request_origin and supplied_origin not in get_auth_allowed_origins())
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cross-origin authentication request rejected.",
        )
