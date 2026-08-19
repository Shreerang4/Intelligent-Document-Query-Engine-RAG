"""Auth-scoped HTTP error handling that never reflects credential input."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response


async def auth_request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> Response:
    """Redact submitted values from auth validation errors only."""
    if not request.url.path.startswith("/auth/"):
        return await request_validation_exception_handler(request, exc)

    redacted_errors = [
        {
            key: value
            for key, value in error.items()
            if key not in {"input", "ctx"}
        }
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": redacted_errors})


def install_auth_http_error_handlers(app: FastAPI) -> None:
    """Install credential-redacting validation handling on a FastAPI app."""
    app.add_exception_handler(
        RequestValidationError,
        auth_request_validation_exception_handler,
    )
