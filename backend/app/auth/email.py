"""Email identity validation and normalization."""

from email_validator import EmailNotValidError, validate_email

from backend.app.auth.errors import InvalidEmailError


def normalize_email(raw_email: str) -> str:
    """Return email-validator's normalized identity without DNS checks."""
    try:
        validated = validate_email(raw_email, check_deliverability=False)
    except (EmailNotValidError, TypeError) as exc:
        raise InvalidEmailError("Invalid email address.") from exc
    return validated.normalized
