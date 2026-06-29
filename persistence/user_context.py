"""Current-user shim for persistence.

OAuth can later replace get_current_user_id() without changing database schema.
"""

DEFAULT_USER_ID = "local-dev-user"


def get_current_user_id() -> str:
    return DEFAULT_USER_ID
