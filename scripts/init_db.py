"""Create persistence tables explicitly.

Usage:
    python scripts/init_db.py

Set DATABASE_URL to choose a database. The default is sqlite:///./rag_persistence.db.
The script intentionally avoids printing DATABASE_URL because managed database
URLs often contain credentials.
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.exc import SQLAlchemyError


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from persistence import models  # noqa: F401,E402 - registers SQLAlchemy metadata
from persistence.db import Base, engine  # noqa: E402
from persistence.models import User  # noqa: E402
from persistence.user_context import DEFAULT_USER_ID  # noqa: E402


def _seed_default_user() -> None:
    dialect_name = engine.dialect.name
    with engine.begin() as connection:
        if dialect_name == "mysql":
            statement = mysql_insert(User.__table__).values(id=DEFAULT_USER_ID)
            statement = statement.on_duplicate_key_update(id=statement.inserted.id)
            connection.execute(statement)
            return

        connection.execute(
            User.__table__.insert().prefix_with("OR IGNORE").values(id=DEFAULT_USER_ID)
        )


def main() -> None:
    try:
        Base.metadata.create_all(bind=engine)
        _seed_default_user()
    except SQLAlchemyError as exc:
        raise RuntimeError("Database initialization failed. Check DATABASE_URL and DB_CA_CERT configuration.") from exc

    table_names = ", ".join(sorted(Base.metadata.tables))
    print(f"Using configured database engine. Created/verified persistence tables: {table_names}")
    print(f"Ensured default user exists: {DEFAULT_USER_ID}")


if __name__ == "__main__":
    main()
