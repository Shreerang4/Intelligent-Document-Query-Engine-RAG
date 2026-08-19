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

from sqlalchemy.exc import SQLAlchemyError


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from persistence import models  # noqa: F401,E402 - registers SQLAlchemy metadata
from persistence.db import Base, engine  # noqa: E402


def main() -> None:
    try:
        Base.metadata.create_all(bind=engine)
    except SQLAlchemyError as exc:
        raise RuntimeError("Database initialization failed. Check DATABASE_URL and DB_CA_CERT configuration.") from exc

    table_names = ", ".join(sorted(Base.metadata.tables))
    print(f"Using configured database engine. Created/verified persistence tables: {table_names}")


if __name__ == "__main__":
    main()
