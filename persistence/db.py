"""Database primitives for persistence models.

Tables are created only by scripts/init_db.py. Importing this module must not
modify the database.
"""

from __future__ import annotations

import os
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./rag_persistence.db")

_connect_args = (
    {"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite:")
    else {}
)

engine = create_engine(DATABASE_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def get_session() -> Iterator[Session]:
    """Yield a SQLAlchemy session for future API dependencies."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
