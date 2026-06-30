"""Database primitives for persistence models.

Tables are created only by scripts/init_db.py. Importing this module must not
modify the database.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from dotenv import load_dotenv


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./rag_persistence.db")
DB_CA_CERT = os.getenv("DB_CA_CERT")


def _is_mysql_url(database_url: str) -> bool:
    return make_url(database_url).drivername.startswith("mysql")


def _build_connect_args(database_url: str) -> dict:
    if database_url.startswith("sqlite:"):
        return {"check_same_thread": False}

    if not _is_mysql_url(database_url):
        return {}

    if not DB_CA_CERT:
        raise RuntimeError("Managed MySQL requires DB_CA_CERT to enable TLS certificate verification.")

    ca_path = Path(DB_CA_CERT)
    if not ca_path.exists():
        raise RuntimeError("DB_CA_CERT is configured, but the certificate file does not exist.")

    return {"ssl": {"ca": str(ca_path)}}


_connect_args = _build_connect_args(DATABASE_URL)
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
