"""Database primitives for persistence models.

Tables are created only by scripts/init_db.py. Importing this module must not
modify the database.
"""

from __future__ import annotations

import base64
import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from dotenv import load_dotenv


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./rag_persistence.db")
DB_CA_CERT_B64 = os.getenv("DB_CA_CERT_B64")
LOCAL_CERT_HOSTNAME_OVERRIDE_ENV = "DB_ALLOW_LOCAL_TEST_CERT_HOSTNAME_MISMATCH"


def _bootstrap_db_ca_cert_from_b64() -> str | None:
    """Materialize the HF-provided CA certificate secret without logging it."""
    if not DB_CA_CERT_B64:
        return os.getenv("DB_CA_CERT")

    ca_path = Path("/tmp/aiven-ca.pem")
    try:
        ca_path.parent.mkdir(parents=True, exist_ok=True)
        ca_path.write_bytes(base64.b64decode(DB_CA_CERT_B64, validate=True))
        ca_path.chmod(0o600)
    except Exception as exc:
        raise RuntimeError("DB_CA_CERT_B64 is configured, but the CA certificate could not be prepared.") from exc

    os.environ["DB_CA_CERT"] = str(ca_path)
    return str(ca_path)


DB_CA_CERT = _bootstrap_db_ca_cert_from_b64()


def _is_mysql_url(database_url: str) -> bool:
    return make_url(database_url).drivername.startswith("mysql")


def _build_connect_args(database_url: str) -> dict:
    if database_url.startswith("sqlite:"):
        return {"check_same_thread": False}

    if not _is_mysql_url(database_url):
        return {}

    if not DB_CA_CERT:
        raise RuntimeError("Managed MySQL requires DB_CA_CERT or DB_CA_CERT_B64 for TLS certificate verification.")

    ca_path = Path(DB_CA_CERT)
    if not ca_path.exists():
        raise RuntimeError("DB_CA_CERT is configured, but the certificate file does not exist.")

    ssl_options: dict[str, object] = {"ca": str(ca_path)}
    allow_local_hostname_mismatch = os.getenv(LOCAL_CERT_HOSTNAME_OVERRIDE_ENV, "").strip().lower()
    if allow_local_hostname_mismatch in {"1", "true", "yes", "on"}:
        parsed = make_url(database_url)
        database_name = (parsed.database or "").lower()
        if parsed.host not in {"127.0.0.1", "localhost", "::1"} or not any(
            marker in database_name for marker in ("test", "testing", "disposable")
        ):
            raise RuntimeError(
                f"{LOCAL_CERT_HOSTNAME_OVERRIDE_ENV} is restricted to loopback disposable/test databases."
            )
        ssl_options["check_hostname"] = False

    return {"ssl": ssl_options}


_connect_args = _build_connect_args(DATABASE_URL)
engine = create_engine(DATABASE_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


@contextmanager
def mysql_document_lock(
    session: Session,
    *,
    user_id: str,
    document_key_parts: Sequence[str],
    timeout_seconds: int = 30,
) -> Iterator[None]:
    """Serialize work for one document on MySQL; remain a no-op elsewhere."""
    if session.get_bind().dialect.name != "mysql":
        yield
        return

    raw_key = ":".join((user_id, *document_key_parts))
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:48]
    lock_name = f"rag-document:{digest}"
    acquired = session.execute(
        text("SELECT GET_LOCK(:lock_name, :timeout_seconds)"),
        {"lock_name": lock_name, "timeout_seconds": timeout_seconds},
    ).scalar_one()
    if acquired != 1:
        raise TimeoutError("Timed out waiting for the document persistence lock.")

    try:
        yield
    finally:
        try:
            session.execute(
                text("SELECT RELEASE_LOCK(:lock_name)"),
                {"lock_name": lock_name},
            )
        except Exception:
            try:
                session.invalidate()
            except Exception:
                pass


def get_session() -> Iterator[Session]:
    """Yield a SQLAlchemy session for future API dependencies."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
