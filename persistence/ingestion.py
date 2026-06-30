"""Best-effort persistence side effects for document ingestion."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Mapping, Optional, Sequence

from sqlalchemy import select

from persistence.user_context import get_current_user_id


logger = logging.getLogger(__name__)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _compute_source_hash(*, pdf_bytes: Optional[bytes], source_url: Optional[str], cache_key: Optional[str]) -> str:
    if pdf_bytes is not None:
        return hashlib.sha256(pdf_bytes).hexdigest()
    if cache_key:
        return _hash_text(cache_key)
    if source_url:
        return _hash_text(source_url)
    raise ValueError("Cannot compute source hash without PDF bytes, source URL, or cache key.")


def _persist_ingested_document(
    *,
    user_id: str,
    source_type: str,
    chunks: Sequence[Mapping[str, Any]],
    filename: Optional[str],
    source_url: Optional[str],
    source_hash: str,
    cache_key: Optional[str],
    embedding_model: str,
    embedding_format: str,
    retrieval_mode: str,
    reranker_model: str,
    k_initial: int,
    k_final: int,
) -> Optional[str]:
    from persistence.db import SessionLocal
    from persistence.models import Chunk, Document, User

    with SessionLocal() as session:
        existing_document = session.execute(
            select(Document)
            .where(Document.user_id == user_id, Document.source_hash == source_hash)
            .order_by(Document.created_at.desc())
        ).scalars().first()
        if existing_document is not None:
            return existing_document.id

        if session.get(User, user_id) is None:
            session.add(User(id=user_id))

        document = Document(
            user_id=user_id,
            source_type=source_type,
            filename=filename,
            source_url=source_url,
            source_hash=source_hash,
            cache_key=cache_key,
            status="ingested",
            embedding_model=embedding_model,
            embedding_format=embedding_format,
            retrieval_mode=retrieval_mode,
            reranker_model=reranker_model,
            k_initial=k_initial,
            k_final=k_final,
        )
        session.add(document)
        session.flush()

        chunk_rows = [
            {
                "user_id": user_id,
                "document_id": document.id,
                "chunk_id": int(chunk["chunk_id"]),
                "chunk_index": index,
                "page_number": int(chunk["page"]),
                "text": str(chunk["text"]),
                "text_hash": _hash_text(str(chunk["text"])),
                "char_count": len(str(chunk["text"])),
            }
            for index, chunk in enumerate(chunks)
        ]
        if chunk_rows:
            session.bulk_insert_mappings(Chunk, chunk_rows)

        session.commit()
        return document.id


def persist_ingested_document_best_effort(
    *,
    source_type: str,
    chunks: Sequence[Mapping[str, Any]],
    filename: Optional[str] = None,
    source_url: Optional[str] = None,
    pdf_bytes: Optional[bytes] = None,
    cache_key: Optional[str] = None,
    embedding_model: str,
    embedding_format: str,
    retrieval_mode: str,
    reranker_model: str,
    k_initial: int,
    k_final: int,
) -> Optional[str]:
    """Persist a document and chunks without letting DB failures affect ingestion."""
    try:
        source_hash = _compute_source_hash(pdf_bytes=pdf_bytes, source_url=source_url, cache_key=cache_key)
        return _persist_ingested_document(
            user_id=get_current_user_id(),
            source_type=source_type,
            chunks=chunks,
            filename=filename,
            source_url=source_url,
            source_hash=source_hash,
            cache_key=cache_key,
            embedding_model=embedding_model,
            embedding_format=embedding_format,
            retrieval_mode=retrieval_mode,
            reranker_model=reranker_model,
            k_initial=k_initial,
            k_final=k_final,
        )
    except Exception as exc:
        logger.warning("Persistence ingestion side effect failed safely: %s", type(exc).__name__)
        return None
