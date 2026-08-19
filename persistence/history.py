"""Ownership-scoped persistent history reads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import func, select

from persistence.ownership import OwnedResourceNotFoundError


@dataclass(frozen=True)
class StoredHistoryDocument:
    id: str
    source_type: str
    filename: Optional[str]
    source_url: Optional[str]
    status: str
    created_at: datetime
    chunk_count: int
    query_count: int


def list_user_documents(*, user_id: str, limit: int) -> list[StoredHistoryDocument]:
    """List only documents owned by one explicit user."""
    from persistence.db import SessionLocal
    from persistence.models import Chunk, Document, Query

    with SessionLocal() as session:
        chunk_counts = (
            select(Chunk.document_id, func.count(Chunk.id).label("chunk_count"))
            .where(Chunk.user_id == user_id)
            .group_by(Chunk.document_id)
            .subquery()
        )
        query_counts = (
            select(Query.document_id, func.count(Query.id).label("query_count"))
            .where(Query.user_id == user_id)
            .group_by(Query.document_id)
            .subquery()
        )
        rows = session.execute(
            select(
                Document,
                func.coalesce(chunk_counts.c.chunk_count, 0),
                func.coalesce(query_counts.c.query_count, 0),
            )
            .outerjoin(chunk_counts, chunk_counts.c.document_id == Document.id)
            .outerjoin(query_counts, query_counts.c.document_id == Document.id)
            .where(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
            .limit(limit)
        ).all()
        return [
            StoredHistoryDocument(
                id=document.id,
                source_type=document.source_type,
                filename=document.filename,
                source_url=document.source_url,
                status=document.status,
                created_at=document.created_at,
                chunk_count=int(chunk_count),
                query_count=int(query_count),
            )
            for document, chunk_count, query_count in rows
        ]


def list_user_document_queries(*, user_id: str, document_id: str, limit: int) -> list[Any]:
    """List queries only after resolving the document in the user's namespace."""
    from persistence.db import SessionLocal
    from persistence.models import Document, Query

    with SessionLocal() as session:
        owned_document_id = session.execute(
            select(Document.id).where(
                Document.id == document_id,
                Document.user_id == user_id,
            )
        ).scalar_one_or_none()
        if owned_document_id is None:
            raise OwnedResourceNotFoundError("Document not found.")

        return list(
            session.execute(
                select(Query)
                .where(
                    Query.user_id == user_id,
                    Query.document_id == document_id,
                )
                .order_by(Query.created_at.desc())
                .limit(limit)
            ).scalars().all()
        )


def list_user_query_citations(*, user_id: str, query_id: str, limit: int) -> list[Any]:
    """List citations only after resolving the query in the user's namespace."""
    from persistence.db import SessionLocal
    from persistence.models import Citation, Query

    with SessionLocal() as session:
        owned_query = session.execute(
            select(Query.id, Query.document_id).where(
                Query.id == query_id,
                Query.user_id == user_id,
            )
        ).one_or_none()
        if owned_query is None:
            raise OwnedResourceNotFoundError("Query not found.")

        return list(
            session.execute(
                select(Citation)
                .where(
                    Citation.user_id == user_id,
                    Citation.query_id == query_id,
                    Citation.document_id == owned_query.document_id,
                )
                .order_by(Citation.rank.asc())
                .limit(limit)
            ).scalars().all()
        )
