"""SQLAlchemy models for persisted RAG documents, chunks, queries, and citations."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, List, Optional

from sqlalchemy import (
    BINARY,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from persistence.db import Base


def _new_id() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(128), primary_key=True, default=_new_id)
    email: Mapped[Optional[str]] = mapped_column(String(320), nullable=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    auth_provider: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    password_hash: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    documents: Mapped[List["Document"]] = relationship(back_populates="user")
    chunks: Mapped[List["Chunk"]] = relationship(back_populates="user")
    queries: Mapped[List["Query"]] = relationship(back_populates="user")
    citations: Mapped[List["Citation"]] = relationship(back_populates="user")
    refresh_sessions: Mapped[List["RefreshSession"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("email", name="uq_users_email"),
    )


class RefreshSession(Base):
    __tablename__ = "refresh_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    token_hash: Mapped[bytes] = mapped_column(BINARY(32), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    replaced_by_session_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("refresh_sessions.id", ondelete="SET NULL"),
        nullable=True,
    )

    user: Mapped["User"] = relationship(back_populates="refresh_sessions")
    replaced_by: Mapped[Optional["RefreshSession"]] = relationship(
        remote_side="RefreshSession.id",
        foreign_keys=[replaced_by_session_id],
    )

    __table_args__ = (
        UniqueConstraint("token_hash", name="uq_refresh_sessions_token_hash"),
        Index("ix_refresh_sessions_expires_at", "expires_at"),
        Index(
            "ix_refresh_sessions_user_id_revoked_at_expires_at",
            "user_id",
            "revoked_at",
            "expires_at",
        ),
    )


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(String(128), ForeignKey("users.id"), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    filename: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    cache_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    object_key: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    content_type: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    byte_size: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="queued",
        server_default="queued",
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_format: Mapped[str] = mapped_column(String(64), nullable=False)
    retrieval_mode: Mapped[str] = mapped_column(String(64), nullable=False)
    reranker_model: Mapped[str] = mapped_column(String(255), nullable=False)
    k_initial: Mapped[int] = mapped_column(Integer, nullable=False)
    k_final: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user: Mapped["User"] = relationship(back_populates="documents")
    chunks: Mapped[List["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    queries: Mapped[List["Query"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    citations: Mapped[List["Citation"]] = relationship(back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_documents_user_id_created_at", "user_id", "created_at"),
        Index("ix_documents_user_id_source_hash", "user_id", "source_hash"),
        UniqueConstraint("object_key", name="uq_documents_object_key"),
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(String(128), ForeignKey("users.id"), nullable=False)
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id"), nullable=False)
    chunk_id: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    char_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    embedding_blob: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    embedding_dimension: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    embedding_dtype: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user: Mapped["User"] = relationship(back_populates="chunks")
    document: Mapped["Document"] = relationship(back_populates="chunks")
    citations: Mapped[List["Citation"]] = relationship(back_populates="chunk")

    __table_args__ = (
        Index("ix_chunks_user_id_document_id", "user_id", "document_id"),
        UniqueConstraint("document_id", "chunk_id", name="uq_chunks_document_id_chunk_id"),
    )


class Query(Base):
    __tablename__ = "queries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(String(128), ForeignKey("users.id"), nullable=False)
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id"), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    is_abstained: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    request_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    request_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    claim_verifications_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    retrieval_mode: Mapped[str] = mapped_column(String(64), nullable=False)
    reranker_model: Mapped[str] = mapped_column(String(255), nullable=False)
    k_initial: Mapped[int] = mapped_column(Integer, nullable=False)
    k_final: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user: Mapped["User"] = relationship(back_populates="queries")
    document: Mapped["Document"] = relationship(back_populates="queries")
    citations: Mapped[List["Citation"]] = relationship(back_populates="query", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_queries_user_id_document_id_created_at", "user_id", "document_id", "created_at"),
        UniqueConstraint(
            "user_id",
            "request_id",
            "request_index",
            name="uq_queries_user_request_index",
        ),
    )


class Citation(Base):
    __tablename__ = "citations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(String(128), ForeignKey("users.id"), nullable=False)
    query_id: Mapped[str] = mapped_column(String(36), ForeignKey("queries.id"), nullable=False)
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id"), nullable=False)
    chunk_db_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("chunks.id"), nullable=True)
    chunk_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    excerpt: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    reranker_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user: Mapped["User"] = relationship(back_populates="citations")
    query: Mapped["Query"] = relationship(back_populates="citations")
    document: Mapped["Document"] = relationship(back_populates="citations")
    chunk: Mapped[Optional["Chunk"]] = relationship(back_populates="citations")

    __table_args__ = (
        Index("ix_citations_user_id_query_id", "user_id", "query_id"),
        Index("ix_citations_query_id_rank", "query_id", "rank"),
    )
