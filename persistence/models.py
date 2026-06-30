"""SQLAlchemy models for persisted RAG documents, chunks, queries, and citations."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from persistence.db import Base


def _new_id() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    email: Mapped[Optional[str]] = mapped_column(String(320), nullable=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    auth_provider: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
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


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(String(128), ForeignKey("users.id"), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    filename: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    cache_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="processing")
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
    claim_verifications_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
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
