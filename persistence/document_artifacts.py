"""Persistent chunk and embedding artifact loading for document reuse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np
from sqlalchemy import select


EMBEDDING_DTYPE = "float32"


class ArtifactValidationError(ValueError):
    """Raised when persisted chunk ordering cannot be trusted."""


@dataclass(frozen=True)
class StoredDocumentArtifacts:
    document_id: str
    chunks: list[dict[str, Any]]
    embedding_matrix: Optional[np.ndarray]
    needs_embedding_backfill: bool


def serialize_embedding(embedding: Any) -> tuple[bytes, int, str]:
    vector = np.asarray(embedding, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise ArtifactValidationError("A chunk embedding must be a non-empty one-dimensional vector.")

    contiguous_vector = np.ascontiguousarray(vector, dtype=np.float32)
    return contiguous_vector.tobytes(), int(contiguous_vector.shape[0]), EMBEDDING_DTYPE


def deserialize_embedding(blob: Any, dimension: Any, dtype: Any) -> np.ndarray:
    if blob is None or dimension is None or dtype != EMBEDDING_DTYPE:
        raise ArtifactValidationError("A stored chunk embedding is missing required metadata.")

    try:
        parsed_dimension = int(dimension)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError("A stored embedding dimension is invalid.") from exc
    if parsed_dimension <= 0:
        raise ArtifactValidationError("A stored embedding dimension must be positive.")

    vector = np.frombuffer(bytes(blob), dtype=np.float32)
    if vector.size != parsed_dimension:
        raise ArtifactValidationError("A stored embedding byte length does not match its dimension.")
    return np.ascontiguousarray(vector, dtype=np.float32)


def embedding_matrix_from_chunk_rows(chunk_rows: Sequence[Any]) -> np.ndarray:
    if not chunk_rows:
        raise ArtifactValidationError("A persisted document has no chunks.")

    expected_indices = list(range(len(chunk_rows)))
    actual_indices = [int(row.chunk_index) for row in chunk_rows]
    if actual_indices != expected_indices:
        raise ArtifactValidationError("Persisted chunks are not in contiguous chunk_index order.")

    vectors = [
        deserialize_embedding(row.embedding_blob, row.embedding_dimension, row.embedding_dtype)
        for row in chunk_rows
    ]
    dimensions = {int(vector.shape[0]) for vector in vectors}
    if len(dimensions) != 1:
        raise ArtifactValidationError("Persisted chunk embedding dimensions do not match.")

    matrix = np.vstack(vectors).astype(np.float32, copy=False)
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(chunk_rows):
        raise ArtifactValidationError("Persisted embedding count does not match chunk count.")
    return matrix


def load_document_artifacts(
    *,
    user_id: str,
    source_hash: str,
    embedding_model: str,
    embedding_format: str,
) -> Optional[StoredDocumentArtifacts]:
    from persistence.db import SessionLocal
    from persistence.models import Chunk, Document

    with SessionLocal() as session:
        document = session.execute(
            select(Document)
            .where(Document.user_id == user_id, Document.source_hash == source_hash)
            .order_by(Document.created_at.desc())
        ).scalars().first()
        if document is None:
            return None

        chunk_rows = session.execute(
            select(Chunk)
            .where(Chunk.user_id == user_id, Chunk.document_id == document.id)
            .order_by(Chunk.chunk_index.asc())
        ).scalars().all()

        expected_indices = list(range(len(chunk_rows)))
        actual_indices = [int(row.chunk_index) for row in chunk_rows]
        if actual_indices != expected_indices:
            raise ArtifactValidationError("Persisted chunks are not in contiguous chunk_index order.")

        chunks = [
            {
                "text": row.text,
                "page": int(row.page_number),
                "chunk_id": int(row.chunk_id),
            }
            for row in chunk_rows
        ]

        embedding_matrix: Optional[np.ndarray] = None
        configuration_matches = (
            document.embedding_model == embedding_model
            and document.embedding_format == embedding_format
        )
        if configuration_matches and chunk_rows:
            try:
                embedding_matrix = embedding_matrix_from_chunk_rows(chunk_rows)
            except ArtifactValidationError:
                embedding_matrix = None

        return StoredDocumentArtifacts(
            document_id=document.id,
            chunks=chunks,
            embedding_matrix=embedding_matrix,
            needs_embedding_backfill=embedding_matrix is None,
        )


def load_or_backfill_document_embeddings(
    *,
    user_id: str,
    document_id: str,
    source_hash: str,
    embedding_model: str,
    embedding_format: str,
    embedding_generator: Callable[[list[dict[str, Any]]], np.ndarray],
) -> StoredDocumentArtifacts:
    """Reload and, when still necessary, repair embeddings under a document lock."""
    from persistence.db import SessionLocal, mysql_document_lock
    from persistence.models import Chunk, Document

    with SessionLocal() as session:
        try:
            with mysql_document_lock(
                session,
                user_id=user_id,
                document_key_parts=(source_hash,),
            ):
                document = session.execute(
                    select(Document).where(
                        Document.id == document_id,
                        Document.user_id == user_id,
                        Document.source_hash == source_hash,
                    )
                ).scalar_one_or_none()
                if document is None:
                    raise ArtifactValidationError("Document disappeared before embedding backfill.")

                chunk_rows = session.execute(
                    select(Chunk)
                    .where(Chunk.user_id == user_id, Chunk.document_id == document_id)
                    .order_by(Chunk.chunk_index.asc())
                ).scalars().all()
                expected_indices = list(range(len(chunk_rows)))
                if [int(row.chunk_index) for row in chunk_rows] != expected_indices:
                    raise ArtifactValidationError("Persisted chunks are not in contiguous chunk_index order.")
                if not chunk_rows:
                    raise ArtifactValidationError("A persisted document has no chunks.")

                chunks = [
                    {
                        "text": row.text,
                        "page": int(row.page_number),
                        "chunk_id": int(row.chunk_id),
                    }
                    for row in chunk_rows
                ]
                matrix: Optional[np.ndarray] = None
                configuration_matches = (
                    document.embedding_model == embedding_model
                    and document.embedding_format == embedding_format
                )
                if configuration_matches:
                    try:
                        matrix = embedding_matrix_from_chunk_rows(chunk_rows)
                    except ArtifactValidationError:
                        matrix = None

                if matrix is None:
                    matrix = np.ascontiguousarray(embedding_generator(chunks), dtype=np.float32)
                    if matrix.ndim != 2 or matrix.shape[0] != len(chunk_rows) or matrix.shape[1] == 0:
                        raise ArtifactValidationError("Embedding backfill count does not match chunk count.")

                    for row, vector in zip(chunk_rows, matrix):
                        blob, dimension, dtype = serialize_embedding(vector)
                        row.embedding_blob = blob
                        row.embedding_dimension = dimension
                        row.embedding_dtype = dtype

                    document.embedding_model = embedding_model
                    document.embedding_format = embedding_format
                    session.flush()
                    persisted_document_id = str(document.id)
                    session.commit()
                else:
                    persisted_document_id = str(document.id)

                result = StoredDocumentArtifacts(
                    document_id=persisted_document_id,
                    chunks=chunks,
                    embedding_matrix=matrix,
                    needs_embedding_backfill=False,
                )
            return result
        except Exception:
            session.rollback()
            raise
