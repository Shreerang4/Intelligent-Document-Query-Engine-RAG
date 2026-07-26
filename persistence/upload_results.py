"""Atomic upload persistence and committed request recovery."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from sqlalchemy import select

from persistence.db import mysql_document_lock
from persistence.document_artifacts import ArtifactValidationError, serialize_embedding


class RequestConflictError(ValueError):
    """Raised when one request ID is reused for different input."""


@dataclass(frozen=True)
class RecoveredUploadResult:
    document_id: str
    response_payload: dict[str, Any]


def _mapping_value(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _dump_model(item: Any) -> Any:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if isinstance(item, Mapping):
        return dict(item)
    return item


def persist_upload_result_atomic(
    *,
    user_id: str,
    proposed_document_id: str,
    source_hash: str,
    filename: Optional[str],
    cache_key: str,
    chunks: Sequence[Mapping[str, Any]],
    embedding_matrix: Optional[np.ndarray],
    question_results: Sequence[Any],
    request_id: Optional[str],
    embedding_model: str,
    embedding_format: str,
    retrieval_mode: str,
    reranker_model: str,
    k_initial: int,
    k_final: int,
) -> str:
    from persistence.db import SessionLocal
    from persistence.models import Chunk, Citation, Document, Query, User

    matrix: Optional[np.ndarray] = None
    if embedding_matrix is not None:
        matrix = np.ascontiguousarray(embedding_matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks) or matrix.shape[1] == 0:
            raise ArtifactValidationError("Embedding matrix shape does not match the chunk sequence.")

    with SessionLocal() as session:
        try:
            with mysql_document_lock(
                session,
                user_id=user_id,
                document_key_parts=(source_hash,),
            ):
                if session.get(User, user_id) is None:
                    session.add(User(id=user_id))

                document = session.execute(
                    select(Document)
                    .where(Document.user_id == user_id, Document.source_hash == source_hash)
                    .order_by(Document.created_at.desc())
                    .with_for_update()
                ).scalars().first()

                if document is None:
                    if matrix is None:
                        raise ArtifactValidationError("A new document requires persisted embeddings.")
                    document = Document(
                        id=proposed_document_id,
                        user_id=user_id,
                        source_type="upload",
                        filename=filename,
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

                    chunk_rows = []
                    for chunk_index, (chunk, vector) in enumerate(zip(chunks, matrix)):
                        chunk_text = str(chunk["text"])
                        blob, dimension, dtype = serialize_embedding(vector)
                        chunk_rows.append(
                            Chunk(
                                user_id=user_id,
                                document_id=document.id,
                                chunk_id=int(chunk["chunk_id"]),
                                chunk_index=chunk_index,
                                page_number=int(chunk["page"]),
                                text=chunk_text,
                                text_hash=hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
                                char_count=len(chunk_text),
                                embedding_blob=blob,
                                embedding_dimension=dimension,
                                embedding_dtype=dtype,
                            )
                        )
                    session.add_all(chunk_rows)
                    session.flush()
                else:
                    chunk_rows = session.execute(
                        select(Chunk)
                        .where(Chunk.user_id == user_id, Chunk.document_id == document.id)
                        .order_by(Chunk.chunk_index.asc())
                    ).scalars().all()
                    if [int(row.chunk_index) for row in chunk_rows] != list(range(len(chunk_rows))):
                        raise ArtifactValidationError("Persisted chunks are not in contiguous chunk_index order.")
                    if not chunk_rows and matrix is not None:
                        for chunk_index, (chunk, vector) in enumerate(zip(chunks, matrix)):
                            chunk_text = str(chunk["text"])
                            blob, dimension, dtype = serialize_embedding(vector)
                            chunk_rows.append(
                                Chunk(
                                    user_id=user_id,
                                    document_id=document.id,
                                    chunk_id=int(chunk["chunk_id"]),
                                    chunk_index=chunk_index,
                                    page_number=int(chunk["page"]),
                                    text=chunk_text,
                                    text_hash=hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
                                    char_count=len(chunk_text),
                                    embedding_blob=blob,
                                    embedding_dimension=dimension,
                                    embedding_dtype=dtype,
                                )
                            )
                        session.add_all(chunk_rows)
                        session.flush()
                    if len(chunk_rows) != len(chunks):
                        raise ArtifactValidationError("Persisted chunk count does not match the request artifact.")

                chunk_by_id = {int(chunk.chunk_id): chunk for chunk in chunk_rows}
                query_rows = []
                for request_index, result in enumerate(question_results):
                    answer_item = _mapping_value(result, "answer_item")
                    claim_verifications = _mapping_value(answer_item, "claim_verifications") or []
                    query_rows.append(
                        Query(
                            user_id=user_id,
                            document_id=document.id,
                            question=str(_mapping_value(answer_item, "question")),
                            answer=str(_mapping_value(answer_item, "answer")),
                            status=str(_mapping_value(answer_item, "status")),
                            is_abstained=bool(_mapping_value(result, "is_abstained")),
                            request_id=request_id,
                            request_index=request_index,
                            claim_verifications_json=(
                                [_dump_model(item) for item in claim_verifications]
                                if claim_verifications
                                else None
                            ),
                            embedding_model=embedding_model,
                            retrieval_mode=retrieval_mode,
                            reranker_model=reranker_model,
                            k_initial=k_initial,
                            k_final=k_final,
                            latency_ms=_mapping_value(result, "latency_ms"),
                        )
                    )
                session.add_all(query_rows)
                session.flush()

                citation_rows = []
                for query, result in zip(query_rows, question_results):
                    answer_item = _mapping_value(result, "answer_item")
                    sources = _mapping_value(answer_item, "sources") or []
                    raw_chunks = _mapping_value(result, "source_chunks") or []
                    raw_chunk_by_id = {
                        int(chunk["chunk_id"]): chunk
                        for chunk in raw_chunks
                        if chunk.get("chunk_id") is not None
                    }
                    for rank, source in enumerate(sources, start=1):
                        chunk_id_raw = _mapping_value(source, "chunk_id")
                        chunk_id = int(chunk_id_raw) if chunk_id_raw is not None else None
                        raw_chunk = raw_chunk_by_id.get(chunk_id) if chunk_id is not None else None
                        stored_chunk = chunk_by_id.get(chunk_id) if chunk_id is not None else None
                        citation_rows.append(
                            Citation(
                                user_id=user_id,
                                query_id=query.id,
                                document_id=document.id,
                                chunk_db_id=stored_chunk.id if stored_chunk is not None else None,
                                chunk_id=chunk_id,
                                rank=rank,
                                page_number=_mapping_value(source, "page"),
                                excerpt=str(_mapping_value(source, "excerpt") or ""),
                                retrieval_score=(raw_chunk or {}).get("retrieval_score"),
                                reranker_score=(raw_chunk or {}).get("reranker_score"),
                            )
                        )
                session.add_all(citation_rows)
                persisted_document_id = str(document.id)
                session.commit()
                return persisted_document_id
        except Exception:
            session.rollback()
            raise


def recover_upload_request(
    *,
    user_id: str,
    request_id: str,
    source_hash: str,
    questions: Sequence[str],
) -> Optional[RecoveredUploadResult]:
    from persistence.db import SessionLocal
    from persistence.models import Citation, Document, Query

    with SessionLocal() as session:
        query_rows = session.execute(
            select(Query)
            .where(Query.user_id == user_id, Query.request_id == request_id)
            .order_by(Query.request_index.asc())
        ).scalars().all()
        if not query_rows:
            return None

        document_ids = {query.document_id for query in query_rows}
        request_indices = [query.request_index for query in query_rows]
        stored_questions = [query.question for query in query_rows]
        if (
            len(document_ids) != 1
            or request_indices != list(range(len(questions)))
            or stored_questions != list(questions)
        ):
            raise RequestConflictError("request_id was already used with different or incomplete input.")

        document_id = next(iter(document_ids))
        document = session.execute(
            select(Document).where(Document.id == document_id, Document.user_id == user_id)
        ).scalar_one_or_none()
        if document is None or document.source_hash != source_hash:
            raise RequestConflictError("request_id was already used with a different document.")

        query_ids = [query.id for query in query_rows]
        citations = session.execute(
            select(Citation)
            .where(Citation.user_id == user_id, Citation.query_id.in_(query_ids))
            .order_by(Citation.query_id.asc(), Citation.rank.asc())
        ).scalars().all()
        citations_by_query: dict[str, list[Any]] = {query_id: [] for query_id in query_ids}
        for citation in citations:
            citations_by_query[citation.query_id].append(citation)

        answers = []
        for query in query_rows:
            sources = [
                {
                    "page": int(citation.page_number),
                    "chunk_id": int(citation.chunk_id),
                    "excerpt": citation.excerpt,
                }
                for citation in citations_by_query[query.id]
                if citation.page_number is not None and citation.chunk_id is not None
            ]
            answers.append(
                {
                    "question": query.question,
                    "answer": query.answer,
                    "status": query.status,
                    "sources": sources,
                    "claim_verifications": query.claim_verifications_json or [],
                }
            )

        return RecoveredUploadResult(
            document_id=document.id,
            response_payload={"answers": answers},
        )
