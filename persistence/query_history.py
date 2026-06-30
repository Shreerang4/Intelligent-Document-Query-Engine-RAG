"""Best-effort persistence side effects for answered queries and citations."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Sequence

from persistence.user_context import get_current_user_id


logger = logging.getLogger(__name__)


def _dump_model(item: Any) -> Any:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if isinstance(item, Mapping):
        return dict(item)
    return item


def _get_mapping_value(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _persist_query_result(
    *,
    user_id: str,
    document_id: str,
    question: str,
    answer: str,
    status: str,
    is_abstained: bool,
    claim_verifications_json: Optional[list[dict[str, Any]]],
    sources: Sequence[Any],
    source_chunks: Sequence[Mapping[str, Any]],
    embedding_model: str,
    retrieval_mode: str,
    reranker_model: str,
    k_initial: int,
    k_final: int,
    latency_ms: Optional[float],
) -> Optional[str]:
    from sqlalchemy import select

    from persistence.db import SessionLocal
    from persistence.models import Chunk, Citation, Query

    source_chunk_by_id = {
        int(chunk["chunk_id"]): chunk
        for chunk in source_chunks
        if chunk.get("chunk_id") is not None
    }

    with SessionLocal() as session:
        query = Query(
            user_id=user_id,
            document_id=document_id,
            question=question,
            answer=answer,
            status=status,
            is_abstained=is_abstained,
            claim_verifications_json=claim_verifications_json,
            embedding_model=embedding_model,
            retrieval_mode=retrieval_mode,
            reranker_model=reranker_model,
            k_initial=k_initial,
            k_final=k_final,
            latency_ms=latency_ms,
        )
        session.add(query)
        session.flush()

        chunk_ids = [
            int(chunk_id)
            for source in sources
            if (chunk_id := _get_mapping_value(source, "chunk_id")) is not None
        ]
        chunk_db_ids: dict[int, str] = {}
        if chunk_ids:
            stored_chunks = session.execute(
                select(Chunk).where(
                    Chunk.document_id == document_id,
                    Chunk.chunk_id.in_(chunk_ids),
                )
            ).scalars()
            chunk_db_ids = {chunk.chunk_id: chunk.id for chunk in stored_chunks}

        citation_rows = []
        for rank, source in enumerate(sources, start=1):
            chunk_id_raw = _get_mapping_value(source, "chunk_id")
            chunk_id = int(chunk_id_raw) if chunk_id_raw is not None else None
            raw_chunk = source_chunk_by_id.get(chunk_id) if chunk_id is not None else None
            citation_rows.append(
                {
                    "user_id": user_id,
                    "query_id": query.id,
                    "document_id": document_id,
                    "chunk_db_id": chunk_db_ids.get(chunk_id) if chunk_id is not None else None,
                    "chunk_id": chunk_id,
                    "rank": rank,
                    "page_number": _get_mapping_value(source, "page"),
                    "excerpt": _get_mapping_value(source, "excerpt") or "",
                    "retrieval_score": raw_chunk.get("retrieval_score") if raw_chunk else None,
                    "reranker_score": raw_chunk.get("reranker_score") if raw_chunk else None,
                }
            )

        if citation_rows:
            session.bulk_insert_mappings(Citation, citation_rows)

        session.commit()
        return query.id


def persist_query_result_best_effort(
    *,
    document_id: Optional[str],
    question: str,
    answer: str,
    status: str,
    is_abstained: bool,
    claim_verifications: Sequence[Any],
    sources: Sequence[Any],
    source_chunks: Sequence[Mapping[str, Any]],
    embedding_model: str,
    retrieval_mode: str,
    reranker_model: str,
    k_initial: int,
    k_final: int,
    latency_ms: Optional[float],
) -> Optional[str]:
    """Persist query/citation history without letting DB failures affect answers."""
    if not document_id:
        return None

    try:
        claim_payload = [_dump_model(item) for item in claim_verifications] if claim_verifications else None
        return _persist_query_result(
            user_id=get_current_user_id(),
            document_id=document_id,
            question=question,
            answer=answer,
            status=status,
            is_abstained=is_abstained,
            claim_verifications_json=claim_payload,
            sources=sources,
            source_chunks=source_chunks,
            embedding_model=embedding_model,
            retrieval_mode=retrieval_mode,
            reranker_model=reranker_model,
            k_initial=k_initial,
            k_final=k_final,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        logger.warning("Persistence query side effect failed safely: %s", type(exc).__name__)
        return None
