"""Shared E5 formatting, model loading, and document embedding logic."""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import Any, Optional, Sequence

import numpy as np

from backend.app.rag.config import get_embedding_model_name
from backend.app.rag.ingestion import ChunkRecord


logger = logging.getLogger(__name__)
_process_model_cache: dict[str, object] = {}


def uses_e5_embedding_format(model_name: Optional[str] = None) -> bool:
    selected_model = (model_name or get_embedding_model_name()).strip().lower()
    return (
        selected_model.startswith("intfloat/e5-")
        or selected_model.startswith("intfloat/multilingual-e5-")
        or "/e5-" in selected_model
        or "/multilingual-e5-" in selected_model
    )


def get_embedding_input_format_version(model_name: Optional[str] = None) -> str:
    return "e5-query-passage-v1" if uses_e5_embedding_format(model_name) else "raw-v1"


def format_documents_for_embedding(
    texts: Sequence[str],
    model_name: Optional[str] = None,
) -> list[str]:
    normalized_texts = list(texts)
    if uses_e5_embedding_format(model_name):
        return [f"passage: {text}" for text in normalized_texts]
    return normalized_texts


def format_query_for_embedding(question: str, model_name: Optional[str] = None) -> str:
    if uses_e5_embedding_format(model_name):
        return f"query: {question}"
    return question


def get_embedding_model(
    model_cache: MutableMapping[str, object] | None = None,
) -> Any:
    selected_cache = model_cache if model_cache is not None else _process_model_cache
    model_name = get_embedding_model_name()
    cache_key = f"embedding_model:{model_name}"
    if cache_key not in selected_cache:
        logger.info("Loading embedding model: %s", model_name)
        from langchain_huggingface import HuggingFaceEmbeddings

        selected_cache[cache_key] = HuggingFaceEmbeddings(model_name=model_name)
    return selected_cache[cache_key]


def create_chunk_embeddings(
    chunks: Sequence[ChunkRecord],
    embedding_model: Any,
) -> np.ndarray:
    if not chunks:
        raise ValueError("No text chunks to process.")

    model_name = get_embedding_model_name()
    chunk_texts = format_documents_for_embedding(
        [chunk["text"] for chunk in chunks],
        model_name,
    )
    chunk_embeddings = embedding_model.embed_documents(chunk_texts)
    embedding_array = np.asarray(chunk_embeddings, dtype=np.float32)
    if (
        embedding_array.ndim != 2
        or embedding_array.shape[0] != len(chunks)
        or embedding_array.shape[1] == 0
    ):
        raise RuntimeError("Embedding model returned an invalid chunk embedding matrix.")
    return np.ascontiguousarray(embedding_array, dtype=np.float32)
