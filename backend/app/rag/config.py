"""Shared configuration used by synchronous and queue-driven ingestion."""

from __future__ import annotations

import logging
import os


logger = logging.getLogger(__name__)

DEFAULT_MAX_PDF_BYTES = 15_728_640
DEFAULT_EMBEDDING_MODEL_NAME = "intfloat/e5-small-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_RETRIEVAL_K_INITIAL = 20
DEFAULT_RETRIEVAL_K_FINAL = 8
DEFAULT_RETRIEVAL_MODE = "faiss_reranker"
DEFAULT_RERANKER_MODEL_NAME = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
_invalid_integer_warnings: set[tuple[str, str]] = set()


def _positive_integer_setting(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        parsed_value = int(raw_value)
    except ValueError:
        parsed_value = 0
    if parsed_value <= 0:
        warning_key = (name, raw_value)
        if warning_key not in _invalid_integer_warnings:
            _invalid_integer_warnings.add(warning_key)
            logger.warning("Invalid %s=%r; using default %s.", name, raw_value, default)
        return default
    return parsed_value


def get_max_pdf_bytes() -> int:
    return _positive_integer_setting("MAX_PDF_BYTES", DEFAULT_MAX_PDF_BYTES)


def get_embedding_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL_NAME") or DEFAULT_EMBEDDING_MODEL_NAME


def get_chunk_size() -> int:
    return DEFAULT_CHUNK_SIZE


def get_chunk_overlap() -> int:
    return DEFAULT_CHUNK_OVERLAP


def get_retrieval_k_initial() -> int:
    return _positive_integer_setting("RETRIEVAL_K_INITIAL", DEFAULT_RETRIEVAL_K_INITIAL)


def get_retrieval_k_final() -> int:
    return _positive_integer_setting("RETRIEVAL_K_FINAL", DEFAULT_RETRIEVAL_K_FINAL)


def get_retrieval_mode() -> str:
    selected = (os.getenv("RETRIEVAL_MODE") or DEFAULT_RETRIEVAL_MODE).strip().lower()
    return "faiss_reranker" if selected == "e5" else selected


def get_reranker_model_name() -> str:
    return os.getenv("RERANKER_MODEL_NAME") or DEFAULT_RERANKER_MODEL_NAME
