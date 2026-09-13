"""Shared configuration used by synchronous and queue-driven ingestion."""

from __future__ import annotations

import logging
import os


logger = logging.getLogger(__name__)

DEFAULT_MAX_PDF_BYTES = 15_728_640
DEFAULT_EMBEDDING_MODEL_NAME = "intfloat/e5-small-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
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
