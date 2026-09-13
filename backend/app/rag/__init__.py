"""Shared RAG ingestion and embedding primitives."""

from backend.app.rag.config import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_MAX_PDF_BYTES,
    get_embedding_model_name,
    get_max_pdf_bytes,
)
from backend.app.rag.embeddings import (
    create_chunk_embeddings,
    format_documents_for_embedding,
    format_query_for_embedding,
    get_embedding_input_format_version,
    get_embedding_model,
    uses_e5_embedding_format,
)
from backend.app.rag.ingestion import (
    ChunkRecord,
    InvalidPdfError,
    NoMeaningfulTextError,
    PdfContentError,
    PdfTooLargeError,
    parse_and_chunk_pdf_bytes,
)

__all__ = [
    "ChunkRecord",
    "DEFAULT_EMBEDDING_MODEL_NAME",
    "DEFAULT_MAX_PDF_BYTES",
    "InvalidPdfError",
    "NoMeaningfulTextError",
    "PdfContentError",
    "PdfTooLargeError",
    "create_chunk_embeddings",
    "format_documents_for_embedding",
    "format_query_for_embedding",
    "get_embedding_input_format_version",
    "get_embedding_model",
    "get_embedding_model_name",
    "get_max_pdf_bytes",
    "parse_and_chunk_pdf_bytes",
    "uses_e5_embedding_format",
]
