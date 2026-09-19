"""Application orchestration for private, asynchronously ingested PDF uploads."""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from backend.app.rag.config import (
    get_embedding_model_name,
    get_max_pdf_bytes,
    get_reranker_model_name,
    get_retrieval_k_final,
    get_retrieval_k_initial,
    get_retrieval_mode,
)
from backend.app.rag.embeddings import get_embedding_input_format_version
from backend.app.services.document_ingestion_queue import enqueue_document_ingestion
from backend.app.storage import ObjectStorage, document_pdf_object_key, get_object_storage
from persistence import document_uploads as upload_persistence
from persistence.document_ingestion import (
    DOCUMENT_STATUS_FAILED,
    DOCUMENT_STATUS_PROCESSING,
    DOCUMENT_STATUS_QUEUED,
    DOCUMENT_STATUS_READY,
)


logger = logging.getLogger(__name__)
SUPPORTED_UPLOAD_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/octet-stream",
}
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]+")


class InvalidDocumentUploadError(ValueError):
    """Raised before any upload side effect when request input is invalid."""


class DocumentUploadConflictError(ValueError):
    """Raised when an upload request UUID is reused for different PDF bytes."""


class DocumentUploadUnavailableError(RuntimeError):
    """Safe infrastructure error that may include an authoritative document ID."""

    def __init__(
        self,
        message: str,
        *,
        document_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.document_id = document_id
        self.status = status


@dataclass(frozen=True)
class DocumentUploadResult:
    document: upload_persistence.StoredDocumentStatus
    recovered: bool

    @property
    def http_status_code(self) -> int:
        if self.recovered and self.document.status in {
            DOCUMENT_STATUS_READY,
            DOCUMENT_STATUS_FAILED,
        }:
            return 200
        return 202


def normalize_upload_request_id(value: Optional[str]) -> Optional[str]:
    normalized = (value or "").strip()
    if not normalized:
        return None
    try:
        return str(uuid.UUID(normalized))
    except ValueError as exc:
        raise InvalidDocumentUploadError(
            "upload_request_id must be a valid UUID."
        ) from exc


def normalize_pdf_filename(filename: Optional[str]) -> str:
    raw_filename = (filename or "").replace("\\", "/").split("/")[-1]
    normalized = " ".join(_CONTROL_CHARACTERS.sub(" ", raw_filename).split()).strip()
    if not normalized or not normalized.lower().endswith(".pdf"):
        raise InvalidDocumentUploadError("Uploaded filename must end with .pdf.")
    if len(normalized) > 512:
        normalized = f"{normalized[:508]}.pdf"
    return normalized


def validate_pdf_upload(
    *,
    filename: Optional[str],
    content_type: Optional[str],
    pdf_bytes: bytes,
) -> str:
    normalized_filename = normalize_pdf_filename(filename)
    normalized_content_type = (content_type or "").split(";", 1)[0].strip().lower()
    if normalized_content_type and normalized_content_type not in SUPPORTED_UPLOAD_CONTENT_TYPES:
        raise InvalidDocumentUploadError("Uploaded content type must identify a PDF.")
    if not pdf_bytes:
        raise InvalidDocumentUploadError("Uploaded PDF is empty.")
    if len(pdf_bytes) > get_max_pdf_bytes():
        raise InvalidDocumentUploadError("PDF exceeds maximum allowed size.")
    if not pdf_bytes.startswith(b"%PDF-"):
        raise InvalidDocumentUploadError("Uploaded file does not have a valid PDF signature.")
    return normalized_filename


def _delete_new_object_best_effort(storage: ObjectStorage, object_key: str) -> None:
    try:
        storage.delete(object_key)
    except Exception as exc:
        logger.warning(
            "Document upload compensation could not delete a newly written object "
            "(error_type=%s).",
            type(exc).__name__,
        )


def _publish_if_queued(
    document: upload_persistence.StoredDocumentStatus,
    *,
    enqueue: Callable[[str], str],
) -> None:
    if document.status != DOCUMENT_STATUS_QUEUED:
        return
    try:
        enqueue(document.document_id)
    except Exception as exc:
        # Publisher-confirmation failures can be ambiguous. The row must stay
        # queued so an accepted message may still be processed and an
        # idempotent client retry may safely publish another delivery.
        logger.warning(
            "Document task publication failed safely document_id=%s error_type=%s.",
            document.document_id,
            type(exc).__name__,
        )
        raise DocumentUploadUnavailableError(
            "Document was stored but task publication could not be confirmed.",
            document_id=document.document_id,
            status=DOCUMENT_STATUS_QUEUED,
        ) from exc


def _load_current_document(
    *,
    user_id: str,
    document: upload_persistence.StoredDocumentStatus,
) -> upload_persistence.StoredDocumentStatus:
    try:
        return upload_persistence.should_enqueue_owned_document(
            user_id=user_id,
            document_id=document.document_id,
        )
    except Exception as exc:
        raise DocumentUploadUnavailableError(
            "Document status is temporarily unavailable.",
            document_id=document.document_id,
            status=document.status,
        ) from exc


def create_document_upload(
    *,
    user_id: str,
    filename: Optional[str],
    content_type: Optional[str],
    pdf_bytes: bytes,
    upload_request_id: Optional[str],
    object_storage: Optional[ObjectStorage] = None,
    enqueue: Callable[[str], str] = enqueue_document_ingestion,
) -> DocumentUploadResult:
    """Validate, store, persist, and publish one owned PDF document."""
    normalized_filename = validate_pdf_upload(
        filename=filename,
        content_type=content_type,
        pdf_bytes=pdf_bytes,
    )
    normalized_request_id = normalize_upload_request_id(upload_request_id)
    source_hash = hashlib.sha256(pdf_bytes).hexdigest()

    if normalized_request_id is not None:
        try:
            recovered = upload_persistence.recover_owned_upload_request(
                user_id=user_id,
                upload_request_id=normalized_request_id,
                source_hash=source_hash,
            )
        except upload_persistence.UploadRequestConflictError as exc:
            raise DocumentUploadConflictError(str(exc)) from exc
        except Exception as exc:
            raise DocumentUploadUnavailableError(
                "Document upload recovery is temporarily unavailable."
            ) from exc
        if recovered is not None:
            current = _load_current_document(
                user_id=user_id,
                document=recovered,
            )
            _publish_if_queued(current, enqueue=enqueue)
            return DocumentUploadResult(document=current, recovered=True)

    document_id = str(uuid.uuid4())
    object_key = document_pdf_object_key(document_id)
    storage = object_storage if object_storage is not None else get_object_storage()
    try:
        storage.put(object_key, pdf_bytes, content_type="application/pdf")
    except Exception as exc:
        raise DocumentUploadUnavailableError(
            "Private document storage is temporarily unavailable."
        ) from exc

    model_name = get_embedding_model_name()
    try:
        write_result = upload_persistence.create_or_recover_queued_upload(
            user_id=user_id,
            document_id=document_id,
            upload_request_id=normalized_request_id,
            filename=normalized_filename,
            source_hash=source_hash,
            object_key=object_key,
            content_type="application/pdf",
            byte_size=len(pdf_bytes),
            embedding_model=model_name,
            embedding_format=get_embedding_input_format_version(model_name),
            retrieval_mode=get_retrieval_mode(),
            reranker_model=get_reranker_model_name(),
            k_initial=get_retrieval_k_initial(),
            k_final=get_retrieval_k_final(),
        )
    except upload_persistence.UploadRequestConflictError as exc:
        _delete_new_object_best_effort(storage, object_key)
        raise DocumentUploadConflictError(str(exc)) from exc
    except Exception as exc:
        _delete_new_object_best_effort(storage, object_key)
        raise DocumentUploadUnavailableError(
            "Document metadata could not be stored."
        ) from exc

    if not write_result.created:
        _delete_new_object_best_effort(storage, object_key)

    current = _load_current_document(
        user_id=user_id,
        document=write_result.document,
    )
    _publish_if_queued(current, enqueue=enqueue)
    return DocumentUploadResult(document=current, recovered=not write_result.created)


def get_document_status(*, user_id: str, document_id: str) -> upload_persistence.StoredDocumentStatus:
    return upload_persistence.get_owned_document_status(
        user_id=user_id,
        document_id=document_id,
    )


__all__ = [
    "DocumentUploadConflictError",
    "DocumentUploadResult",
    "DocumentUploadUnavailableError",
    "InvalidDocumentUploadError",
    "create_document_upload",
    "get_document_status",
    "normalize_upload_request_id",
    "validate_pdf_upload",
]
