"""Safe request/response models for asynchronous document ingestion APIs."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel


DocumentLifecycleStatus = Literal["queued", "processing", "ready", "failed"]


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: Optional[str]
    status: DocumentLifecycleStatus


class DuplicateDocumentResponse(BaseModel):
    code: Literal["duplicate_document"] = "duplicate_document"
    document: DocumentUploadResponse


class DocumentStatusResponse(DocumentUploadResponse):
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime


class DocumentUploadUnavailableResponse(BaseModel):
    document_id: Optional[str] = None
    status: Optional[DocumentLifecycleStatus] = None
    message: str


class DocumentQueryRequest(BaseModel):
    question: str
