"""Authenticated asynchronous document upload and lifecycle endpoints."""

from __future__ import annotations

import asyncio
from typing import Annotated, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Response, UploadFile, status
from fastapi.responses import JSONResponse

from backend.app.auth.dependencies import get_authenticated_user_id
from backend.app.rag.config import get_max_pdf_bytes
from backend.app.schemas.documents import (
    DocumentQueryRequest,
    DocumentStatusResponse,
    DocumentUploadResponse,
    DocumentUploadUnavailableResponse,
)
from backend.app.schemas.query import QueryResponse
from backend.app.services.document_upload import (
    DocumentUploadConflictError,
    DocumentUploadUnavailableError,
    InvalidDocumentUploadError,
    create_document_upload,
    get_document_status,
)
from persistence.ownership import OwnedResourceNotFoundError


router = APIRouter(prefix="/documents", tags=["documents"])


def _unavailable_response(exc: DocumentUploadUnavailableError) -> JSONResponse:
    payload = DocumentUploadUnavailableResponse(
        document_id=exc.document_id,
        status=exc.status,
        message=str(exc),
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=payload.model_dump(mode="json", exclude_none=True),
    )


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_200_OK: {"model": DocumentUploadResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": DocumentUploadUnavailableResponse,
        },
    },
)
async def upload_document(
    response: Response,
    file: Annotated[UploadFile, File(...)],
    upload_request_id: Annotated[Optional[str], Form()] = None,
    user_id: str = Depends(get_authenticated_user_id),
):
    try:
        pdf_bytes = await file.read(get_max_pdf_bytes() + 1)
    finally:
        await file.close()

    try:
        result = await asyncio.to_thread(
            create_document_upload,
            user_id=user_id,
            filename=file.filename,
            content_type=file.content_type,
            pdf_bytes=pdf_bytes,
            upload_request_id=upload_request_id,
        )
    except InvalidDocumentUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DocumentUploadConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DocumentUploadUnavailableError as exc:
        return _unavailable_response(exc)

    response.status_code = result.http_status_code
    return DocumentUploadResponse(
        document_id=result.document.document_id,
        filename=result.document.filename,
        status=result.document.status,
    )


@router.get("/{document_id}", response_model=DocumentStatusResponse)
async def read_document_status(
    document_id: str,
    user_id: str = Depends(get_authenticated_user_id),
) -> DocumentStatusResponse:
    try:
        document = await asyncio.to_thread(
            get_document_status,
            user_id=user_id,
            document_id=document_id,
        )
    except OwnedResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found.") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Document status is unavailable.") from exc

    return DocumentStatusResponse(
        document_id=document.document_id,
        filename=document.filename,
        status=document.status,
        error_message=document.error_message,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


@router.post("/{document_id}/queries", response_model=QueryResponse)
async def query_document(
    document_id: str,
    request: DocumentQueryRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_authenticated_user_id),
) -> QueryResponse:
    # main owns the shared RAG runner and process-wide FAISS cache.
    import main
    question = main._normalize_questions([request.question])[0]
    try:
        document = await asyncio.to_thread(
            get_document_status, user_id=user_id, document_id=document_id
        )
        if document.status in {"queued", "processing"}:
            raise HTTPException(status_code=409, detail="Document is not ready yet.")
        if document.status == "failed":
            raise HTTPException(status_code=409, detail="Document processing failed.")
        return await main.query_ready_document(
            document_id=document_id,
            question=question,
            user_id=user_id,
            response=response,
            background_tasks=background_tasks,
        )
    except OwnedResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found.") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Document query is unavailable.") from exc
