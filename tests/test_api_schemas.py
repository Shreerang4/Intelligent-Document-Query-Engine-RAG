from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from backend.app.schemas import (
    AnswerItem,
    ClaimVerificationItem,
    HistoryDocumentItem,
    QueryRequest,
    QueryResponse,
)


def test_query_request_accepts_http_document_url() -> None:
    request = QueryRequest(
        documents="https://example.com/report.pdf",
        questions=["What was the revenue?"],
    )

    assert str(request.documents) == "https://example.com/report.pdf"
    assert request.questions == ["What was the revenue?"]


def test_query_request_rejects_invalid_document_url() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(documents="not-a-url", questions=["What was the revenue?"])


def test_answer_item_uses_independent_verification_lists() -> None:
    first = AnswerItem(
        question="First?",
        answer="First answer",
        status="success",
        sources=[],
    )
    second = AnswerItem(
        question="Second?",
        answer="Second answer",
        status="success",
        sources=[],
    )

    first.claim_verifications.append(
        ClaimVerificationItem(
            claim="A claim",
            verdict="supported",
            rationale="The source supports it.",
            sources=[],
        )
    )

    assert len(first.claim_verifications) == 1
    assert second.claim_verifications == []


def test_response_and_history_schemas_serialize() -> None:
    response = QueryResponse(
        answers=[
            AnswerItem(
                question="What happened?",
                answer="An event happened.",
                status="success",
                sources=[],
            )
        ]
    )
    history_item = HistoryDocumentItem(
        id="document-1",
        filename="report.pdf",
        source_type="upload",
        source_url=None,
        status="ready",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        chunk_count=12,
        query_count=3,
    )

    assert response.model_dump()["answers"][0]["status"] == "success"
    assert history_item.model_dump()["chunk_count"] == 12
