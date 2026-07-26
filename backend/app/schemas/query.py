from typing import Any, List, Literal

from pydantic import BaseModel, Field, HttpUrl


class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[Any]


class SourceReference(BaseModel):
    page: int
    chunk_id: int
    excerpt: str


class ClaimVerificationSource(BaseModel):
    page: int
    chunk_id: int
    excerpt: str


class ClaimVerificationItem(BaseModel):
    claim: str
    verdict: Literal["supported", "weakly_supported", "unsupported"]
    rationale: str
    sources: List[ClaimVerificationSource]


class AnswerItem(BaseModel):
    question: str
    answer: str
    status: str
    sources: List[SourceReference]
    claim_verifications: List[ClaimVerificationItem] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answers: List[AnswerItem]
