from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class HistoryDocumentItem(BaseModel):
    id: str
    filename: Optional[str]
    source_type: str
    source_url: Optional[str]
    status: str
    created_at: datetime
    chunk_count: int
    query_count: int


class HistoryDocumentsResponse(BaseModel):
    documents: List[HistoryDocumentItem]


class HistoryQueryItem(BaseModel):
    id: str
    question: str
    answer: str
    is_abstained: bool
    status: str
    latency_ms: Optional[float]
    created_at: datetime


class HistoryQueriesResponse(BaseModel):
    queries: List[HistoryQueryItem]


class HistoryCitationItem(BaseModel):
    rank: int
    page_number: Optional[int]
    excerpt: str
    chunk_id: Optional[int]


class HistoryCitationsResponse(BaseModel):
    citations: List[HistoryCitationItem]
