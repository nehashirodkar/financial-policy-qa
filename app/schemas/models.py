from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="User question")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of docs to retrieve")
    doc_type: Optional[str] = Field(None, description="Filter by document type e.g. 'policy', 'regulation'")


class SourceDocument(BaseModel):
    doc_id: str
    text: str
    score: float
    metadata: dict


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceDocument]
    latency_ms: float
    token_cost: dict
    hallucination_score: Optional[float] = Field(
        None,
        description="Faithfulness score 0.0–1.0 (higher = more grounded). -1.0 = scoring unavailable.",
    )


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
