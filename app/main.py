"""
FastAPI application entry point.

Deployed on AWS Lambda + API Gateway via Mangum adapter.
"""

import time
import logging

from fastapi import FastAPI, HTTPException
from mangum import Mangum

from app.schemas.models import QueryRequest, QueryResponse, HealthResponse, SourceDocument
from app.agents.coordinator import run_query
from app.config import config

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial Policy Q&A API",
    description="Multi-agent RAG system for financial policy documents.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    t0 = time.perf_counter()

    try:
        result = run_query(
            query=request.query,
            top_k=request.top_k or 5,
            doc_type=request.doc_type,
        )
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    sources = [SourceDocument(**d) for d in result.get("retrieved_docs", [])]

    return QueryResponse(
        query=request.query,
        answer=result.get("validated_answer", ""),
        sources=sources,
        latency_ms=latency_ms,
        token_cost=result.get("token_cost", {}),
        hallucination_score=result.get("hallucination_score", -1.0),
    )


# AWS Lambda handler
handler = Mangum(app)
