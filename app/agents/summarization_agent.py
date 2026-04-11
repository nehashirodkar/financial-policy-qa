"""
Summarization specialist agent.

Takes retrieved documents and the original query, then uses Claude via
AWS Bedrock to produce a grounded answer.
"""

import json
import logging

import boto3

from app.config import config
from app.tracing.langsmith import trace_node, compute_token_cost

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a financial policy analyst. Answer the user's question
using ONLY the provided context documents. Be precise and cite the document IDs.
If the answer cannot be determined from the context, say so explicitly.
Respond in plain English — no markdown."""


def _build_prompt(query: str, docs: list[dict]) -> str:
    context = "\n\n".join(
        f"[DOC {i+1} | id={d['doc_id']} | score={d['score']}]\n{d['text']}"
        for i, d in enumerate(docs)
    )
    return f"Context:\n{context}\n\nQuestion: {query}"


@trace_node("summarization_agent")
def summarization_agent(state: dict) -> dict:
    """
    Graph node: generates a grounded answer from retrieved docs.

    Input state keys : query, retrieved_docs
    Output state keys: raw_answer, token_cost
    """
    query = state["query"]
    docs = state.get("retrieved_docs", [])

    if not docs:
        return {"raw_answer": "No relevant documents found.", "token_cost": {}}

    client = boto3.client(
        "bedrock-runtime",
        region_name=config.AWS_REGION,
        aws_access_key_id=config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
    )

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": _build_prompt(query, docs)}],
    }

    response = client.invoke_model(
        modelId=config.BEDROCK_MODEL_ID,
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json",
    )

    body = json.loads(response["body"].read())
    raw_answer = body["content"][0]["text"]
    usage = body.get("usage", {})

    token_cost = compute_token_cost(
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        model=config.BEDROCK_MODEL_ID,
    )

    logger.info("Answer generated (%d output tokens)", usage.get("output_tokens", 0))
    return {"raw_answer": raw_answer, "token_cost": token_cost}
