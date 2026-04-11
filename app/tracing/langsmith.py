"""
LangSmith instrumentation helpers.

Sets up tracing for per-node latency, token cost, and hallucination rate.
"""

import json
import os
import logging
from functools import wraps
from time import perf_counter

import boto3

from app.config import config

logger = logging.getLogger(__name__)

# Configure LangSmith env vars before any langchain import
os.environ["LANGCHAIN_TRACING_V2"] = config.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = config.LANGCHAIN_PROJECT

_HALLUCINATION_PROMPT = """\
You are a strict faithfulness evaluator.

Given the CONTEXT (retrieved document excerpts) and an ANSWER, score how well
every claim in the ANSWER is supported by the CONTEXT.

Score rules:
  1.0  — every claim is directly supported by the context
  0.5  — some claims are supported, others are not verifiable from the context
  0.0  — most claims contradict or are absent from the context

Return ONLY a JSON object with one key: {{"faithfulness": <float 0.0-1.0>}}

CONTEXT:
{context}

ANSWER:
{answer}"""


def compute_hallucination_score(answer: str, docs: list[dict]) -> float:
    """
    Ask Claude via Bedrock to score answer faithfulness against retrieved docs.
    Returns a float in [0.0, 1.0]. Higher = more grounded.
    Falls back to -1.0 on any error so callers can distinguish unknown from scored.
    """
    if not answer or not docs:
        return -1.0

    context = "\n\n".join(
        f"[DOC {i+1}] {d.get('text', '')}" for i, d in enumerate(docs)
    )
    prompt = _HALLUCINATION_PROMPT.format(context=context[:6000], answer=answer)

    try:
        client = boto3.client(
            "bedrock-runtime",
            region_name=config.AWS_REGION,
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
        )
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = client.invoke_model(
            modelId=config.BEDROCK_MODEL_ID,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
        body = json.loads(response["body"].read())
        text = body["content"][0]["text"].strip()
        score = float(json.loads(text)["faithfulness"])
        score = max(0.0, min(1.0, score))
        logger.info("Hallucination score: %.2f", score)
        return round(score, 4)
    except Exception as exc:
        logger.warning("Hallucination scoring failed: %s", exc)
        return -1.0


def trace_node(node_name: str):
    """
    Decorator that logs node entry/exit with latency and attaches metadata
    to the active LangSmith run via run_on_thread callbacks.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = perf_counter()
            logger.info("[%s] start", node_name)
            try:
                result = fn(*args, **kwargs)
                elapsed_ms = (perf_counter() - t0) * 1000
                logger.info("[%s] done in %.1f ms", node_name, elapsed_ms)
                # Attach latency to result dict so the graph state carries it
                if isinstance(result, dict):
                    result.setdefault("node_latencies", {})[node_name] = round(elapsed_ms, 2)
                return result
            except Exception as exc:
                elapsed_ms = (perf_counter() - t0) * 1000
                logger.error("[%s] failed after %.1f ms: %s", node_name, elapsed_ms, exc)
                raise
        return wrapper
    return decorator


def compute_token_cost(input_tokens: int, output_tokens: int, model: str) -> dict:
    """Rough cost estimate in USD for Bedrock Claude models."""
    RATES = {
        "claude-3-sonnet": {"input": 3e-6, "output": 15e-6},
        "claude-3-haiku":  {"input": 0.25e-6, "output": 1.25e-6},
    }
    key = next((k for k in RATES if k in model), "claude-3-sonnet")
    rate = RATES[key]
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(
            input_tokens * rate["input"] + output_tokens * rate["output"], 6
        ),
    }
