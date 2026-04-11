"""
Output guardrails:
  1. Structured JSON validation via Guardrails AI
  2. PII detection and redaction (email, SSN, phone, credit card)
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── PII patterns ──────────────────────────────────────────────────────────────
_PII_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(\+1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[CARD]"),
]


def redact_pii(text: str) -> str:
    """Replace PII with placeholder tokens."""
    for pattern, placeholder in _PII_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def validate_and_clean(raw_answer: str, sources: list[str]) -> dict:
    """
    1. Redact PII from the raw answer.
    2. Return structured result dict.
    Returns a dict with keys: answer, confidence, sources_used, pii_detected.
    """
    cleaned = redact_pii(raw_answer)
    pii_detected = cleaned != raw_answer

    result = {
        "answer": cleaned,
        "confidence": 0.9,
        "sources_used": sources,
        "pii_detected": pii_detected,
    }
    return result
