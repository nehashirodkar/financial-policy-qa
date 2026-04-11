"""
Locust load test for the Financial Policy Q&A API.

Validates sub-2-second p95 response time under 50 concurrent users.

Usage:
    # Headless run — 50 users, 5 spawn/sec, 2-minute duration
    locust -f scripts/load_test.py --headless \
        -u 50 -r 5 --run-time 2m \
        --host http://localhost:8000

    # Or open the web UI (default: http://localhost:8089):
    locust -f scripts/load_test.py --host http://localhost:8000

Pass TARGET_HOST env var to override --host for CI:
    TARGET_HOST=https://<api-id>.execute-api.us-east-1.amazonaws.com/Prod \
        locust -f scripts/load_test.py --headless -u 50 -r 5 --run-time 2m
"""

import os
import random

from locust import HttpUser, task, between, events

# ---------------------------------------------------------------------------
# Representative financial-policy queries drawn from the golden set
# ---------------------------------------------------------------------------
_SAMPLE_QUERIES = [
    {"query": "What is the maximum loan-to-value ratio for residential mortgages?", "doc_type": "policy"},
    {"query": "How does the policy define Tier 1 capital?", "doc_type": "regulation"},
    {"query": "What are the reporting requirements under Basel III?", "doc_type": "regulation"},
    {"query": "Describe the insider trading prohibition clauses.", "doc_type": "policy"},
    {"query": "What is the penalty for late submission of quarterly reports?", "doc_type": "regulation"},
    {"query": "How are credit derivatives valued under current policy?", "doc_type": "policy"},
    {"query": "What disclosures are required for related-party transactions?", "doc_type": "report"},
    {"query": "Define the stress-testing methodology for market risk.", "doc_type": "regulation"},
    {"query": "What are the retention requirements for trading records?", "doc_type": "policy"},
    {"query": "Summarise the anti-money-laundering obligations for broker-dealers.", "doc_type": "regulation"},
]


class FinancialQAUser(HttpUser):
    """Simulates a single concurrent user hitting the /query endpoint."""

    # Each task waits 1-3 s between requests to model realistic think time
    wait_time = between(1, 3)

    @task
    def query(self):
        payload = random.choice(_SAMPLE_QUERIES)
        with self.client.post(
            "/query",
            json=payload,
            catch_response=True,
            name="/query",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
                return

            data = resp.json()
            latency_ms = data.get("latency_ms", 0)

            # Fail the sample if server-reported latency exceeds 2 000 ms
            if latency_ms > 2000:
                resp.failure(f"p95 SLO breach: server latency {latency_ms:.0f} ms > 2000 ms")
            else:
                resp.success()

    @task(weight=1)
    def health(self):
        """Keep the health check in the mix to ensure Lambda stays warm."""
        self.client.get("/health", name="/health")


# ---------------------------------------------------------------------------
# CI exit-code hook: fail the run if p95 > 2 000 ms or error rate > 1 %
# ---------------------------------------------------------------------------
@events.quitting.add_listener
def assert_slo(environment, **_kwargs):
    stats = environment.stats.total
    p95 = stats.get_response_time_percentile(0.95)
    error_rate = stats.fail_ratio * 100

    print(f"\n[SLO check] p95={p95:.0f} ms | error_rate={error_rate:.2f}%")

    if p95 > 2000:
        print(f"[FAIL] p95 {p95:.0f} ms exceeds 2 000 ms SLO")
        environment.process_exit_code = 1
    elif error_rate > 1.0:
        print(f"[FAIL] Error rate {error_rate:.2f}% exceeds 1% threshold")
        environment.process_exit_code = 1
    else:
        print("[PASS] All SLOs met")
        environment.process_exit_code = 0
