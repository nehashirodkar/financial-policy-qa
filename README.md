<div align="center">

# Financial Policy Q&A

**A multi-agent RAG system that answers questions about financial policy & regulatory documents — with hybrid retrieval, PII guardrails, live faithfulness scoring, and a single-page demo.**

<br/>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-1C3C3C)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=langchain&logoColor=white)
![Claude](https://img.shields.io/badge/Claude-Sonnet%204-D4A27F?logo=anthropic&logoColor=white)
![Bedrock](https://img.shields.io/badge/AWS-Bedrock-FF9900?logo=amazonaws&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-hybrid%20search-1B17F4)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/tests-11%20passing-2ea44f)
![License](https://img.shields.io/badge/license-MIT-blue)

[Architecture](#-architecture) · [Demo](#-demo) · [Quickstart](#-quickstart) · [Results](#-measured-results) · [Guardrails](#-guardrails--security)

<br/>

<img src="demo.gif" alt="Financial Policy Q&A demo — ask a policy question, get a grounded answer with cited sources, faithfulness score, latency and cost" width="100%" />

<sub>Live agent: retrieve policy chunks → generate a grounded answer → redact PII → score faithfulness — with cited sources, latency, and cost.</sub>

</div>

---

## ⟡ What it is

**Financial Policy Q&A** is a production-grade RAG system for answering questions about financial policy and regulatory filings. A **LangGraph** multi-agent workflow (Claude Sonnet 4 on AWS Bedrock) retrieves relevant document chunks via **hybrid dense + sparse search**, generates a context-only grounded answer, redacts PII, and computes a **0–1 faithfulness score** by re-asking Claude how grounded the answer is in the source documents.

> **Status: complete + working demo.** 3-agent LangGraph workflow, hybrid Pinecone retrieval, PII guardrails, LangSmith tracing, RAGAS quality gate in CI, and a single-page UI. 11 automated tests; agent + demo live-validated end-to-end.

## ⟡ Architecture

```
                       ┌──────────────────┐
        user query ──► │  FastAPI / Lambda │
                       └────────┬─────────┘
                                │
                       ┌────────▼─────────┐
                       │  LangGraph state  │
                       │  machine          │
                       └────────┬──────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
       ┌────────────┐   ┌──────────────┐   ┌────────────┐
       │ Retrieval  │──►│ Summarization│──►│ Validation │
       │ (hybrid)   │   │ (Bedrock)    │   │ (PII +     │
       │            │   │              │   │  hallucin.)│
       └─────┬──────┘   └──────┬───────┘   └─────┬──────┘
             │                 │                 │
             ▼                 ▼                 ▼
       ┌──────────┐      ┌──────────┐      ┌──────────┐
       │ Pinecone │      │ Bedrock  │      │ LangSmith│
       │ (dense + │      │ Claude 4 │      │ tracing  │
       │   BM25)  │      │          │      │          │
       └──────────┘      └──────────┘      └──────────┘
```

## ⟡ Demo

A single-page UI (no build step): type a policy question and watch the agent **retrieve → answer → redact → score** in real time, with cited sources and full cost/latency transparency.

```powershell
venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
# open http://127.0.0.1:8000   (any free port works; matches the demo gif)
```

| Query | Behavior |
|---|---|
| *"What is the minimum Liquidity Coverage Ratio required?"* | grounded answer + cited LCR clause, high faithfulness |
| *"How is Tier 1 capital defined under the regulatory framework?"* | multi-source synthesis from the capital-requirements docs |
| *"What is the company's vacation policy?"* | correctly declines — not in the document corpus (anti-hallucination) |

## ⟡ The three agents

| Agent | What it does | Implementation |
|------|--------------|----------------|
| `Retrieval` | Returns the top-K most relevant chunks for a query | **Hybrid** Pinecone search — dense (Bedrock Titan embeddings) + sparse (BM25), fused with Reciprocal Rank Fusion; document-type + chunking-strategy aware |
| `Summarization` | Generates a grounded answer from retrieved context | Bedrock Claude Sonnet 4 with a **strict context-only** system prompt — refuses to answer beyond the documents |
| `Validation` | Redacts PII and scores answer faithfulness | Regex PII redaction (email, SSN, phone, card) + a **0–1 hallucination score** by re-asking Claude how grounded the answer is in the sources |

Every node is traced in LangSmith with per-node latency and token cost.

## ⟡ Quickstart

```powershell
# 1. Install
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Configure secrets (never committed)
copy .env.example .env
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY   (Bedrock — required)
#   PINECONE_API_KEY                            (required)
#   LANGCHAIN_API_KEY                           (optional — tracing only)

# 3. Ingest  ·  4. Demo  ·  5. Test
venv\Scripts\python.exe scripts\ingest.py
venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
venv\Scripts\python.exe -m pytest tests/ -v
```

```bash
# Query the API directly
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the minimum Liquidity Coverage Ratio required?"}'
```

## ⟡ Measured results

- **62% lower latency** — mean response time **5.1s → 1.9s** vs. a single-chain baseline, via parallelized retrieval and a leaner agent graph.
- **96% answer relevancy · 100% faithfulness** — RAGAS methodology implemented as Claude-as-judge over the golden set.
- **~$0.004 per query** — cache-aware token accounting across the 3-agent workflow.
- **11/11 unit tests passing** — all 3 agents (mocked Bedrock/Pinecone) plus the chunking strategies.

| Configuration | Mean latency | Answer relevancy |
|---|---:|---:|
| Single-chain baseline | 5.1s | — |
| Multi-agent (dense-only) | 2.4s | 0.91 |
| **Multi-agent (hybrid + RRF)** | **1.9s** | **0.96** |

## ⟡ Document-aware chunking

Three configurable strategies (benchmarked via `scripts/compare_chunking.py`):

| Strategy | When to use |
|---|---|
| **fixed-size** | Fast baseline; uniform chunks |
| **hierarchical** | Default; respects headings → paragraphs → sentences |
| **sentence-window** | Highest precision; each chunk = sentence + N neighbors |

Chunk sizes adapt per document type: `regulation` 256 tokens · `policy` 512 · `report` 1024.

## ⟡ Guardrails & security

| Concern | Mitigation |
|---|---|
| PII leakage in answers | Regex redaction of email, SSN, phone, and card numbers in the Validation agent before any response is returned |
| Hallucination / ungrounded claims | Strict context-only prompt **plus** a 0–1 faithfulness score; low-grounding answers are flagged |
| Quality regressions on prompt/retrieval changes | CI **RAGAS quality gate** — PR is blocked if `answer_relevancy < 0.80` or `faithfulness < 0.85` |
| Secrets in repo | All keys read from `.env` (git-ignored); `.env.example` ships non-secret placeholders only |

## ⟡ Project structure

```
app/
├── agents/          # Retrieval, summarization, validation specialists
├── graph/           # LangGraph state schema + workflow
├── retrieval/       # Pinecone client, hybrid search, chunking strategies
├── guardrails/      # PII redaction
├── tracing/         # LangSmith integration, hallucination scoring
├── schemas/         # Pydantic models
├── static/          # Single-page demo UI (served at /)
└── main.py          # FastAPI app + Mangum Lambda handler

scripts/
├── ingest.py            # Ingest documents to Pinecone
├── baseline.py          # Single-chain (no multi-agent) baseline
├── benchmark.py         # Latency comparison
├── compare_retrieval.py # Dense vs hybrid quality comparison
├── compare_chunking.py  # Chunking strategy comparison
└── load_test.py         # Locust load test

tests/                   # 11 PyTest cases (agents + chunking)
.github/workflows/       # CI: pytest + RAGAS quality gate
```

## ⟡ Deployment

| Target | How |
|---|---|
| **Local** | `uvicorn app.main:app` → web UI + API on `:8000` |
| **Docker** | `docker compose up` → containerized API on `:8000` |
| **AWS Lambda** | `sam build && sam deploy --guided` (`template.yaml`, Mangum adapter) |
| **AWS ECS** | `deploy/ecs-task-definition.json` + `deploy/ecs-autoscaling.json` |

## ⟡ Benchmarks

```bash
python scripts/benchmark.py --n 5            # latency: multi-agent vs baseline
python scripts/compare_retrieval.py --n 5    # retrieval: dense vs hybrid
python scripts/compare_chunking.py --n 5     # chunking strategy comparison
locust -f scripts/load_test.py --headless -u 10 -r 2 --run-time 30s --host http://127.0.0.1:8000
```

## License

MIT
