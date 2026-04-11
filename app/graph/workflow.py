"""
LangGraph multi-agent workflow.

Flow:
    [START]
       │
       ▼
  retrieval_node          ← hybrid Pinecone search (dense + sparse)
       │
       ▼
 summarization_node       ← Bedrock Claude generates grounded answer
       │
       ▼
  validation_node         ← PII redaction + hallucination scoring
       │
     [END]
"""

from langgraph.graph import StateGraph, END
from app.graph.state import AgentState
from app.agents.retrieval_agent import retrieval_agent
from app.agents.summarization_agent import summarization_agent
from app.agents.validation_agent import validation_agent


def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("retrieval", retrieval_agent)
    workflow.add_node("summarization", summarization_agent)
    workflow.add_node("validation", validation_agent)

    # Sequential edges
    workflow.set_entry_point("retrieval")
    workflow.add_edge("retrieval", "summarization")
    workflow.add_edge("summarization", "validation")
    workflow.add_edge("validation", END)

    return workflow.compile()
