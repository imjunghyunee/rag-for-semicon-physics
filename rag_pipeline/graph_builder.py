from __future__ import annotations
from pathlib import Path
from langgraph.graph import StateGraph
from rag_pipeline.graph_state import GraphState
from rag_pipeline import nodes
from typing import List


# from langgraph.checkpoint.memory import MemorySaver


def build_graph(
    pdf_path: Path | None = None,
    retrieval_type: str | None = None,
    hybrid_weights: List[float] | None = None,
):
    g = StateGraph(GraphState)

    init_state = {}
    if hybrid_weights:
        init_state["hybrid_weights"] = hybrid_weights

    # Nodes
    if pdf_path:
        g.add_node(
            "retrieve", lambda s: nodes.node_retrieve_file_embedding(s, pdf_path)
        )
    elif retrieval_type == "hyde" and hybrid_weights:
        g.add_node("retrieve", nodes.node_retrieve_hyde_hybrid)
    elif retrieval_type == "hyde":
        g.add_node("retrieve", nodes.node_retrieve_hyde)
    elif retrieval_type == "summary" and hybrid_weights:
        g.add_node("retrieve", nodes.node_retrieve_summary_hybrid)
    elif retrieval_type == "summary":
        g.add_node("retrieve", nodes.node_retrieve_summary)
    elif hybrid_weights:
        g.add_node("retrieve", nodes.node_retrieve_hybrid)
    else:
        g.add_node("retrieve", nodes.node_retrieve)

    g.add_node("relevance_check", nodes.node_relevance_check)
    g.add_node("llm_answer", nodes.node_llm_answer)

    # Edges
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "relevance_check")
    g.add_edge("relevance_check", "llm_answer")
    g.add_edge("llm_answer", END := "__end__")

    return g.compile()
