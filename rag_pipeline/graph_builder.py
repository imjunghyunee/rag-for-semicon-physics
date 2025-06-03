from __future__ import annotations
from pathlib import Path
from langgraph.graph import StateGraph
from rag_pipeline.graph_state import GraphState
from rag_pipeline import nodes
from typing import List


def build_graph(
    pdf_path: Path | None = None,
    img_path: Path | None = None,
    retrieval_type: str | None = None,
    hybrid_weights: List[float] | None = None,
):
    g = StateGraph(GraphState)

    # Simple/Complex routing function
    def route_complexity(state: GraphState) -> str:
        """Route based on query complexity"""
        decision = state.get("next", "simple")
        return decision

    # Nodes
    g.add_node("complexity_check", nodes.node_simple_or_not)

    # Simple query processing nodes
    if pdf_path:
        g.add_node(
            "retrieve_simple", lambda s: nodes.node_retrieve_file_embedding(s, pdf_path)
        )
    elif img_path:
        g.add_node(
            "retrieve_simple", lambda s: nodes.node_retrieve_img_embedding(s, img_path)
        )
    elif retrieval_type == "hyde" and hybrid_weights:
        g.add_node("retrieve_simple", nodes.node_retrieve_hyde_hybrid)
    elif retrieval_type == "hyde":
        g.add_node("retrieve_simple", nodes.node_retrieve_hyde)
    elif retrieval_type == "summary" and hybrid_weights:
        g.add_node("retrieve_simple", nodes.node_retrieve_summary_hybrid)
    elif retrieval_type == "summary":
        g.add_node("retrieve_simple", nodes.node_retrieve_summary)
    elif hybrid_weights:
        g.add_node("retrieve_simple", nodes.node_retrieve_hybrid)
    else:
        g.add_node("retrieve_simple", nodes.node_retrieve)

    g.add_node("relevance_check", nodes.node_relevance_check)
    g.add_node("llm_answer_simple", nodes.node_llm_answer)

    # Complex query processing nodes - 단일 노드로 단순화
    g.add_node("plan_and_execute", nodes.node_plan_and_execute)

    # Entry point
    g.set_entry_point("complexity_check")

    # Conditional edges based on complexity
    g.add_conditional_edges(
        "complexity_check",
        route_complexity,
        {
            "simple": "retrieve_simple",
            "complex": "plan_and_execute",  # 🔥 직접 종료
        },
    )

    # Simple query path
    g.add_edge("retrieve_simple", "relevance_check")
    g.add_edge("relevance_check", "llm_answer_simple")
    g.add_edge("llm_answer_simple", "__end__")

    # Complex query path - 🔥 바로 종료
    g.add_edge("plan_and_execute", "__end__")

    return g.compile()
