from __future__ import annotations
from pathlib import Path
from langgraph.graph import StateGraph
from rag_pipeline.graph_state import GraphState
from rag_pipeline import nodes, config
from typing import List


def build_graph(
    pdf_path: Path | None = None,
    img_path: Path | None = None,
    retrieval_type: str | None = None,
    hybrid_weights: List[float] | None = None,
):
    g = StateGraph(GraphState)

    # Routing function for complexity
    def route_complexity(state: GraphState) -> str:
        """Route based on query complexity"""
        decision = state.get("next", "simple")
        return decision

    # 1. Entry point: Extract variables
    g.add_node("extract_variables", nodes.node_extract_variables)

    # 2. Complexity check
    g.add_node("complexity_check", nodes.node_simple_or_not)

    # 3. Query expansion retrieval (used for both simple and complex)
    g.add_node("query_expansion_retrieve", nodes.node_query_expansion_retrieve)

    # 4. Simple query path
    g.add_node("simple_answer", nodes.node_simple_llm_answer)

    # 5. Complex query path

    g.add_node(
        "query_decomposition_expansion", nodes.node_query_decomposition_with_expansion
    )
    g.add_node("complex_answer", nodes.node_complex_llm_answer)

    # Set entry point
    g.set_entry_point("extract_variables")

    # Build the workflow
    g.add_edge("extract_variables", "complexity_check")

    # After complexity check, always do query expansion
    g.add_conditional_edges(
        "complexity_check",
        route_complexity,
        {
            "simple": "query_expansion_retrieve",
            "complex": "query_expansion_retrieve",
        },
    )

    # After query expansion, route to appropriate processing
    def route_after_expansion(state: GraphState) -> str:
        """Route after query expansion based on original complexity decision"""
        decision = state.get("next", "simple")
        if decision == "complex":
            return "query_decomposition_expansion"
        else:
            return "simple_answer"

    g.add_conditional_edges(
        "query_expansion_retrieve",
        route_after_expansion,
        {
            "simple_answer": "simple_answer",
            "query_decomposition_expansion": "query_decomposition_expansion",
        },
    )

    # Final edges
    g.add_edge("simple_answer", "__end__")
    g.add_edge("query_decomposition_expansion", "complex_answer")
    g.add_edge("complex_answer", "__end__")

    return g.compile()
