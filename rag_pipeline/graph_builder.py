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

    # Complex query processing nodes - 초기 상태 정보를 전달하는 래퍼 함수들
    # def query_decomposition_with_params(state: GraphState) -> GraphState:
    #     # 초기화 시 전달받은 파라미터들을 상태에 추가
    #     if retrieval_type and "retrieval_type" not in state:
    #         state["retrieval_type"] = retrieval_type
    #     if hybrid_weights and "hybrid_weights" not in state:
    #         state["hybrid_weights"] = hybrid_weights
    #     return nodes.node_query_decomposition(state)

    # g.add_node("query_decomposition", query_decomposition_with_params)
    g.add_node("query_decomposition", nodes.node_query_decomposition)
    g.add_node("llm_answer_complex", nodes.node_complex_llm_answer)

    # Entry point
    g.set_entry_point("complexity_check")

    # Conditional edges based on complexity
    g.add_conditional_edges(
        "complexity_check",
        route_complexity,
        {
            "simple": "retrieve_simple",
            "complex": "query_decomposition",
        },
    )

    # Simple query path
    g.add_edge("retrieve_simple", "relevance_check")
    g.add_edge("relevance_check", "llm_answer_simple")
    g.add_edge("llm_answer_simple", "__end__")

    # Complex query path
    g.add_edge("query_decomposition", "llm_answer_complex")
    g.add_edge("llm_answer_complex", "__end__")

    return g.compile()
